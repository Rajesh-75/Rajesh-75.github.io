"""
reward_model_training.py
==================================================================
 STAGE A OF RLHF: REWARD MODEL (RM) TRAINING
==================================================================

Goal:
    Train a scalar reward model r_phi(prompt, response) -> float
    from HUMAN PREFERENCE DATA: for a given prompt, a human ranked
    one response as "chosen" and another as "rejected".

    The reward model is later used (frozen) inside PPO to score
    the policy's generations — it is a stand-in for "what would a
    human rate this response as", so PPO doesn't need a human in
    the loop for every single update.

Architecture:
    Take the SFT model, REPLACE its language-modeling head with a
    single linear layer -> 1 scalar. Feed it (prompt + response),
    read the scalar at the LAST non-padding token position.

Loss (Bradley-Terry preference model):
    P(chosen > rejected) = sigmoid(r(chosen) - r(rejected))
    loss = -log(sigmoid(r_chosen - r_rejected))

    This is standard pairwise ranking loss — we never need an
    absolute reward value, only that chosen scores higher than
    rejected for the same prompt.

Expected data format (JSONL), one object per line:
    {"prompt": "...", "chosen": "...", "rejected": "..."}
==================================================================
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModel,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    get_cosine_schedule_with_warmup,
)

logger = logging.getLogger("reward_model_training")


# ==================================================================
# Configuration
# ==================================================================
@dataclass
class RewardModelConfig:
    base_model_name: str          # typically your SFT checkpoint, not the raw base model
    train_file: str
    val_file: str
    output_dir: str = "./reward_model_checkpoints"

    max_length: int = 1024
    epochs: int = 1               # RM training typically needs far fewer epochs than SFT
    batch_size: int = 4
    grad_accum_steps: int = 4
    learning_rate: float = 1e-5   # lower than SFT — RM training is prone to overfitting
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0

    eval_every_steps: int = 100
    seed: int = 42
    dtype: str = "bfloat16"
    num_workers: int = 2

    def torch_dtype(self) -> torch.dtype:
        return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[self.dtype]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ==================================================================
# Model: base transformer + scalar reward head
# ==================================================================
class RewardModel(nn.Module):
    """Wraps a pretrained transformer body with a linear head that
    outputs one scalar per sequence (read at the last real token)."""

    def __init__(self, base_model_name: str, torch_dtype: torch.dtype) -> None:
        super().__init__()
        # AutoModel (not AutoModelForCausalLM) gives us hidden states
        # without a pretrained LM head we'd just be throwing away.
        self.backbone = AutoModel.from_pretrained(base_model_name, torch_dtype=torch_dtype)
        hidden_size = self.backbone.config.hidden_size
        self.reward_head = nn.Linear(hidden_size, 1, dtype=torch_dtype)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden)

        # Index of the last non-padding token per sequence
        last_token_idx = attention_mask.sum(dim=1) - 1  # (batch,)
        batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
        last_hidden = hidden_states[batch_idx, last_token_idx]  # (batch, hidden)

        reward = self.reward_head(last_hidden).squeeze(-1)  # (batch,)
        return reward


# ==================================================================
# Data
# ==================================================================
def load_jsonl(path: str) -> list[dict[str, Any]]:
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    examples: list[dict[str, Any]] = []
    with path_obj.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning("Skipping malformed JSON at %s:%d (%s)", path, line_num, e)
                continue
            if not all(k in record for k in ("prompt", "chosen", "rejected")):
                logger.warning("Skipping record missing prompt/chosen/rejected at %s:%d", path, line_num)
                continue
            examples.append(record)

    if not examples:
        raise ValueError(f"No valid examples loaded from {path}")
    logger.info("Loaded %d preference pairs from %s", len(examples), path)
    return examples


class PreferenceDataset(Dataset):
    """Each item yields a TOKENIZED (chosen_ids, rejected_ids) pair
    for the same prompt. Both share the prompt but diverge in the
    response — that divergence is exactly the training signal."""

    def __init__(self, examples: list[dict[str, Any]], tokenizer: PreTrainedTokenizerBase, max_length: int) -> None:
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def _encode(self, prompt: str, response: str) -> torch.Tensor:
        prompt_text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        full_text = prompt_text + response + self.tokenizer.eos_token
        ids = self.tokenizer(full_text, add_special_tokens=False, truncation=True, max_length=self.max_length)["input_ids"]
        return torch.tensor(ids, dtype=torch.long)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ex = self.examples[idx]
        return {
            "chosen_ids": self._encode(ex["prompt"], ex["chosen"]),
            "rejected_ids": self._encode(ex["prompt"], ex["rejected"]),
        }


def build_collate_fn(pad_token_id: int):
    def _pad_batch(sequences: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        max_len = max(len(s) for s in sequences)
        input_ids = torch.stack([F.pad(s, (0, max_len - len(s)), value=pad_token_id) for s in sequences])
        attention_mask = torch.stack(
            [torch.cat([torch.ones(len(s)), torch.zeros(max_len - len(s))]) for s in sequences]
        ).long()
        return input_ids, attention_mask

    def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        chosen_ids, chosen_mask = _pad_batch([b["chosen_ids"] for b in batch])
        rejected_ids, rejected_mask = _pad_batch([b["rejected_ids"] for b in batch])
        return {
            "chosen_ids": chosen_ids, "chosen_mask": chosen_mask,
            "rejected_ids": rejected_ids, "rejected_mask": rejected_mask,
        }

    return collate_fn


# ==================================================================
# Loss + evaluation
# ==================================================================
def pairwise_loss(chosen_rewards: torch.Tensor, rejected_rewards: torch.Tensor) -> torch.Tensor:
    """Bradley-Terry pairwise loss: -log(sigmoid(r_chosen - r_rejected))."""
    return -F.logsigmoid(chosen_rewards - rejected_rewards).mean()


@torch.no_grad()
def evaluate(model: RewardModel, dataloader: DataLoader, device: torch.device) -> tuple[float, float]:
    """Returns (mean loss, pairwise accuracy — how often chosen > rejected)."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        chosen_r = model(batch["chosen_ids"], batch["chosen_mask"])
        rejected_r = model(batch["rejected_ids"], batch["rejected_mask"])

        loss = pairwise_loss(chosen_r, rejected_r)
        total_loss += loss.item() * chosen_r.size(0)
        correct += (chosen_r > rejected_r).sum().item()
        total += chosen_r.size(0)

    model.train()
    return total_loss / max(total, 1), correct / max(total, 1)


def save_checkpoint(model: RewardModel, tokenizer: PreTrainedTokenizerBase, output_dir: str, tag: str) -> str:
    checkpoint_dir = os.path.join(output_dir, tag)
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, "reward_model.pt"))
    tokenizer.save_pretrained(checkpoint_dir)
    logger.info("Saved reward model checkpoint to %s", checkpoint_dir)
    return checkpoint_dir


# ==================================================================
# Training driver
# ==================================================================
def train_reward_model(config: RewardModelConfig) -> tuple[RewardModel, PreTrainedTokenizerBase]:
    set_seed(config.seed)
    os.makedirs(config.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = RewardModel(config.base_model_name, config.torch_dtype()).to(device)
    model.train()

    train_examples = load_jsonl(config.train_file)
    val_examples = load_jsonl(config.val_file)
    train_ds = PreferenceDataset(train_examples, tokenizer, config.max_length)
    val_ds = PreferenceDataset(val_examples, tokenizer, config.max_length)
    collate_fn = build_collate_fn(tokenizer.pad_token_id)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True,
                               collate_fn=collate_fn, num_workers=config.num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=config.num_workers)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    total_steps = max((len(train_loader) // config.grad_accum_steps) * config.epochs, 1)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=int(config.warmup_ratio * total_steps), num_training_steps=total_steps
    )

    global_step = 0
    try:
        for epoch in range(config.epochs):
            running_loss = 0.0
            for step, batch in enumerate(train_loader):
                batch = {k: v.to(device) for k, v in batch.items()}

                chosen_r = model(batch["chosen_ids"], batch["chosen_mask"])
                rejected_r = model(batch["rejected_ids"], batch["rejected_mask"])
                loss = pairwise_loss(chosen_r, rejected_r) / config.grad_accum_steps
                loss.backward()
                running_loss += loss.item() * config.grad_accum_steps

                if (step + 1) % config.grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    if global_step % config.eval_every_steps == 0:
                        val_loss, val_acc = evaluate(model, val_loader, device)
                        logger.info(
                            "epoch %d step %d | train_loss=%.4f val_loss=%.4f val_pairwise_acc=%.3f",
                            epoch + 1, global_step, running_loss / (step + 1), val_loss, val_acc,
                        )

            val_loss, val_acc = evaluate(model, val_loader, device)
            logger.info("=== epoch %d/%d complete | val_loss=%.4f val_pairwise_acc=%.3f ===",
                        epoch + 1, config.epochs, val_loss, val_acc)
            save_checkpoint(model, tokenizer, config.output_dir, f"epoch_{epoch + 1}")

    except KeyboardInterrupt:
        logger.warning("Training interrupted; saving current state")
        save_checkpoint(model, tokenizer, config.output_dir, "interrupted")
        raise

    save_checkpoint(model, tokenizer, config.output_dir, "final")
    return model, tokenizer


# ==================================================================
# CLI entry point
# ==================================================================
def parse_args() -> RewardModelConfig:
    parser = argparse.ArgumentParser(description="Reward model (RM) trainer for RLHF")
    parser.add_argument("--base_model_name", required=True, help="Path to your SFT checkpoint")
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--val_file", required=True)
    parser.add_argument("--output_dir", default="./reward_model_checkpoints")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    return RewardModelConfig(**vars(args))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    config = parse_args()
    logger.info("Starting reward model training with config: %s", config)
    train_reward_model(config)


if __name__ == "__main__":
    main()