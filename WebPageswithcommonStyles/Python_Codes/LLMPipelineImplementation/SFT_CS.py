"""
sft_training.py
==================================================================
 SUPERVISED FINE-TUNING (SFT) TRAINING ALGORITHM
==================================================================

Trains a pretrained causal LLM on (prompt, response) pairs, with
loss computed only over response tokens (prompt tokens masked to
IGNORE_INDEX).

Usage:
    python sft_training.py --model_name meta-llama/Llama-3.2-1B \
                            --train_file data/train.jsonl \
                            --val_file data/val.jsonl \
                            --output_dir ./sft_checkpoints

Expected data format (JSONL), one object per line:
    {"prompt": "...", "response": "..."}
==================================================================
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    get_cosine_schedule_with_warmup,
)

logger = logging.getLogger("sft_training")

IGNORE_INDEX: int = -100  # torch.nn.CrossEntropyLoss default ignore_index


# ==================================================================
# Configuration
# ==================================================================
@dataclass
class SFTConfig:
    """All hyperparameters and paths for an SFT run, in one place
    so a run is fully reproducible from a single object / CLI call."""

    model_name: str
    train_file: str
    val_file: str
    output_dir: str = "./sft_checkpoints"

    max_length: int = 1024
    epochs: int = 3
    batch_size: int = 4
    grad_accum_steps: int = 8          # effective batch = batch_size * grad_accum_steps
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0

    eval_every_steps: int = 200
    save_every_epoch: bool = True
    keep_best_only: bool = False       # if True, only keep checkpoint with lowest val_loss

    seed: int = 42
    dtype: str = "bfloat16"            # "float32" | "float16" | "bfloat16"
    gradient_checkpointing: bool = True
    num_workers: int = 2

    def torch_dtype(self) -> torch.dtype:
        return {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[self.dtype]


# ==================================================================
# Reproducibility
# ==================================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ==================================================================
# Data loading
# ==================================================================
def load_jsonl(path: str) -> list[dict[str, Any]]:
    """Load a JSONL file of {"prompt": ..., "response": ...} records,
    skipping and logging malformed lines instead of crashing the run."""
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
            if "prompt" not in record or "response" not in record:
                logger.warning("Skipping record missing prompt/response at %s:%d", path, line_num)
                continue
            examples.append(record)

    if not examples:
        raise ValueError(f"No valid examples loaded from {path}")

    logger.info("Loaded %d examples from %s", len(examples), path)
    return examples


# ==================================================================
# Dataset: formatting + tokenization + loss masking
# ==================================================================
class SFTDataset(Dataset):
    """Tokenizes (prompt, response) pairs into a single sequence and
    masks prompt tokens out of the loss via IGNORE_INDEX labels.

        input_ids: [ <prompt tokens> | <response tokens> <eos> ]
        labels:    [ -100 ... -100   | <response tokens> <eos> ]
    """

    def __init__(
        self,
        examples: list[dict[str, Any]],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 1024,
    ) -> None:
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        example = self.examples[idx]

        prompt_text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": example["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        response_text = example["response"] + self.tokenizer.eos_token

        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        response_ids = self.tokenizer(response_text, add_special_tokens=False)["input_ids"]

        input_ids = (prompt_ids + response_ids)[: self.max_length]
        labels = ([IGNORE_INDEX] * len(prompt_ids) + response_ids)[: self.max_length]

        if len(input_ids) == len(prompt_ids) and len(prompt_ids) >= self.max_length:
            # Prompt alone exceeded max_length: response was fully truncated away.
            logger.warning("Example %d: prompt exceeds max_length, response fully truncated", idx)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def build_collate_fn(pad_token_id: int):
    """Returns a collate function that dynamically pads each batch to
    its own longest sequence (not a fixed global max_length)."""

    def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        max_len = max(len(x["input_ids"]) for x in batch)

        input_ids, labels, attention_mask = [], [], []
        for item in batch:
            pad_len = max_len - len(item["input_ids"])
            input_ids.append(F.pad(item["input_ids"], (0, pad_len), value=pad_token_id))
            labels.append(F.pad(item["labels"], (0, pad_len), value=IGNORE_INDEX))
            attention_mask.append(
                torch.cat([torch.ones(len(item["input_ids"])), torch.zeros(pad_len)])
            )

        return {
            "input_ids": torch.stack(input_ids),
            "labels": torch.stack(labels),
            "attention_mask": torch.stack(attention_mask).long(),
        }

    return collate_fn


# ==================================================================
# Evaluation
# ==================================================================
@torch.no_grad()
def evaluate(
    model: PreTrainedModel,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """Returns (mean token-level loss, perplexity) over the dataloader."""
    model.eval()
    total_loss, total_tokens = 0.0, 0

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        n_tokens = int((batch["labels"] != IGNORE_INDEX).sum().item())
        total_loss += outputs.loss.item() * n_tokens
        total_tokens += n_tokens

    model.train()
    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(min(avg_loss, 20))  # guard against overflow on bad checkpoints
    return avg_loss, perplexity


# ==================================================================
# Checkpointing
# ==================================================================
def save_checkpoint(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    output_dir: str,
    tag: str,
) -> str:
    checkpoint_dir = os.path.join(output_dir, tag)
    os.makedirs(checkpoint_dir, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    logger.info("Saved checkpoint to %s", checkpoint_dir)
    return checkpoint_dir


# ==================================================================
# Training driver
# ==================================================================
def train_sft(config: SFTConfig) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    set_seed(config.seed)
    os.makedirs(config.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # --- Tokenizer + model ---
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name, torch_dtype=config.torch_dtype()
    ).to(device)
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.train()

    # --- Data ---
    train_examples = load_jsonl(config.train_file)
    val_examples = load_jsonl(config.val_file)

    train_ds = SFTDataset(train_examples, tokenizer, config.max_length)
    val_ds = SFTDataset(val_examples, tokenizer, config.max_length)
    collate_fn = build_collate_fn(tokenizer.pad_token_id)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.num_workers,
    )

    # --- Optimizer + LR schedule ---
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    total_optim_steps = max((len(train_loader) // config.grad_accum_steps) * config.epochs, 1)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(config.warmup_ratio * total_optim_steps),
        num_training_steps=total_optim_steps,
    )

    # --- Training loop ---
    global_step = 0
    best_val_loss = float("inf")

    try:
        for epoch in range(config.epochs):
            running_loss = 0.0

            for step, batch in enumerate(train_loader):
                batch = {k: v.to(device) for k, v in batch.items()}

                outputs = model(**batch)
                loss = outputs.loss / config.grad_accum_steps
                loss.backward()
                running_loss += outputs.loss.item()

                if (step + 1) % config.grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    if global_step % config.eval_every_steps == 0:
                        val_loss, val_ppl = evaluate(model, val_loader, device)
                        logger.info(
                            "epoch %d step %d | train_loss=%.4f val_loss=%.4f val_ppl=%.2f",
                            epoch + 1, global_step, running_loss / (step + 1), val_loss, val_ppl,
                        )
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            if config.keep_best_only:
                                save_checkpoint(model, tokenizer, config.output_dir, "best")

            val_loss, val_ppl = evaluate(model, val_loader, device)
            logger.info("=== epoch %d/%d complete | val_loss=%.4f val_ppl=%.2f ===",
                        epoch + 1, config.epochs, val_loss, val_ppl)

            if config.save_every_epoch and not config.keep_best_only:
                save_checkpoint(model, tokenizer, config.output_dir, f"epoch_{epoch + 1}")

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user; saving current state before exit")
        save_checkpoint(model, tokenizer, config.output_dir, "interrupted")
        raise
    except torch.cuda.OutOfMemoryError:
        logger.error(
            "CUDA OOM at epoch %d step %d. Reduce batch_size or increase grad_accum_steps.",
            epoch + 1, step,
        )
        raise

    save_checkpoint(model, tokenizer, config.output_dir, "final")
    return model, tokenizer


# ==================================================================
# CLI entry point
# ==================================================================
def parse_args() -> SFTConfig:
    parser = argparse.ArgumentParser(description="Supervised fine-tuning (SFT) trainer")
    parser.add_argument("--model_name", required=True, help="HF model id or local path")
    parser.add_argument("--train_file", required=True, help="Path to train.jsonl")
    parser.add_argument("--val_file", required=True, help="Path to val.jsonl")
    parser.add_argument("--output_dir", default="./sft_checkpoints")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    args = parser.parse_args()
    return SFTConfig(**vars(args))


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    config = parse_args()
    logger.info("Starting SFT run with config: %s", config)
    train_sft(config)


if __name__ == "__main__":
    main()