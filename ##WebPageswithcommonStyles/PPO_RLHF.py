"""
ppo_rlhf_training.py
==================================================================
 STAGE B OF RLHF: PPO POLICY FINE-TUNING
==================================================================

Inputs to this stage:
    - policy_model_name : your SFT checkpoint (this gets updated)
    - reward_model_path : the frozen RM trained by reward_model_training.py
    - a set of PROMPTS ONLY (no responses needed — the policy
      generates its own responses, which is the whole point of RL)

The RLHF PPO loop, at a glance:

    for each batch of prompts:
        1. ROLLOUT   : policy generates a response for each prompt
        2. SCORE     : frozen reward model scores each (prompt, response)
        3. KL PENALTY: subtract KL(policy || reference) per-token, so
                       the policy can't drift arbitrarily far from the
                       SFT model just to exploit the reward model
        4. ADVANTAGE : value head estimates a baseline; advantage =
                       reward - value (this reduces gradient variance)
        5. PPO UPDATE: clipped surrogate objective — take a step in
                       the reward-improving direction, but clip how
                       far any single update can move the policy

Why the clipping + KL penalty exist (the two ideas that matter most):
    - Reward models are imperfect proxies for human preference. An
      unconstrained optimizer will find degenerate outputs that score
      well on the RM but are bad text ("reward hacking").
    - The KL penalty anchors the policy near the SFT model.
    - The PPO clip prevents any single batch of noisy rollouts from
      making a catastrophically large policy update.

Expected data format (JSONL): one object per line, prompts only:
    {"prompt": "..."}
==================================================================
"""

from __future__ import annotations

import argparse
import copy
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
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from reward_model_training import RewardModel

logger = logging.getLogger("ppo_rlhf_training")


# ==================================================================
# Configuration
# ==================================================================
@dataclass
class PPOConfig:
    policy_model_name: str        # your SFT checkpoint
    reward_model_path: str        # dir saved by reward_model_training.py (contains reward_model.pt)
    prompt_file: str
    output_dir: str = "./ppo_checkpoints"

    max_prompt_length: int = 512
    max_new_tokens: int = 256

    total_ppo_steps: int = 500
    batch_size: int = 8
    ppo_epochs_per_batch: int = 4     # how many gradient passes over each rollout batch
    minibatch_size: int = 2

    learning_rate: float = 1e-6       # PPO uses a much smaller LR than SFT/RM
    clip_range: float = 0.2           # PPO clipping epsilon
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    kl_coef: float = 0.05             # weight on the KL(policy || reference) penalty
    max_grad_norm: float = 1.0
    gamma: float = 1.0                # discount — 1.0 is standard for single-turn response generation
    gae_lambda: float = 0.95

    save_every_steps: int = 100
    log_every_steps: int = 10
    seed: int = 42
    dtype: str = "bfloat16"

    def torch_dtype(self) -> torch.dtype:
        return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[self.dtype]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ==================================================================
# Policy model with an attached value head
# ==================================================================
class PolicyWithValueHead(nn.Module):
    """The policy IS the causal LM (produces token logits for
    generation) PLUS a small value head reading the same hidden
    states, used only as a baseline for advantage estimation."""

    def __init__(self, model_name: str, torch_dtype: torch.dtype) -> None:
        super().__init__()
        self.lm = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch_dtype, output_hidden_states=True
        )
        hidden_size = self.lm.config.hidden_size
        self.value_head = nn.Linear(hidden_size, 1, dtype=torch_dtype)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        outputs = self.lm(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.hidden_states[-1]           # (batch, seq_len, hidden)
        values = self.value_head(last_hidden).squeeze(-1)  # (batch, seq_len)
        return outputs.logits, values

    def generate(self, *args, **kwargs):
        return self.lm.generate(*args, **kwargs)


# ==================================================================
# Data: prompts only
# ==================================================================
def load_prompts(path: str) -> list[str]:
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")

    prompts: list[str] = []
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
            if "prompt" not in record:
                logger.warning("Skipping record missing 'prompt' at %s:%d", path, line_num)
                continue
            prompts.append(record["prompt"])

    if not prompts:
        raise ValueError(f"No valid prompts loaded from {path}")
    logger.info("Loaded %d prompts from %s", len(prompts), path)
    return prompts


class PromptDataset(Dataset):
    def __init__(self, prompts: list[str]) -> None:
        self.prompts = prompts

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> str:
        return self.prompts[idx]


# ==================================================================
# Rollout: generate responses + score them
# ==================================================================
@dataclass
class Rollout:
    input_ids: torch.Tensor        # prompt + response tokens, padded
    attention_mask: torch.Tensor
    response_mask: torch.Tensor    # 1 where token is part of the RESPONSE (not prompt, not pad)
    rewards: torch.Tensor          # scalar reward model score per sequence
    old_log_probs: torch.Tensor    # log pi_old(token) for each response token, under the policy AT ROLLOUT TIME
    ref_log_probs: torch.Tensor    # log pi_ref(token) for each response token, under the frozen SFT reference
    values: torch.Tensor           # value head estimate per response token, at rollout time


def compute_log_probs(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """log pi(token_t | tokens_<t) for each position, shifted by one
    exactly as in standard causal-LM next-token loss computation."""
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
    targets = input_ids[:, 1:]
    return torch.gather(log_probs, dim=2, index=targets.unsqueeze(-1)).squeeze(-1)


@torch.no_grad()
def generate_rollout(
    policy: PolicyWithValueHead,
    reference: AutoModelForCausalLM,
    reward_model: RewardModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    config: PPOConfig,
    device: torch.device,
) -> Rollout:
    policy.eval()

    prompt_texts = [
        tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
        for p in prompts
    ]
    encoded = tokenizer(
        prompt_texts, padding=True, truncation=True, max_length=config.max_prompt_length,
        return_tensors="pt", add_special_tokens=False,
    ).to(device)
    prompt_len = encoded["input_ids"].shape[1]

    # --- Step 1: rollout (sample, don't greedy-decode — PPO needs stochasticity) ---
    generated = policy.generate(
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        max_new_tokens=config.max_new_tokens,
        do_sample=True,
        top_p=0.9,
        temperature=1.0,
        pad_token_id=tokenizer.pad_token_id,
    )
    attention_mask = (generated != tokenizer.pad_token_id).long()
    response_mask = attention_mask.clone()
    response_mask[:, :prompt_len] = 0  # only response tokens count toward PPO loss / KL

    # --- Step 2: reward model scores the full (prompt, response) sequence ---
    rewards = reward_model(generated, attention_mask)  # (batch,) — one scalar per sequence

    # --- Step 3: log-probs under policy (old) and frozen reference, for the KL penalty ---
    policy_logits, values = policy(generated, attention_mask)
    old_log_probs = compute_log_probs(policy_logits, generated)

    ref_logits = reference(input_ids=generated, attention_mask=attention_mask).logits
    ref_log_probs = compute_log_probs(ref_logits, generated)

    policy.train()
    return Rollout(
        input_ids=generated,
        attention_mask=attention_mask,
        response_mask=response_mask[:, 1:],  # aligned with the shifted log-prob tensors
        rewards=rewards,
        old_log_probs=old_log_probs,
        ref_log_probs=ref_log_probs,
        values=values[:, :-1],
    )


# ==================================================================
# Advantage estimation (GAE) with the KL penalty folded into the reward
# ==================================================================
def compute_advantages(rollout: Rollout, config: PPOConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Per-token reward = -kl_coef * KL(policy || reference) at every
    step, PLUS the scalar RM reward added ONLY at the final response
    token (the RM only scores complete sequences).

    Then standard GAE over that per-token reward sequence, masked to
    response tokens only.
    """
    kl = rollout.old_log_probs - rollout.ref_log_probs          # (batch, seq_len-1)
    per_token_reward = -config.kl_coef * kl * rollout.response_mask

    # add the terminal RM reward at the LAST response token of each sequence
    last_response_idx = rollout.response_mask.sum(dim=1).long() - 1
    batch_idx = torch.arange(per_token_reward.size(0), device=per_token_reward.device)
    per_token_reward[batch_idx, last_response_idx] += rollout.rewards

    values = rollout.values
    advantages = torch.zeros_like(per_token_reward)
    last_gae = torch.zeros(per_token_reward.size(0), device=per_token_reward.device)

    seq_len = per_token_reward.size(1)
    for t in reversed(range(seq_len)):
        next_value = values[:, t + 1] if t + 1 < seq_len else torch.zeros_like(values[:, 0])
        delta = per_token_reward[:, t] + config.gamma * next_value - values[:, t]
        last_gae = delta + config.gamma * config.gae_lambda * last_gae * rollout.response_mask[:, t]
        advantages[:, t] = last_gae * rollout.response_mask[:, t]

    returns = advantages + values
    # Normalize advantages over response tokens only — standard PPO variance reduction
    mask = rollout.response_mask.bool()
    if mask.sum() > 1:
        adv_mean, adv_std = advantages[mask].mean(), advantages[mask].std()
        advantages = (advantages - adv_mean) / (adv_std + 1e-8)
    return advantages.detach(), returns.detach()


# ==================================================================
# PPO clipped update
# ==================================================================
def ppo_update(
    policy: PolicyWithValueHead,
    optimizer: torch.optim.Optimizer,
    rollout: Rollout,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    config: PPOConfig,
    device: torch.device,
) -> dict[str, float]:
    batch_size = rollout.input_ids.size(0)
    indices = torch.randperm(batch_size)
    stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0}
    n_updates = 0

    for _ in range(config.ppo_epochs_per_batch):
        for start in range(0, batch_size, config.minibatch_size):
            mb_idx = indices[start:start + config.minibatch_size]

            logits, values = policy(rollout.input_ids[mb_idx], rollout.attention_mask[mb_idx])
            log_probs = compute_log_probs(logits, rollout.input_ids[mb_idx])
            mask = rollout.response_mask[mb_idx]

            # PPO clipped surrogate objective
            log_ratio = log_probs - rollout.old_log_probs[mb_idx]
            ratio = torch.exp(log_ratio)
            adv = advantages[mb_idx]

            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - config.clip_range, 1 + config.clip_range) * adv
            policy_loss = -(torch.min(surr1, surr2) * mask).sum() / mask.sum().clamp(min=1)

            # Value loss (clipped, matching the PPO2 convention)
            values_pred = values[:, :-1]
            value_loss = 0.5 * (((values_pred - returns[mb_idx]) ** 2) * mask).sum() / mask.sum().clamp(min=1)

            # Entropy bonus encourages continued exploration, offsetting the KL penalty's pull toward the reference
            probs = F.softmax(logits[:, :-1, :], dim=-1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(-1)
            entropy = (entropy * mask).sum() / mask.sum().clamp(min=1)

            loss = policy_loss + config.value_loss_coef * value_loss - config.entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), config.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                approx_kl = ((ratio - 1) - log_ratio)  # unbiased KL estimator (Schulman et al.)
                approx_kl = (approx_kl * mask).sum() / mask.sum().clamp(min=1)

            stats["policy_loss"] += policy_loss.item()
            stats["value_loss"] += value_loss.item()
            stats["entropy"] += entropy.item()
            stats["approx_kl"] += approx_kl.item()
            n_updates += 1

    return {k: v / max(n_updates, 1) for k, v in stats.items()}


# ==================================================================
# Checkpointing
# ==================================================================
def save_checkpoint(policy: PolicyWithValueHead, tokenizer: PreTrainedTokenizerBase, output_dir: str, tag: str) -> str:
    checkpoint_dir = os.path.join(output_dir, tag)
    os.makedirs(checkpoint_dir, exist_ok=True)
    policy.lm.save_pretrained(checkpoint_dir)          # the actual generative model
    torch.save(policy.value_head.state_dict(), os.path.join(checkpoint_dir, "value_head.pt"))
    tokenizer.save_pretrained(checkpoint_dir)
    logger.info("Saved PPO policy checkpoint to %s", checkpoint_dir)
    return checkpoint_dir


# ==================================================================
# Training driver
# ==================================================================
def train_ppo(config: PPOConfig) -> PolicyWithValueHead:
    set_seed(config.seed)
    os.makedirs(config.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    tokenizer = AutoTokenizer.from_pretrained(config.policy_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # required for batched generation with a causal LM

    policy = PolicyWithValueHead(config.policy_model_name, config.torch_dtype()).to(device)

    # Reference model: frozen copy of the SFT model, used only for the KL penalty.
    reference = AutoModelForCausalLM.from_pretrained(
        config.policy_model_name, torch_dtype=config.torch_dtype()
    ).to(device)
    reference.eval()
    for param in reference.parameters():
        param.requires_grad = False

    # Reward model: frozen, loaded from Stage A's checkpoint.
    reward_model = RewardModel(config.policy_model_name, config.torch_dtype()).to(device)
    reward_state = torch.load(os.path.join(config.reward_model_path, "reward_model.pt"), map_location=device)
    reward_model.load_state_dict(reward_state)
    reward_model.eval()
    for param in reward_model.parameters():
        param.requires_grad = False

    prompts = load_prompts(config.prompt_file)
    prompt_ds = PromptDataset(prompts)
    prompt_loader = DataLoader(
        prompt_ds, batch_size=config.batch_size, shuffle=True,
        collate_fn=lambda batch: batch,  # keep as list[str]; tokenization happens in generate_rollout
    )

    optimizer = torch.optim.AdamW(policy.parameters(), lr=config.learning_rate)

    global_step = 0
    prompt_iter = iter(prompt_loader)

    try:
        while global_step < config.total_ppo_steps:
            try:
                batch_prompts = next(prompt_iter)
            except StopIteration:
                prompt_iter = iter(prompt_loader)
                batch_prompts = next(prompt_iter)

            rollout = generate_rollout(policy, reference, reward_model, tokenizer, batch_prompts, config, device)
            advantages, returns = compute_advantages(rollout, config)
            stats = ppo_update(policy, optimizer, rollout, advantages, returns, config, device)

            global_step += 1
            if global_step % config.log_every_steps == 0:
                mean_reward = rollout.rewards.mean().item()
                logger.info(
                    "step %d | mean_reward=%.4f policy_loss=%.4f value_loss=%.4f "
                    "entropy=%.4f approx_kl=%.5f",
                    global_step, mean_reward, stats["policy_loss"], stats["value_loss"],
                    stats["entropy"], stats["approx_kl"],
                )

            if global_step % config.save_every_steps == 0:
                save_checkpoint(policy, tokenizer, config.output_dir, f"step_{global_step}")

    except KeyboardInterrupt:
        logger.warning("PPO training interrupted; saving current state")
        save_checkpoint(policy, tokenizer, config.output_dir, "interrupted")
        raise

    save_checkpoint(policy, tokenizer, config.output_dir, "final")
    return policy


# ==================================================================
# CLI entry point
# ==================================================================
def parse_args() -> PPOConfig:
    parser = argparse.ArgumentParser(description="PPO policy fine-tuning for RLHF")
    parser.add_argument("--policy_model_name", required=True, help="Path to your SFT checkpoint")
    parser.add_argument("--reward_model_path", required=True, help="Dir from reward_model_training.py")
    parser.add_argument("--prompt_file", required=True)
    parser.add_argument("--output_dir", default="./ppo_checkpoints")
    parser.add_argument("--total_ppo_steps", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--kl_coef", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    return PPOConfig(**vars(args))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    config = parse_args()
    logger.info("Starting PPO RLHF training with config: %s", config)
    train_ppo(config)


if __name__ == "__main__":
    main()