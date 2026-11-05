"""
response_generation.py
==================================================================
 STAGE B OF RLHF: RESPONSE GENERATION / SAMPLING
==================================================================

Pipeline position:
    Stage 0  -> SFT (sft_training.py)
    Stage A  -> Reward model training (reward_model_training.py)
    Stage B  -> Response generation / sampling (THIS FILE)
    Stage B.5-> Preference collection (human or RLAIF labeling ->
                the {"prompt","chosen","rejected"} JSONL consumed
                by Stage A)
    Stage C  -> PPO policy fine-tuning (ppo_rlhf_training.py)

Goal:
    Take the SFT checkpoint and a set of prompts, and generate
    MULTIPLE candidate completions per prompt. This is the raw
    material preference labeling (Stage B.5) operates on — you
    cannot ask "which response is better" with only one response.

Why multiple samples, and why sampling (not greedy):
    - Greedy / low-temperature decoding collapses toward one mode
      of the distribution -> completions for the same prompt end
      up near-identical, which gives labelers (human or LLM judge)
      nothing to discriminate between. The reward model then learns
      from a preference signal that's mostly noise.
    - Too high temperature/top_p -> completions diverge into
      incoherent text that's unlabelable for a different reason.
    - This script defaults to top_p=0.9, temperature=0.9 as a
      starting point that usually gives discriminable-but-coherent
      pairs; both are exposed as config for students to sweep.

Output format (JSONL), one object per PROMPT (not per completion):
    {"prompt": "...", "responses": ["...", "...", ...]}

This is the file Stage B.5 (preference collection) reads and turns
into {"prompt", "chosen", "rejected"} pairs for Stage A.
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
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

logger = logging.getLogger("response_generation")


# ==================================================================
# Configuration
# ==================================================================
@dataclass
class GenerationConfig:
    model_name: str            # your SFT checkpoint
    prompt_file: str
    output_file: str = "./generated_completions.jsonl"

    num_samples_per_prompt: int = 4   # how many completions to draw per prompt
    max_prompt_length: int = 512
    max_new_tokens: int = 256

    temperature: float = 0.9
    top_p: float = 0.9
    top_k: int = 0                    # 0 disables top-k filtering

    batch_size: int = 8                # PROMPTS per batch (not completions)
    min_unique_responses: int = 2      # warn if fewer unique completions survive dedup

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
# Data: prompts only (same format PPO's Stage C expects as input)
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
# Generation: batched sampling, num_samples_per_prompt completions each
# ==================================================================
def _dedup_preserve_order(responses: list[str]) -> list[str]:
    """Exact-match dedup. Near-duplicate detection (e.g. edit-distance
    or embedding similarity) is deliberately NOT done here — that's a
    judgment call best left to Stage B.5 labeling, since "too similar
    to be worth labeling" is itself a threshold worth making visible
    and tunable there, not silently baked into generation."""
    seen: set[str] = set()
    unique: list[str] = []
    for r in responses:
        key = r.strip()
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique


@torch.no_grad()
def generate_for_batch(
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    config: GenerationConfig,
    device: torch.device,
) -> list[list[str]]:
    """Returns, for each prompt in the batch, a list of
    num_samples_per_prompt decoded completion strings (pre-dedup)."""

    prompt_texts = [
        tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
        for p in prompts
    ]
    encoded = tokenizer(
        prompt_texts, padding=True, truncation=True, max_length=config.max_prompt_length,
        return_tensors="pt", add_special_tokens=False,
    ).to(device)
    prompt_len = encoded["input_ids"].shape[1]

    # num_return_sequences handles the "N samples per prompt" replication
    # internally (HF expands the batch under the hood) — cheaper and less
    # error-prone than manually repeating input_ids ourselves.
    generated = model.generate(
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        max_new_tokens=config.max_new_tokens,
        do_sample=True,
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=config.top_k if config.top_k > 0 else None,
        num_return_sequences=config.num_samples_per_prompt,
        pad_token_id=tokenizer.pad_token_id,
    )

    # generated shape: (len(prompts) * num_samples_per_prompt, seq_len)
    # Rows are grouped contiguously per prompt: [p0_s0, p0_s1, ..., p1_s0, ...]
    response_ids = generated[:, prompt_len:]
    decoded = tokenizer.batch_decode(response_ids, skip_special_tokens=True)

    grouped: list[list[str]] = []
    n = config.num_samples_per_prompt
    for i in range(len(prompts)):
        grouped.append(decoded[i * n:(i + 1) * n])
    return grouped


def save_jsonl(records: list[dict[str, Any]], path: str) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("Wrote %d prompt groups to %s", len(records), path)


# ==================================================================
# Driver
# ==================================================================
def run_generation(config: GenerationConfig) -> list[dict[str, Any]]:
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # required for batched generation with a causal LM

    model = AutoModelForCausalLM.from_pretrained(config.model_name, torch_dtype=config.torch_dtype()).to(device)
    model.eval()

    prompts = load_prompts(config.prompt_file)
    prompt_ds = PromptDataset(prompts)
    prompt_loader = DataLoader(
        prompt_ds, batch_size=config.batch_size, shuffle=False,
        collate_fn=lambda batch: batch, num_workers=config.num_workers,
    )

    results: list[dict[str, Any]] = []
    low_diversity_count = 0

    for batch_idx, batch_prompts in enumerate(prompt_loader):
        grouped = generate_for_batch(model, tokenizer, batch_prompts, config, device)

        for prompt, responses in zip(batch_prompts, grouped):
            unique_responses = _dedup_preserve_order(responses)
            if len(unique_responses) < config.min_unique_responses:
                low_diversity_count += 1
                logger.warning(
                    "Prompt produced only %d unique completion(s) out of %d samples "
                    "(min_unique_responses=%d) — consider raising temperature/top_p. Prompt: %.80r",
                    len(unique_responses), config.num_samples_per_prompt,
                    config.min_unique_responses, prompt,
                )
            results.append({"prompt": prompt, "responses": responses})

        if (batch_idx + 1) % 10 == 0:
            logger.info("Generated completions for %d/%d prompts", len(results), len(prompts))

    if low_diversity_count:
        logger.warning(
            "%d/%d prompts (%.1f%%) had fewer than %d unique completions after dedup. "
            "High duplication rate suggests sampling is too conservative for useful "
            "preference labeling downstream (Stage B.5).",
            low_diversity_count, len(prompts), 100 * low_diversity_count / len(prompts),
            config.min_unique_responses,
        )

    save_jsonl(results, config.output_file)
    return results


# ==================================================================
# CLI entry point
# ==================================================================
def parse_args() -> GenerationConfig:
    parser = argparse.ArgumentParser(description="Stage B: response generation/sampling for RLHF")
    parser.add_argument("--model_name", required=True, help="Path to your SFT checkpoint")
    parser.add_argument("--prompt_file", required=True)
    parser.add_argument("--output_file", default="./generated_completions.jsonl")
    parser.add_argument("--num_samples_per_prompt", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    args = parser.parse_args()
    return GenerationConfig(**vars(args))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    config = parse_args()
    logger.info("Starting Stage B response generation with config: %s", config)
    run_generation(config)


if __name__ == "__main__":
    main()