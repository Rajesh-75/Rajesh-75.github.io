"""
convert_pretrained_checkpoint.py
==================================================================
 BRIDGE: raw pretraining checkpoint -> HuggingFace save_pretrained() format
==================================================================

Why this exists:
    llm_pretraining.py saves checkpoints as a raw torch.save() dict:
        {"model": state_dict, "optimizer": ..., "scheduler": ..., "step": ...}

    sft_training.py (and reward_model_training.py, ppo_rlhf_training.py)
    all load their starting model with:
        AutoModelForCausalLM.from_pretrained(model_name)

    That call expects a DIRECTORY containing config.json + weight files,
    which is NOT what llm_pretraining.py produces. Without this
    conversion step, pretraining's output cannot be handed to SFT.

What it does:
    1. Rebuilds the exact GPT2Config/GPT2LMHeadModel architecture used
       during pretraining (you supply the same architecture flags you
       trained with).
    2. Loads the state_dict from the raw checkpoint into it.
    3. Calls model.save_pretrained() / tokenizer.save_pretrained() to
       produce a directory AutoModelForCausalLM.from_pretrained() can
       load directly.

Usage:
    python convert_pretrained_checkpoint.py \
        --checkpoint ./pretrain_ckpts/ckpt_step100000.pt \
        --tokenizer_name gpt2 \
        --context_length 1024 --n_layer 12 --n_head 12 --n_embd 768 \
        --output_dir ./pretrain_ckpts/hf_format
==================================================================
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

logger = logging.getLogger("convert_pretrained_checkpoint")
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"


@dataclass(frozen=True)
class ConvertConfig:
    checkpoint: Path
    tokenizer_name: str
    context_length: int
    n_layer: int
    n_head: int
    n_embd: int
    output_dir: Path


def parse_args(argv: list[str] | None = None) -> ConvertConfig:
    parser = argparse.ArgumentParser(
        description="Convert a raw llm_pretraining.py checkpoint to HF save_pretrained() format"
    )
    parser.add_argument("--checkpoint", required=True, help="Path to ckpt_step*.pt")
    parser.add_argument("--tokenizer_name", required=True, help="Must match the tokenizer used in pretraining")
    parser.add_argument("--context_length", type=int, required=True)
    parser.add_argument("--n_layer", type=int, required=True)
    parser.add_argument("--n_head", type=int, required=True)
    parser.add_argument("--n_embd", type=int, required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        parser.error(f"Checkpoint not found: {checkpoint}")

    return ConvertConfig(
        checkpoint=checkpoint,
        tokenizer_name=args.tokenizer_name,
        context_length=args.context_length,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        output_dir=Path(args.output_dir),
    )


def convert(config: ConvertConfig) -> Path:
    """Rebuild the pretrained model architecture, load its weights, and
    write it out in a format downstream stages (SFT, RM, PPO) can load
    via AutoModelForCausalLM.from_pretrained().

    Returns:
        Path to the written HF-format directory.

    Raises:
        RuntimeError: If the checkpoint is malformed or the tokenizer
            cannot be loaded.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to load tokenizer '{config.tokenizer_name}': {exc}") from exc
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    gpt2_config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=config.context_length,
        n_ctx=config.context_length,
        n_embd=config.n_embd,
        n_layer=config.n_layer,
        n_head=config.n_head,
    )
    model = GPT2LMHeadModel(gpt2_config)

    try:
        ckpt = torch.load(config.checkpoint, map_location="cpu")
        model.load_state_dict(ckpt["model"])
    except (KeyError, RuntimeError) as exc:
        raise RuntimeError(
            f"Checkpoint at {config.checkpoint} is malformed or doesn't match the "
            f"architecture flags given (n_layer/n_head/n_embd/context_length): {exc}"
        ) from exc

    config.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    logger.info(
        "Converted step %s checkpoint -> HF format at %s", ckpt.get("step", "?"), config.output_dir
    )
    return config.output_dir


def main(argv: list[str] | None = None) -> int:
    config = parse_args(argv)
    try:
        convert(config)
    except RuntimeError as exc:
        logger.error("Conversion failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())