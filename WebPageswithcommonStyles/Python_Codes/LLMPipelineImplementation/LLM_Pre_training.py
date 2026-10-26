"""
llm_pretraining.py
===================
Industry-standard pretraining script for a decoder-only (GPT-style) causal
language model. This is the FIRST stage of the LLM training pipeline —
the model here learns raw next-token prediction over a large unlabeled
corpus, before Supervised Fine-Tuning (SFT) and RLHF/alignment are applied
on top of these pretrained weights.

Pipeline position:
    [ PRETRAINING (this script) ] -> [ SFT ] -> [ RLHF / DPO ] -> [ Inference ]

Design choices, and why they're "industry standard":
  - Model & config come from HuggingFace `transformers` (the same library
    used to define/pretrain models like GPT-2, GPT-Neo, Llama, etc.),
    rather than a hand-rolled Transformer, so architecture details
    (position embeddings, layer norm placement, attention masking) match
    production implementations.
  - Streaming HF `datasets` for the corpus — real pretraining corpora
    (web-scale text) are far too large to load into memory at once.
  - Mixed precision (bf16/fp16) + gradient accumulation + gradient
    clipping — standard techniques to fit large batch sizes on limited
    GPU memory while keeping training numerically stable.
  - Cosine learning-rate schedule with linear warmup — the schedule used
    by GPT-2/3, LLaMA, and most modern pretraining runs.
  - Checkpointing + resumption — pretraining runs span days/weeks and
    must survive interruption, including a graceful save on Ctrl-C.
  - The training loop is written explicitly (not hidden behind
    `Trainer.train()`) so every step maps onto the pretraining
    pseudo-algorithm taught alongside this script.

Coding standards applied (PEP 8 / PEP 257 / industry conventions):
  - Full type hints on every function signature.
  - Docstrings (Google style) on every public function.
  - `logging` module instead of bare `print` for all run-time output,
    with configurable verbosity.
  - Immutable `@dataclass` config instead of a loose argparse.Namespace,
    so the training config is a typed, IDE-inspectable object.
  - `pathlib.Path` instead of raw string path manipulation.
  - Explicit exception handling around I/O (dataset loading, checkpoint
    save/load) with actionable error messages, and a graceful
    KeyboardInterrupt handler that checkpoints before exiting.
  - Fixed random seed for reproducibility.
  - No mutable default arguments; no magic numbers (named constants).
  - Module is import-safe: no top-level side effects outside `main()`.

Usage:
    python llm_pretraining.py \
        --dataset_name wikitext --dataset_config wikitext-103-raw-v1 \
        --tokenizer_name gpt2 --context_length 1024 \
        --n_layer 12 --n_head 12 --n_embd 768 \
        --batch_size 8 --grad_accum_steps 4 --max_steps 100000

Requirements (pin in requirements.txt for reproducible environments):
    torch>=2.2
    transformers>=4.40
    datasets>=2.19
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional, Tuple

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    GPT2Config,
    GPT2LMHeadModel,
    PreTrainedTokenizerBase,
    get_cosine_schedule_with_warmup,
)

# --------------------------------------------------------------------------- #
# Constants (no magic numbers scattered through the logic below)
# --------------------------------------------------------------------------- #
DEFAULT_SEED: int = 42
LOSS_PPL_OVERFLOW_GUARD: float = 20.0  # exp(20) is already astronomically large
LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(message)s"

logger = logging.getLogger("llm_pretraining")


# --------------------------------------------------------------------------- #
# 1. Configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TrainingConfig:
    """Typed, immutable configuration for a pretraining run.

    Using a frozen dataclass (rather than a raw argparse.Namespace) gives
    IDE autocompletion, type checking, and prevents accidental mutation
    of the config mid-run.
    """

    # Data
    dataset_name: str
    dataset_config: Optional[str]
    tokenizer_name: str
    context_length: int

    # Model architecture (trained from scratch — this run IS the pretraining)
    n_layer: int
    n_head: int
    n_embd: int

    # Optimization
    batch_size: int
    grad_accum_steps: int
    max_steps: int
    warmup_steps: int
    lr: float
    weight_decay: float
    max_grad_norm: float
    betas: Tuple[float, float] = field(default=(0.9, 0.95))

    # Logging / checkpointing
    output_dir: Path = field(default=Path("./checkpoints"))
    log_every: int = 50
    save_every: int = 2_000
    resume_from: Optional[Path] = None
    seed: int = DEFAULT_SEED


def parse_args(argv: Optional[list[str]] = None) -> TrainingConfig:
    """Parse command-line arguments into a validated TrainingConfig.

    Args:
        argv: Optional explicit argument list (mainly for testing);
            defaults to `sys.argv[1:]` when None.

    Returns:
        A populated, immutable TrainingConfig instance.
    """
    parser = argparse.ArgumentParser(description="Pretrain a decoder-only causal LM")

    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--dataset_config", type=str, default=None)
    parser.add_argument("--tokenizer_name", type=str, default="gpt2")
    parser.add_argument("--context_length", type=int, default=1024)

    parser.add_argument("--n_layer", type=int, default=12)
    parser.add_argument("--n_head", type=int, default=12)
    parser.add_argument("--n_embd", type=int, default=768)

    parser.add_argument("--batch_size", type=int, default=8, help="Per-device micro-batch size")
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=100_000)
    parser.add_argument("--warmup_steps", type=int, default=2_000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95))

    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=2_000)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--log_level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level), format=LOG_FORMAT)

    if args.batch_size < 1 or args.grad_accum_steps < 1:
        parser.error("batch_size and grad_accum_steps must both be >= 1")
    if args.max_steps < 1:
        parser.error("max_steps must be >= 1")

    return TrainingConfig(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        tokenizer_name=args.tokenizer_name,
        context_length=args.context_length,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        betas=tuple(args.betas),
        output_dir=Path(args.output_dir),
        log_every=args.log_every,
        save_every=args.save_every,
        resume_from=Path(args.resume_from) if args.resume_from else None,
        seed=args.seed,
    )


def set_seed(seed: int) -> None:
    """Seed all relevant RNGs for a reproducible run.

    Args:
        seed: Seed value applied to Python's random state indirectly via
            torch, and to CUDA if a GPU is available.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# 2. Data pipeline: stream, tokenize, chunk into fixed-length blocks
# --------------------------------------------------------------------------- #
def build_dataloader(config: TrainingConfig, tokenizer: PreTrainedTokenizerBase) -> DataLoader:
    """Build a streaming DataLoader that yields fixed-length token blocks.

    Args:
        config: The training configuration (uses dataset name/config and
            context_length).
        tokenizer: A HuggingFace tokenizer already loaded with a pad token.

    Returns:
        A torch DataLoader yielding batches of `input_ids`/`labels` tensors
        of shape (batch_size, context_length).

    Raises:
        RuntimeError: If the dataset cannot be loaded (e.g. bad name,
            network failure, missing "text" column).
    """
    try:
        raw = load_dataset(
            config.dataset_name, config.dataset_config, split="train", streaming=True
        )
    except Exception as exc:  # noqa: BLE001 - surfaced as an actionable RuntimeError
        raise RuntimeError(
            f"Failed to load dataset '{config.dataset_name}' "
            f"(config={config.dataset_config!r}): {exc}"
        ) from exc

    def tokenize(batch: dict) -> dict:
        return tokenizer(batch["text"], return_special_tokens_mask=False)

    tokenized = raw.map(tokenize, batched=True, remove_columns=["text"])

    block_size = config.context_length

    def group_texts(batch: dict) -> dict:
        """Concatenate token streams and split into equal-length blocks.

        This avoids wasting compute on padding, which is standard
        practice for pretraining (as opposed to SFT, where padding per
        example is normal).
        """
        concatenated: list[int] = sum(batch["input_ids"], [])
        total_len = (len(concatenated) // block_size) * block_size
        chunks = [concatenated[i : i + block_size] for i in range(0, total_len, block_size)]
        return {"input_ids": chunks}

    lm_dataset = tokenized.map(group_texts, batched=True)

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    return DataLoader(lm_dataset, batch_size=config.batch_size, collate_fn=collator)


def load_tokenizer(tokenizer_name: str) -> PreTrainedTokenizerBase:
    """Load a tokenizer and ensure it has a pad token defined.

    Args:
        tokenizer_name: HuggingFace hub name or local path.

    Returns:
        A ready-to-use tokenizer.

    Raises:
        RuntimeError: If the tokenizer cannot be loaded.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to load tokenizer '{tokenizer_name}': {exc}") from exc

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


# --------------------------------------------------------------------------- #
# 3. Model: decoder-only Transformer, initialized from scratch
# --------------------------------------------------------------------------- #
def build_model(config: TrainingConfig, vocab_size: int) -> nn.Module:
    """Instantiate a randomly-initialized decoder-only causal LM.

    Args:
        config: Training configuration supplying architecture hyperparameters.
        vocab_size: Tokenizer vocabulary size.

    Returns:
        A `GPT2LMHeadModel` with random initialization — this run's
        purpose is precisely to learn these weights from scratch.
    """
    gpt2_config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=config.context_length,
        n_ctx=config.context_length,
        n_embd=config.n_embd,
        n_layer=config.n_layer,
        n_head=config.n_head,
    )
    return GPT2LMHeadModel(gpt2_config)


# --------------------------------------------------------------------------- #
# 4. Checkpointing
# --------------------------------------------------------------------------- #
def save_checkpoint(
    output_dir: Path,
    step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    config: TrainingConfig,
) -> Path:
    """Save a training checkpoint and return its path.

    Args:
        output_dir: Directory checkpoints are written to (created if absent).
        step: Current global training step.
        model, optimizer, scheduler: Objects whose state is persisted.
        config: The run's configuration, saved alongside for provenance.

    Returns:
        Path to the written checkpoint file.

    Raises:
        OSError: If the checkpoint cannot be written (e.g. disk full,
            permissions).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / f"ckpt_step{step}.pt"
    try:
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "step": step,
                "config": config,
            },
            ckpt_path,
        )
    except OSError as exc:
        raise OSError(f"Failed to write checkpoint to {ckpt_path}: {exc}") from exc
    logger.info("Saved checkpoint: %s", ckpt_path)
    return ckpt_path


def load_checkpoint(
    resume_from: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    device: torch.device,
) -> int:
    """Restore model/optimizer/scheduler state from a checkpoint.

    Args:
        resume_from: Path to a checkpoint written by `save_checkpoint`.
        model, optimizer, scheduler: Objects to restore state into, in place.
        device: Device to map the checkpoint tensors onto.

    Returns:
        The global step the checkpoint was saved at, to resume counting from.

    Raises:
        FileNotFoundError: If `resume_from` does not exist.
        RuntimeError: If the checkpoint is malformed or incompatible.
    """
    if not resume_from.exists():
        raise FileNotFoundError(f"Checkpoint not found: {resume_from}")
    try:
        ckpt = torch.load(resume_from, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
    except (KeyError, RuntimeError) as exc:
        raise RuntimeError(f"Checkpoint at {resume_from} is malformed: {exc}") from exc
    logger.info("Resumed from step %d (%s)", ckpt["step"], resume_from)
    return ckpt["step"]


# --------------------------------------------------------------------------- #
# 5. Training loop
# --------------------------------------------------------------------------- #
def run_training_step(
    model: nn.Module,
    data_iter: Iterator[dict],
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    amp_dtype: torch.dtype,
    grad_accum_steps: int,
    max_grad_norm: float,
) -> Tuple[float, Iterator[dict]]:
    """Run a single optimizer step (across `grad_accum_steps` micro-batches).

    Args:
        model: The model being trained.
        data_iter: Current iterator over `loader`; re-created internally
            on exhaustion (the pretraining corpus is treated as an
            infinite stream via re-iteration).
        loader: The DataLoader backing `data_iter`, used to rebuild the
            iterator when exhausted.
        optimizer: Optimizer to step.
        scaler: GradScaler for mixed-precision backward/step.
        device: Compute device.
        amp_dtype: Autocast dtype (bf16 preferred, fp16 fallback).
        grad_accum_steps: Number of micro-batches accumulated per step.
        max_grad_norm: Gradient clipping threshold.

    Returns:
        A tuple of (summed loss over the accumulation window, the
        possibly-refreshed data iterator).
    """
    optimizer.zero_grad(set_to_none=True)
    accumulated_loss = 0.0

    for _ in range(grad_accum_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss / grad_accum_steps

        scaler.scale(loss).backward()
        accumulated_loss += loss.item()

    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    scaler.step(optimizer)
    scaler.update()

    return accumulated_loss, data_iter


def train(config: TrainingConfig) -> None:
    """Run the full pretraining loop until `config.max_steps` is reached.

    Args:
        config: Fully-specified, validated training configuration.

    Raises:
        RuntimeError: Propagated from dataset/tokenizer loading or a
            malformed checkpoint.
    """
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    tokenizer = load_tokenizer(config.tokenizer_name)
    loader = build_dataloader(config, tokenizer)
    model = build_model(config, vocab_size=len(tokenizer)).to(device)

    # Weight decay applied only to matrix params, not biases/LayerNorm — standard practice
    decay_params = [p for p in model.parameters() if p.dim() >= 2]
    no_decay_params = [p for p in model.parameters() if p.dim() < 2]
    optimizer = AdamW(
        [
            {"params": decay_params, "weight_decay": config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=config.lr,
        betas=config.betas,
    )

    scheduler: LambdaLR = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=config.max_steps,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    step = 0
    if config.resume_from is not None:
        step = load_checkpoint(config.resume_from, model, optimizer, scheduler, device)

    model.train()
    accum_loss = 0.0
    window_start_time = time.time()
    data_iter = iter(loader)

    try:
        while step < config.max_steps:
            step_loss, data_iter = run_training_step(
                model=model,
                data_iter=data_iter,
                loader=loader,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                amp_dtype=amp_dtype,
                grad_accum_steps=config.grad_accum_steps,
                max_grad_norm=config.max_grad_norm,
            )
            scheduler.step()
            step += 1
            accum_loss += step_loss

            if step % config.log_every == 0:
                _log_progress(step, accum_loss, config.log_every, scheduler, window_start_time)
                accum_loss = 0.0
                window_start_time = time.time()

            if step % config.save_every == 0 or step == config.max_steps:
                save_checkpoint(config.output_dir, step, model, optimizer, scheduler, config)

    except KeyboardInterrupt:
        logger.warning("Interrupted by user — saving checkpoint before exiting.")
        save_checkpoint(config.output_dir, step, model, optimizer, scheduler, config)
        raise

    logger.info("Pretraining complete. Model weights are now ready for the SFT stage.")


def _log_progress(
    step: int,
    accum_loss: float,
    log_every: int,
    scheduler: LambdaLR,
    window_start_time: float,
) -> None:
    """Log a single progress line (loss, perplexity, learning rate, throughput).

    Args:
        step: Current global step.
        accum_loss: Loss summed over the last `log_every` steps.
        log_every: Number of steps the loss was accumulated over.
        scheduler: Used to read the current learning rate.
        window_start_time: `time.time()` value at the start of this
            logging window, used to compute elapsed seconds.
    """
    elapsed = time.time() - window_start_time
    avg_loss = accum_loss / log_every
    perplexity = math.exp(avg_loss) if avg_loss < LOSS_PPL_OVERFLOW_GUARD else float("inf")
    current_lr = scheduler.get_last_lr()[0]
    logger.info(
        "step %7d | loss %.4f | ppl %8.2f | lr %.2e | %.1fs/%dsteps",
        step, avg_loss, perplexity, current_lr, elapsed, log_every,
    )


# --------------------------------------------------------------------------- #
# 6. Entry point
# --------------------------------------------------------------------------- #
def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point.

    Args:
        argv: Optional explicit argument list (mainly for testing).

    Returns:
        Process exit code (0 on success, 1 on a handled failure).
    """
    config = parse_args(argv)
    try:
        train(config)
    except RuntimeError as exc:
        logger.error("Pretraining failed: %s", exc)
        return 1
    except KeyboardInterrupt:
        logger.info("Exiting after user interrupt.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())