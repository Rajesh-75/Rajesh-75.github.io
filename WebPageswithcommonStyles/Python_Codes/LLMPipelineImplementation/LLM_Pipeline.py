"""
llm_pipeline.py
==================================================================
 END-TO-END LLM PIPELINE ORCHESTRATOR
==================================================================

Ties together the four independent training stages into one runnable
application:

    [pretrain] -> [convert] -> [sft] -> [reward_model] -> [ppo]

Each stage is a separate, independently-runnable script (this mirrors
how real training pipelines are built — each stage is its own
process, so it can be scaled, retried, or run on different hardware
independently). This orchestrator's job is narrow but essential:

  1. WORKSPACE SETUP — copies your four stage scripts into a run
     workspace under the exact filenames they expect each other to
     have (ppo_rlhf_training.py hard-imports "reward_model_training",
     so that file must be named reward_model_training.py at run time,
     regardless of what you called it on disk).

  2. STAGE EXECUTION — runs each requested stage as a subprocess with
     its own CLI arguments (exactly as if you ran it by hand), and
     fails the whole run loudly if any stage exits non-zero.

  3. CHECKPOINT HAND-OFF — automatically resolves each stage's output
     checkpoint into the next stage's input path, including running
     the pretrain -> HF-format conversion step your existing scripts
     need but don't do themselves.

  4. MANIFEST — writes a JSON record of every checkpoint path produced,
     so a run is auditable after the fact.

Usage:
    python llm_pipeline.py --config pipeline_config.json

    # Run only a subset of stages (e.g. you already have an SFT model
    # and just want to run reward-model + PPO on top of it):
    python llm_pipeline.py --config pipeline_config.json --stages reward_model,ppo

Example pipeline_config.json:
{
  "workspace": "./pipeline_run",
  "scripts": {
    "pretrain": "./LLM_Pre_training.py",
    "sft": "./SFT_CS.py",
    "reward_model": "./Reward_model.py",
    "ppo": "./PPO_RLHF.py"
  },
  "stages": ["pretrain", "sft", "reward_model", "ppo"],
  "pretrain": {
    "dataset_name": "wikitext", "dataset_config": "wikitext-103-raw-v1",
    "tokenizer_name": "gpt2", "context_length": 1024,
    "n_layer": 12, "n_head": 12, "n_embd": 768,
    "batch_size": 8, "grad_accum_steps": 4, "max_steps": 100000,
    "warmup_steps": 2000, "lr": 0.0003
  },
  "sft": {
    "train_file": "data/sft_train.jsonl", "val_file": "data/sft_val.jsonl",
    "epochs": 3, "batch_size": 4
  },
  "reward_model": {
    "train_file": "data/rm_train.jsonl", "val_file": "data/rm_val.jsonl",
    "epochs": 1
  },
  "ppo": {
    "prompt_file": "data/ppo_prompts.jsonl", "total_ppo_steps": 500
  }
}

Any field left out of "sft"/"reward_model"/"ppo" that names a model
checkpoint (model_name / base_model_name / policy_model_name /
reward_model_path) is auto-filled from the previous stage's output
when that previous stage ran in this same pipeline invocation. Supply
it explicitly if you're skipping the earlier stage(s).
==================================================================
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("llm_pipeline")
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"

# The exact filenames each stage script's internal imports/docs expect.
# ppo_rlhf_training.py specifically does `from reward_model_training import RewardModel`,
# so that one MUST be named exactly this at run time.
CANONICAL_SCRIPT_NAMES: dict[str, str] = {
    "pretrain": "llm_pretraining.py",
    "convert": "convert_pretrained_checkpoint.py",
    "sft": "sft_training.py",
    "reward_model": "reward_model_training.py",
    "ppo": "ppo_rlhf_training.py",
}

STAGE_ORDER: list[str] = ["pretrain", "sft", "reward_model", "ppo"]

# CLI flags each stage's own argparse setup actually exposes.
# Only keys present here are forwarded from the JSON config as --flags;
# anything else in the config's stage dict is silently ignored so typos
# don't crash the run 
STAGE_ALLOWED_FLAGS: dict[str, set[str]] = {
    "pretrain": {
        "dataset_name", "dataset_config", "tokenizer_name", "context_length",
        "n_layer", "n_head", "n_embd", "batch_size", "grad_accum_steps",
        "max_steps", "warmup_steps", "lr", "weight_decay", "max_grad_norm",
        "betas", "output_dir", "log_every", "save_every", "resume_from", "seed",
    },
    "sft": {
        "model_name", "train_file", "val_file", "output_dir", "max_length",
        "epochs", "batch_size", "grad_accum_steps", "learning_rate",
        "warmup_ratio", "seed", "dtype",
    },
    "reward_model": {
        "base_model_name", "train_file", "val_file", "output_dir",
        "epochs", "batch_size", "learning_rate", "seed",
    },
    "ppo": {
        "policy_model_name", "reward_model_path", "prompt_file", "output_dir",
        "total_ppo_steps", "batch_size", "learning_rate", "kl_coef", "seed",
    },
}

DEFAULT_OUTPUT_DIRS: dict[str, str] = {
    "pretrain": "pretrain_ckpts",
    "sft": "sft_ckpts",
    "reward_model": "reward_model_ckpts",
    "ppo": "ppo_ckpts",
}


# ==================================================================
# Configuration
# ==================================================================
@dataclass
class PipelineConfig:
    workspace: Path
    scripts: dict[str, str]          # stage name -> path to the script AS YOU HAVE IT NAMED
    stages: list[str]
    stage_args: dict[str, dict[str, Any]] = field(default_factory=dict)


def load_pipeline_config(config_path: Path, stage_override: Optional[list[str]]) -> PipelineConfig:
    """Load and validate the pipeline JSON config.

    Raises:
        RuntimeError: If the file is missing, malformed, or references
            an unknown stage name.
    """
    if not config_path.exists():
        raise RuntimeError(f"Pipeline config not found: {config_path}")

    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Pipeline config at {config_path} is not valid JSON: {exc}") from exc

    stages = stage_override if stage_override else raw.get("stages", STAGE_ORDER)
    unknown = set(stages) - set(STAGE_ORDER)
    if unknown:
        raise RuntimeError(f"Unknown stage(s) in config: {sorted(unknown)}. Valid: {STAGE_ORDER}")

    scripts = raw.get("scripts", {})
    missing_scripts = [s for s in stages if s not in scripts]
    if missing_scripts:
        raise RuntimeError(f"No script path given in config['scripts'] for stage(s): {missing_scripts}")

    return PipelineConfig(
        workspace=Path(raw.get("workspace", "./pipeline_run")),
        scripts=scripts,
        stages=stages,
        stage_args={s: raw.get(s, {}) for s in STAGE_ORDER},
    )


# ==================================================================
# Workspace setup — copy scripts to the filenames they expect of each other
# ==================================================================
def setup_workspace(config: PipelineConfig) -> Path:
    """Copy every script this run needs into the workspace under its
    canonical name, so cross-script imports (notably ppo_rlhf_training's
    `from reward_model_training import RewardModel`) resolve correctly
    regardless of what you named the files on disk.
    """
    config.workspace.mkdir(parents=True, exist_ok=True)

    stages_needing_scripts = list(config.stages)
    if "pretrain" in stages_needing_scripts:
        stages_needing_scripts.append("convert")
    # reward_model.py must always be present alongside ppo, even if this
    # run isn't training a fresh RM (e.g. you supply --reward_model_path
    # from an earlier run) — ppo_rlhf_training.py imports it unconditionally.
    if "ppo" in stages_needing_scripts and "reward_model" not in stages_needing_scripts:
        stages_needing_scripts.append("reward_model")

    for stage in stages_needing_scripts:
        canonical_name = CANONICAL_SCRIPT_NAMES[stage]
        dest = config.workspace / canonical_name

        if stage == "convert":
            # Ships alongside this orchestrator file.
            src = Path(__file__).parent / "convert_pretrained_checkpoint.py"
        else:
            src_str = config.scripts.get(stage)
            if src_str is None:
                raise RuntimeError(
                    f"Stage '{stage}' needs its script but none was given in config['scripts']"
                )
            src = Path(src_str)

        if not src.exists():
            raise RuntimeError(f"Script for stage '{stage}' not found: {src}")

        shutil.copy2(src, dest)
        logger.info("Staged %s -> %s", src, dest)

    return config.workspace


# ==================================================================
# Stage execution
# ==================================================================
def _dict_to_argv(args: dict[str, Any], allowed: set[str]) -> list[str]:
    """Convert a {flag_name: value} dict into CLI argv, forwarding only
    keys the target script's argparse actually accepts."""
    argv: list[str] = []
    for key, value in args.items():
        if key not in allowed:
            logger.warning("Ignoring unrecognized config field '%s' (not a valid CLI flag for this stage)", key)
            continue
        if value is None:
            continue
        flag = f"--{key}"
        if isinstance(value, (list, tuple)):
            argv.append(flag)
            argv.extend(str(v) for v in value)
        else:
            argv.extend([flag, str(value)])
    return argv


def run_stage(stage: str, workspace: Path, argv: list[str]) -> None:
    """Run one stage as a subprocess, streaming its output live.

    Raises:
        RuntimeError: If the subprocess exits with a non-zero code.
    """
    script = workspace / CANONICAL_SCRIPT_NAMES[stage]
    command = [sys.executable, str(script), *argv]
    logger.info("=== Starting stage '%s' ===", stage)
    logger.info("Command: %s", " ".join(command))

    start = time.time()
    result = subprocess.run(command, cwd=workspace)
    elapsed = time.time() - start

    if result.returncode != 0:
        raise RuntimeError(
            f"Stage '{stage}' failed with exit code {result.returncode} after {elapsed:.1f}s"
        )
    logger.info("=== Stage '%s' complete (%.1fs) ===", stage, elapsed)


# ==================================================================
# Checkpoint hand-off between stages
# ==================================================================
def _find_latest_pretrain_checkpoint(output_dir: Path) -> Path:
    checkpoints = sorted(
        output_dir.glob("ckpt_step*.pt"),
        key=lambda p: int(p.stem.replace("ckpt_step", "")),
    )
    if not checkpoints:
        raise RuntimeError(f"No pretraining checkpoints found in {output_dir}")
    return checkpoints[-1]


def run_pipeline(config: PipelineConfig) -> dict[str, str]:
    """Run every requested stage in order, wiring each stage's output
    checkpoint into the next stage's input. Returns a manifest dict of
    the checkpoint paths produced.
    """
    workspace = setup_workspace(config)
    manifest: dict[str, str] = {}

    # --- Stage: pretrain (+ conversion to HF format) ---
    sft_model_name: Optional[str] = config.stage_args.get("sft", {}).get("model_name")
    if "pretrain" in config.stages:
        pretrain_args = dict(config.stage_args.get("pretrain", {}))
        pretrain_output_dir = Path(pretrain_args.get("output_dir", DEFAULT_OUTPUT_DIRS["pretrain"]))
        pretrain_args.setdefault("output_dir", str(pretrain_output_dir))

        run_stage("pretrain", workspace, _dict_to_argv(pretrain_args, STAGE_ALLOWED_FLAGS["pretrain"]))

        latest_ckpt = _find_latest_pretrain_checkpoint(workspace / pretrain_output_dir)
        manifest["pretrain_checkpoint"] = str(latest_ckpt)

        hf_dir = workspace / pretrain_output_dir / "hf_format"
        run_stage(
            "convert",
            workspace,
            [
                "--checkpoint", str(latest_ckpt),
                "--tokenizer_name", str(pretrain_args.get("tokenizer_name", "gpt2")),
                "--context_length", str(pretrain_args.get("context_length", 1024)),
                "--n_layer", str(pretrain_args.get("n_layer", 12)),
                "--n_head", str(pretrain_args.get("n_head", 12)),
                "--n_embd", str(pretrain_args.get("n_embd", 768)),
                "--output_dir", str(hf_dir),
            ],
        )
        manifest["pretrain_hf_checkpoint"] = str(hf_dir)
        sft_model_name = sft_model_name or str(hf_dir)

    # --- Stage: SFT ---
    rm_base_model: Optional[str] = config.stage_args.get("reward_model", {}).get("base_model_name")
    if "sft" in config.stages:
        sft_args = dict(config.stage_args.get("sft", {}))
        if sft_args.get("model_name") is None:
            if sft_model_name is None:
                raise RuntimeError(
                    "SFT stage needs a starting model: either run the 'pretrain' stage first, "
                    "or set sft.model_name in the config to an existing model (HF hub id or local path)."
                )
            sft_args["model_name"] = sft_model_name
        sft_output_dir = Path(sft_args.get("output_dir", DEFAULT_OUTPUT_DIRS["sft"]))
        sft_args.setdefault("output_dir", str(sft_output_dir))

        run_stage("sft", workspace, _dict_to_argv(sft_args, STAGE_ALLOWED_FLAGS["sft"]))

        sft_final = workspace / sft_output_dir / "final"
        manifest["sft_checkpoint"] = str(sft_final)
        rm_base_model = rm_base_model or str(sft_final)

    # --- Stage: reward model ---
    policy_model_name: Optional[str] = config.stage_args.get("ppo", {}).get("policy_model_name")
    if "reward_model" in config.stages:
        rm_args = dict(config.stage_args.get("reward_model", {}))
        if rm_args.get("base_model_name") is None:
            if rm_base_model is None:
                raise RuntimeError(
                    "reward_model stage needs a starting checkpoint: either run the 'sft' stage first, "
                    "or set reward_model.base_model_name in the config."
                )
            rm_args["base_model_name"] = rm_base_model
        rm_output_dir = Path(rm_args.get("output_dir", DEFAULT_OUTPUT_DIRS["reward_model"]))
        rm_args.setdefault("output_dir", str(rm_output_dir))

        run_stage("reward_model", workspace, _dict_to_argv(rm_args, STAGE_ALLOWED_FLAGS["reward_model"]))

        rm_final = workspace / rm_output_dir / "final"
        manifest["reward_model_checkpoint"] = str(rm_final)
        policy_model_name = policy_model_name or rm_base_model

    # --- Stage: PPO ---
    if "ppo" in config.stages:
        ppo_args = dict(config.stage_args.get("ppo", {}))
        if ppo_args.get("policy_model_name") is None:
            if policy_model_name is None:
                raise RuntimeError(
                    "ppo stage needs a policy checkpoint: either run the 'sft' stage first, "
                    "or set ppo.policy_model_name in the config."
                )
            ppo_args["policy_model_name"] = policy_model_name
        if ppo_args.get("reward_model_path") is None:
            rm_path = manifest.get("reward_model_checkpoint")
            if rm_path is None:
                raise RuntimeError(
                    "ppo stage needs a reward model: either run the 'reward_model' stage first, "
                    "or set ppo.reward_model_path in the config."
                )
            ppo_args["reward_model_path"] = rm_path
        ppo_output_dir = Path(ppo_args.get("output_dir", DEFAULT_OUTPUT_DIRS["ppo"]))
        ppo_args.setdefault("output_dir", str(ppo_output_dir))

        run_stage("ppo", workspace, _dict_to_argv(ppo_args, STAGE_ALLOWED_FLAGS["ppo"]))

        manifest["ppo_checkpoint"] = str(workspace / ppo_output_dir / "final")

    return manifest


# ==================================================================
# CLI entry point
# ==================================================================
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the full pretrain -> SFT -> RM -> PPO pipeline")
    parser.add_argument("--config", required=True, help="Path to pipeline_config.json")
    parser.add_argument(
        "--stages", default=None,
        help="Comma-separated subset to run, e.g. 'sft,reward_model,ppo'. Defaults to config['stages'].",
    )
    parser.add_argument(
        "--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format=LOG_FORMAT)

    stage_override = args.stages.split(",") if args.stages else None

    try:
        config = load_pipeline_config(Path(args.config), stage_override)
        logger.info("Running stages: %s", config.stages)
        manifest = run_pipeline(config)
    except RuntimeError as exc:
        logger.error("Pipeline failed: %s", exc)
        return 1

    manifest_path = Path(config.workspace) / "pipeline_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Pipeline complete. Manifest written to %s", manifest_path)
    for key, value in manifest.items():
        logger.info("  %s: %s", key, value)
    return 0


if __name__ == "__main__":
    sys.exit(main())