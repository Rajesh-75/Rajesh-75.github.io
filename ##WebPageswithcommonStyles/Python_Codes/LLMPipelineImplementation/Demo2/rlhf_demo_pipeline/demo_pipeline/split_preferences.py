"""
split_preferences.py
==================================================================
Splits preferences.jsonl into train/val for reward model training.

Split is done by PROMPT GROUP, not by line — pairs sharing a prompt
must land on the same side of the split, or the reward model can
effectively memorize a prompt from train and get an inflated val
accuracy on a near-duplicate pair from the same prompt in val.

Pulled out into its own script (instead of an inline `python -c`
one-liner) so it runs identically from bash, cmd, or PowerShell —
no shell-specific quoting or line-continuation syntax involved.
==================================================================
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Split preference pairs into train/val by prompt group")
    parser.add_argument("--preferences_file", required=True)
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--val_file", required=True)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    with open(args.preferences_file, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    by_prompt: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_prompt[record["prompt"]].append(record)

    prompts = list(by_prompt.keys())
    rng.shuffle(prompts)
    n_val = max(1, int(args.val_fraction * len(prompts)))
    val_prompts = set(prompts[:n_val])

    Path(args.train_file).parent.mkdir(parents=True, exist_ok=True)
    Path(args.val_file).parent.mkdir(parents=True, exist_ok=True)

    with open(args.train_file, "w", encoding="utf-8") as train_f, \
         open(args.val_file, "w", encoding="utf-8") as val_f:
        for prompt, recs in by_prompt.items():
            target = val_f if prompt in val_prompts else train_f
            for rec in recs:
                target.write(json.dumps(rec) + "\n")

    print(f"Split {len(prompts)} prompts -> {len(prompts) - n_val} train / {n_val} val")


if __name__ == "__main__":
    main()
