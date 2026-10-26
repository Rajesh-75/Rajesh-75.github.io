"""
preference_collection.py
==================================================================
 STAGE B.5 OF RLHF: PREFERENCE COLLECTION
==================================================================

Pipeline position:
    Stage 0  -> SFT (sft_training.py)
    Stage A  -> Reward model training (reward_model_training.py)
    Stage B  -> Response generation / sampling (response_generation.py)
    Stage B.5-> Preference collection (THIS FILE)
    Stage C  -> PPO policy fine-tuning (ppo_rlhf_training.py)

Goal:
    Take Stage B's output — multiple candidate completions per
    prompt — and produce labeled preference pairs:
        {"prompt": "...", "chosen": "...", "rejected": "..."}
    which is exactly the format reward_model_training.py expects.

    Two labeling backends are implemented, selected via config.mode:
        "rlaif"  : an LLM judge compares each pair and picks a winner
        "human"  : an interactive terminal loop for a human labeler

Two things this file treats as non-optional, not nice-to-haves:

1. PAIR EXPLOSION CONTROL
   N completions per prompt -> C(N,2) possible pairs. Labeling all
   of them wastes labeler/judge budget on redundant comparisons and
   overrepresents prompts that happened to produce more unique
   samples. We randomly subsample to max_pairs_per_prompt.

2. POSITION BIAS (RLAIF mode only)
   LLM judges are well documented to favor whichever response is
   shown first (or, for some models, second) independent of content
   quality. Mitigation used here: query the judge TWICE per pair with
   the response order swapped, and only accept a verdict if both
   queries agree once order is accounted for. This doubles judge API
   cost but is the standard fix — skipping it means an unknown
   fraction of your "preferences" are actually judge order artifacts,
   which the reward model will then dutifully learn.

Input format (JSONL, from response_generation.py):
    {"prompt": "...", "responses": ["...", "...", ...]}

Output format (JSONL, consumed by reward_model_training.py):
    {"prompt": "...", "chosen": "...", "rejected": "..."}
==================================================================
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("preference_collection")


# ==================================================================
# Configuration
# ==================================================================
@dataclass
class PreferenceConfig:
    input_file: str                                    # Stage B output
    output_file: str = "./preferences.jsonl"
    mode: str = "rlaif"                                 # "rlaif" | "human"

    max_pairs_per_prompt: int = 3                       # caps C(N,2) explosion
    swap_order_check: bool = True                       # position-bias mitigation, RLAIF only

    judge_model: str = "claude-sonnet-4-6"
    judge_max_tokens: int = 10
    judge_criteria: str = (
        "Which response is more helpful, accurate, and appropriately concise, "
        "while being harmless and honest? Judge substance, not length or formatting."
    )

    seed: int = 42

    def __post_init__(self) -> None:
        if self.mode not in ("rlaif", "human"):
            raise ValueError(f"mode must be 'rlaif' or 'human', got {self.mode!r}")


# ==================================================================
# Data loading: Stage B's per-prompt completion groups
# ==================================================================
def load_generation_groups(path: str) -> list[dict[str, Any]]:
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    groups: list[dict[str, Any]] = []
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
            if "prompt" not in record or "responses" not in record:
                logger.warning("Skipping record missing prompt/responses at %s:%d", path, line_num)
                continue
            groups.append(record)

    if not groups:
        raise ValueError(f"No valid prompt groups loaded from {path}")
    logger.info("Loaded %d prompt groups from %s", len(groups), path)
    return groups


def build_pairs(responses: list[str], max_pairs: int, rng: random.Random) -> list[tuple[str, str]]:
    """All pairwise combinations of responses for one prompt, capped
    and randomly subsampled to max_pairs. Order within each tuple is
    arbitrary — judge_pair_rlaif's swap-order check (or the human
    loop's display-order randomization) is what actually controls
    for position effects, not the order pairs are built in here."""
    all_pairs = list(itertools.combinations(responses, 2))
    if len(all_pairs) <= max_pairs:
        return all_pairs
    return rng.sample(all_pairs, max_pairs)


def save_jsonl(records: list[dict[str, Any]], path: str) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("Wrote %d records to %s", len(records), path)


# ==================================================================
# RLAIF backend: LLM judge with position-bias mitigation
# ==================================================================
def _build_judge_prompt(prompt: str, response_a: str, response_b: str, criteria: str) -> str:
    return (
        "You are evaluating two AI responses to the same user prompt.\n\n"
        f"Judging criteria: {criteria}\n\n"
        f"User prompt:\n{prompt}\n\n"
        f"Response A:\n{response_a}\n\n"
        f"Response B:\n{response_b}\n\n"
        "Which response is better? Reply with EXACTLY one token: 'A' or 'B'. "
        "If they are truly equal in quality, reply 'TIE'. No other text."
    )


def call_judge_model(judge_prompt: str, config: PreferenceConfig) -> str:
    """Pluggable judge backend — default implementation calls the
    Anthropic API. Everything downstream only depends on this
    returning raw text starting with 'A', 'B', or 'TIE', so swapping
    in a different judge (another provider, a local model) means
    replacing only this function.

    Requires: pip install anthropic, and an ANTHROPIC_API_KEY env var.
    """
    try:
        import anthropic
    except ImportError as e:
        raise ImportError(
            "RLAIF mode requires the `anthropic` package (`pip install anthropic`) "
            "and an ANTHROPIC_API_KEY environment variable set. Alternatively, swap "
            "call_judge_model() for your own backend and keep mode='rlaif'."
        ) from e

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    response = client.messages.create(
        model=config.judge_model,
        max_tokens=config.judge_max_tokens,
        messages=[{"role": "user", "content": judge_prompt}],
    )
    return "".join(block.text for block in response.content if hasattr(block, "text")).strip()


def parse_judge_verdict(raw: str) -> Optional[str]:
    """Returns 'A', 'B', or None (covers 'TIE' and any unparseable reply —
    both are treated identically downstream: no confident preference)."""
    normalized = raw.strip().upper()
    if normalized.startswith("A"):
        return "A"
    if normalized.startswith("B"):
        return "B"
    return None


def judge_pair_rlaif(
    prompt: str, response_1: str, response_2: str, config: PreferenceConfig
) -> Optional[tuple[str, str]]:
    """Returns (chosen, rejected), or None if no confident verdict.

    With swap_order_check enabled: queries the judge twice, with
    response order swapped on the second call, and only returns a
    verdict if both calls agree once order is accounted for. A
    disagreement between the two passes means either genuine
    ambiguity or position bias — both are reasons to drop the pair
    rather than force a label, since a forced label here becomes a
    silently wrong training signal for the reward model.
    """
    prompt_pass1 = _build_judge_prompt(prompt, response_1, response_2, config.judge_criteria)
    verdict_1 = parse_judge_verdict(call_judge_model(prompt_pass1, config))

    if not config.swap_order_check:
        if verdict_1 == "A":
            return response_1, response_2
        if verdict_1 == "B":
            return response_2, response_1
        return None

    prompt_pass2 = _build_judge_prompt(prompt, response_2, response_1, config.judge_criteria)
    verdict_2 = parse_judge_verdict(call_judge_model(prompt_pass2, config))

    if verdict_1 is None or verdict_2 is None:
        return None  # a tie or unparseable reply on either pass -> no confident label

    r1_preferred_pass1 = verdict_1 == "A"       # response_1 shown as "A" in pass 1
    r1_preferred_pass2 = verdict_2 == "B"       # response_1 shown as "B" in pass 2

    if r1_preferred_pass1 and r1_preferred_pass2:
        return response_1, response_2
    if not r1_preferred_pass1 and not r1_preferred_pass2:
        return response_2, response_1
    return None  # passes disagree once order is accounted for -> drop


# ==================================================================
# Human backend: interactive terminal labeling
# ==================================================================
def collect_labels_human_cli(
    pending_pairs: list[dict[str, Any]], rng: random.Random
) -> list[dict[str, Any]]:
    """Interactive terminal loop for small-scale classroom labeling.
    For anything beyond a handful of pairs, swap this for a real
    annotation UI/queue with inter-annotator agreement tracking —
    this exists to let students see the labeling step end-to-end,
    not as a production labeling tool.

    Display order per pair is randomized (independent of how the
    pair was built) so a labeler's own attention bias — e.g. a
    tendency to favor whichever response is shown first — doesn't
    correlate with anything meaningful in the underlying data."""
    labeled: list[dict[str, Any]] = []

    for i, item in enumerate(pending_pairs):
        prompt, r1, r2 = item["prompt"], item["response_1"], item["response_2"]
        r1_shown_first = rng.random() < 0.5
        shown_1, shown_2 = (r1, r2) if r1_shown_first else (r2, r1)

        print("\n" + "=" * 80)
        print(f"[{i + 1}/{len(pending_pairs)}] PROMPT:\n{prompt}\n")
        print(f"--- Response 1 ---\n{shown_1}\n")
        print(f"--- Response 2 ---\n{shown_2}\n")
        choice = input("Which is better? [1/2/t=tie/s=skip/q=quit]: ").strip().lower()

        if choice == "q":
            logger.info("Labeling stopped early by user at pair %d/%d", i + 1, len(pending_pairs))
            break
        if choice in ("s", "t"):
            continue  # ties are never converted into a chosen/rejected pair
        if choice not in ("1", "2"):
            logger.warning("Unrecognized input %r, skipping pair", choice)
            continue

        chosen = shown_1 if choice == "1" else shown_2
        rejected = shown_2 if choice == "1" else shown_1
        labeled.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})

    return labeled


# ==================================================================
# Driver
# ==================================================================
def run_preference_collection(config: PreferenceConfig) -> list[dict[str, Any]]:
    rng = random.Random(config.seed)
    groups = load_generation_groups(config.input_file)

    pending_pairs: list[dict[str, Any]] = []
    n_skipped_groups = 0
    for group in groups:
        prompt = group["prompt"]
        responses = group.get("responses", [])
        if len(responses) < 2:
            n_skipped_groups += 1
            continue
        for r1, r2 in build_pairs(responses, config.max_pairs_per_prompt, rng):
            pending_pairs.append({"prompt": prompt, "response_1": r1, "response_2": r2})

    if n_skipped_groups:
        logger.warning("Skipped %d prompt groups with fewer than 2 responses", n_skipped_groups)
    logger.info("Built %d candidate pairs from %d prompt groups", len(pending_pairs), len(groups))

    if config.mode == "rlaif":
        results: list[dict[str, Any]] = []
        n_no_verdict = 0
        for i, item in enumerate(pending_pairs):
            verdict = judge_pair_rlaif(item["prompt"], item["response_1"], item["response_2"], config)
            if verdict is None:
                n_no_verdict += 1
            else:
                chosen, rejected = verdict
                results.append({"prompt": item["prompt"], "chosen": chosen, "rejected": rejected})

            if (i + 1) % 20 == 0:
                logger.info(
                    "Judged %d/%d pairs (%d without a confident verdict so far)",
                    i + 1, len(pending_pairs), n_no_verdict,
                )

        if n_no_verdict:
            logger.warning(
                "%d/%d pairs (%.1f%%) had no confident verdict (tie, unparseable, or "
                "order-swap disagreement) and were dropped. A high rate here usually "
                "means judge_criteria is too vague, or Stage B's sampling settings are "
                "producing genuinely similar-quality completions (see the low-diversity "
                "warning from response_generation.py).",
                n_no_verdict, len(pending_pairs), 100 * n_no_verdict / max(len(pending_pairs), 1),
            )

    else:  # mode == "human"
        results = collect_labels_human_cli(pending_pairs, rng)

    save_jsonl(results, config.output_file)
    logger.info(
        "Saved %d preference pairs to %s (ready for reward_model_training.py)",
        len(results), config.output_file,
    )
    return results


# ==================================================================
# CLI entry point
# ==================================================================
def parse_args() -> PreferenceConfig:
    parser = argparse.ArgumentParser(description="Stage B.5: preference collection for RLHF")
    parser.add_argument("--input_file", required=True, help="Stage B output JSONL")
    parser.add_argument("--output_file", default="./preferences.jsonl")
    parser.add_argument("--mode", choices=["rlaif", "human"], default="rlaif")
    parser.add_argument("--max_pairs_per_prompt", type=int, default=3)
    parser.add_argument("--swap_order_check", action="store_true", default=True)
    parser.add_argument("--no_swap_order_check", dest="swap_order_check", action="store_false")
    parser.add_argument("--judge_model", default="claude-sonnet-4-6")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    return PreferenceConfig(**vars(args))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    config = parse_args()
    logger.info("Starting Stage B.5 preference collection with config: %s", config)
    run_preference_collection(config)


if __name__ == "__main__":
    main()