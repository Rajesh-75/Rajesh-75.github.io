# RLHF pipeline — laptop / CPU demonstration pack

## What's in here (and why these five files)

| File | Stage | Why it's included |
|---|---|---|
| `sft_training.py` | 0 | Starting point every later stage depends on |
| `reward_model_training.py` | A | Trains the scalar reward model PPO needs |
| `response_generation.py` | B | Produces the multiple completions Stage B.5 labels |
| `preference_collection.py` | B.5 | Turns completions into chosen/rejected pairs |
| `ppo_rlhf_training.py` | C | The actual RLHF policy update |

`preference_collection.py` is imported by name in the Makefile only as
a script, but `reward_model_training.py`'s **filename is load-bearing**:
`ppo_rlhf_training.py` does `from reward_model_training import
RewardModel`, so it must keep this exact name in this directory.

## What was deliberately left out of this demo

- **`LLM_Pre_training.py` and `convert_pre_trained.py`** — real
  pretraining needs days/weeks on many GPUs; there's no CPU-laptop
  version of "pretrain a language model" worth demonstrating. Instead
  this pack starts from an existing small instruct model
  (`HuggingFaceTB/SmolLM2-135M-Instruct`, ~135M params) and picks the
  pipeline up at Stage 0 (SFT), exactly like the original
  `makefile.txt` was already set up to do.
- **`LLM_Pipeline.py`** — the Python orchestrator that *also* runs
  pretraining. Not needed once pretraining is out of scope; the plain
  Makefile does the same job for stages SFT→PPO with less moving
  parts to explain in a demo.
- **`BPE_Tokenizer.py`, `GPT2Model_fromScratch.py`,
  `GPTModel_using_only_numpy_lib.py`** — good teaching material for
  *how a tokenizer/transformer works internally*, but none of the
  pipeline scripts actually import or call them (the pipeline uses
  HuggingFace's tokenizer/model classes throughout). Keep these as a
  separate "concepts" demo, not part of this pipeline pack.

## Code tweaks made for CPU-only running

1. `reward_model_training.py` and `ppo_rlhf_training.py` defaulted to
   `dtype="bfloat16"` with **no CLI flag** to change it. Both now
   default to `"float32"` and expose `--dtype`, since mixed precision
   without a GPU either fails or gains you nothing.
2. `reward_model_training.py`'s `grad_accum_steps` defaulted to 4 with
   no CLI flag. On a handful of demo examples that means the optimizer
   would never actually take a step (not enough micro-batches per
   epoch to reach the accumulation target). Default is now 1, with
   `--grad_accum_steps` exposed.
3. The Makefile now explicitly passes `--batch_size`/`--grad_accum_steps`
   to the SFT stage too, for the same reason — the originally uploaded
   `makefile.txt` left them at their (too-large-for-6-examples)
   defaults.
4. `preference_collection.py`'s default mode is switched to `human` in
   this Makefile (an interactive terminal Y/N-style loop) so the demo
   runs with **no API key and no internet dependency**. Flip
   `PREF_MODE=rlaif` (and set `ANTHROPIC_API_KEY`) once you want an LLM
   judge instead of labeling by hand.

## Data

`data/sft_train.jsonl`, `data/sft_val.jsonl`, `data/rl_prompts.jsonl`
are tiny, hand-written stand-ins (a handful of lines each) — just
enough for every stage to have something to chew on end to end. Swap
in your real data later; the format each stage expects is documented
at the top of that stage's script.

## How to run

```bash
pip install -r requirements.txt

make all           # runs sft -> generate -> preferences -> reward_model -> ppo
# or step through it:
make sft
make generate
make preferences   # interactive: you'll be asked to pick the better response for each pair
make reward_model
make ppo
```

Everything here is sized to run in minutes on a laptop CPU, purely to
demonstrate that the mechanics of the pipeline are wired together
correctly end to end. It is not tuned for output quality — six SFT
examples will not produce a good model. Scale-up (real data, a real
base model, GPUs, evaluation, safety checks) is the follow-up phase.
