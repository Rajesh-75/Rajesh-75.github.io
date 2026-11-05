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

### macOS / Linux / WSL (has `make`)

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

### Windows, plain Command Prompt (no `make` needed)

`make` is not a built-in Windows command — cmd has no equivalent unless
you separately install GNU Make (e.g. via Chocolatey: `choco install
make`). To avoid adding tooling on an older machine, use the included
`run_demo.bat` instead — it runs the exact same five stages with the
exact same tuned arguments as `make all`:

```bat
pip install -r requirements.txt
run_demo.bat
```

It stops at the first stage that fails (checked via `errorlevel` after
each step) rather than plowing on with a broken checkpoint, the same
way `make`'s dependency chain does.

Everything else in this repo — the `pip install`, `huggingface-cli
download`, and `python -c "..."` one-liners shown earlier — runs
verbatim in `cmd.exe` with no changes needed; the quoting style used
(double quotes outside, single quotes inside) is not bash-specific.
The one thing that *was* bash-specific was the old inline
`python -c` block used to split preferences into train/val, which
relied on backslash line-continuation — that's now its own file,
`split_preferences.py`, called the same way from any shell.

### A heads-up specific to Windows 8

Windows 8 (not 8.1) reached end of support from Microsoft years ago,
and current Python releases (3.9+) officially require Windows 8.1 or
later — plain Windows 8 is below that floor. In practice this usually
still works if you already have a working Python 3.9–3.11 install,
since Python itself doesn't actively check the OS build. But if
`pip install torch` fails, hangs, or a DLL-load error appears when you
run a stage, that's the likely cause, and the practical fixes are:
run this from a newer machine, a Linux VM, or a free cloud notebook
(Google Colab, Kaggle) rather than fighting an OS this old to run a
current PyTorch build.

Everything here is sized to run in minutes on a laptop CPU, purely to
demonstrate that the mechanics of the pipeline are wired together
correctly end to end. It is not tuned for output quality — six SFT
examples will not produce a good model. Scale-up (real data, a real
base model, GPUs, evaluation, safety checks) is the follow-up phase.
