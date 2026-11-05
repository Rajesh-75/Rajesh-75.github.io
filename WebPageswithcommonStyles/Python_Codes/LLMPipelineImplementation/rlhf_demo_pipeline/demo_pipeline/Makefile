# ==================================================================
# RLHF PIPELINE Makefile — LAPTOP / CPU DEMO CONFIGURATION
# ==================================================================
# This is the uploaded makefile.txt with defaults tuned so every
# stage finishes in minutes on a CPU-only laptop, using a small
# instruct model instead of training one from scratch. Pretraining
# is deliberately out of scope here — see the README for why.
#
# Stage 0   -> sft_training.py
# Stage A   -> reward_model_training.py
# Stage B   -> response_generation.py
# Stage B.5 -> preference_collection.py
# Stage C   -> ppo_rlhf_training.py
#
# Usage:
#     make all                 # run the full pipeline, defaults below
#     make sft                 # run just through Stage 0
#     make reward_model        # run through Stage A (implies B, B.5)
#     make clean                # remove all generated artifacts
# ==================================================================

PYTHON            ?= python

# --- raw inputs you must supply ---
# A small model WITH a chat template baked into its tokenizer (the
# scripts call apply_chat_template) — plain "gpt2"/"distilgpt2" do
# NOT have one and will error out here.
BASE_MODEL        ?= HuggingFaceTB/SmolLM2-135M-Instruct
SFT_TRAIN_FILE    ?= data/sft_train.jsonl
SFT_VAL_FILE      ?= data/sft_val.jsonl
RL_PROMPTS_FILE   ?= data/rl_prompts.jsonl

# --- intermediate / output paths (pipeline-generated) ---
SFT_CKPT_DIR      ?= checkpoints/sft
RM_CKPT_DIR       ?= checkpoints/reward_model
PPO_CKPT_DIR      ?= checkpoints/ppo

GENERATED_FILE    ?= data/generated_completions.jsonl
PREFERENCES_FILE  ?= data/preferences.jsonl
RM_TRAIN_FILE     ?= data/rm_train.jsonl
RM_VAL_FILE       ?= data/rm_val.jsonl

# --- knobs tuned for a tiny CPU demo (override any of these at the CLI) ---
DTYPE             ?= float32     # bfloat16/float16 are unreliable on CPU-only training
SFT_EPOCHS        ?= 3
SFT_BATCH_SIZE    ?= 2           # small on purpose: only 6 demo examples
SFT_GRAD_ACCUM    ?= 1           # must divide evenly into the tiny batch count or the optimizer never steps
NUM_SAMPLES        ?= 3          # completions per prompt for Stage B
PREF_MODE         ?= human       # no API key needed; swap to "rlaif" once you're ready (needs ANTHROPIC_API_KEY)
KL_COEF           ?= 0.05
PPO_STEPS         ?= 5           # a handful of PPO steps, just to see the loop run end to end
VAL_FRACTION      ?= 0.2         # small so at least one prompt group lands in val with so few prompts
SEED              ?= 42

STAMP_DIR         := .stamps

.PHONY: all sft generate preferences reward_model ppo clean clean-stamps help

help:
	@echo "Targets: sft generate preferences reward_model ppo all clean"
	@echo "Tuned for a CPU laptop demo — see header comments for the knobs."

all: ppo

$(STAMP_DIR):
	mkdir -p $(STAMP_DIR)
	mkdir -p data checkpoints

# ------------------------------------------------------------------
# Stage 0: SFT
# ------------------------------------------------------------------
$(STAMP_DIR)/sft.done: sft_training.py $(SFT_TRAIN_FILE) $(SFT_VAL_FILE) | $(STAMP_DIR)
	$(PYTHON) sft_training.py \
		--model_name $(BASE_MODEL) \
		--train_file $(SFT_TRAIN_FILE) \
		--val_file $(SFT_VAL_FILE) \
		--output_dir $(SFT_CKPT_DIR) \
		--epochs $(SFT_EPOCHS) \
		--batch_size $(SFT_BATCH_SIZE) \
		--grad_accum_steps $(SFT_GRAD_ACCUM) \
		--dtype $(DTYPE) \
		--seed $(SEED)
	touch $@

sft: $(STAMP_DIR)/sft.done

# ------------------------------------------------------------------
# Stage B: response generation (needs the SFT checkpoint)
# ------------------------------------------------------------------
$(STAMP_DIR)/generate.done: response_generation.py $(STAMP_DIR)/sft.done $(RL_PROMPTS_FILE) | $(STAMP_DIR)
	$(PYTHON) response_generation.py \
		--model_name $(SFT_CKPT_DIR)/final \
		--prompt_file $(RL_PROMPTS_FILE) \
		--output_file $(GENERATED_FILE) \
		--num_samples_per_prompt $(NUM_SAMPLES) \
		--dtype $(DTYPE) \
		--seed $(SEED)
	touch $@

generate: $(STAMP_DIR)/generate.done

# ------------------------------------------------------------------
# Stage B.5: preference collection (human by default for this demo)
# ------------------------------------------------------------------
$(STAMP_DIR)/preferences.done: preference_collection.py $(STAMP_DIR)/generate.done | $(STAMP_DIR)
	$(PYTHON) preference_collection.py \
		--input_file $(GENERATED_FILE) \
		--output_file $(PREFERENCES_FILE) \
		--mode $(PREF_MODE) \
		--seed $(SEED)
	touch $@

# Split preferences.jsonl into train/val for Stage A, by prompt group
# (not by line — see the original makefile.txt comment for why).
$(STAMP_DIR)/pref_split.done: $(STAMP_DIR)/preferences.done | $(STAMP_DIR)
	$(PYTHON) -c "\
import json, random; \
from collections import defaultdict; \
random.seed($(SEED)); \
records = [json.loads(l) for l in open('$(PREFERENCES_FILE)') if l.strip()]; \
by_prompt = defaultdict(list); \
[by_prompt[r['prompt']].append(r) for r in records]; \
prompts = list(by_prompt.keys()); \
random.shuffle(prompts); \
n_val = max(1, int($(VAL_FRACTION) * len(prompts))); \
val_prompts = set(prompts[:n_val]); \
tf = open('$(RM_TRAIN_FILE)', 'w'); \
vf = open('$(RM_VAL_FILE)', 'w'); \
[ (vf if p in val_prompts else tf).write(json.dumps(r) + chr(10)) for p, recs in by_prompt.items() for r in recs ]; \
tf.close(); vf.close(); \
print(f'Split {len(prompts)} prompts -> {len(prompts) - n_val} train / {n_val} val')"
	touch $@

preferences: $(STAMP_DIR)/pref_split.done

# ------------------------------------------------------------------
# Stage A: reward model training (needs the SFT checkpoint + preference split)
# ------------------------------------------------------------------
$(STAMP_DIR)/reward_model.done: reward_model_training.py $(STAMP_DIR)/sft.done $(STAMP_DIR)/pref_split.done | $(STAMP_DIR)
	$(PYTHON) reward_model_training.py \
		--base_model_name $(SFT_CKPT_DIR)/final \
		--train_file $(RM_TRAIN_FILE) \
		--val_file $(RM_VAL_FILE) \
		--output_dir $(RM_CKPT_DIR) \
		--batch_size 2 \
		--grad_accum_steps 1 \
		--dtype $(DTYPE) \
		--seed $(SEED)
	touch $@

reward_model: $(STAMP_DIR)/reward_model.done

# ------------------------------------------------------------------
# Stage C: PPO (needs the SFT checkpoint AND the reward model)
# ------------------------------------------------------------------
$(STAMP_DIR)/ppo.done: ppo_rlhf_training.py $(STAMP_DIR)/sft.done $(STAMP_DIR)/reward_model.done $(RL_PROMPTS_FILE) | $(STAMP_DIR)
	$(PYTHON) ppo_rlhf_training.py \
		--policy_model_name $(SFT_CKPT_DIR)/final \
		--reward_model_path $(RM_CKPT_DIR)/final \
		--prompt_file $(RL_PROMPTS_FILE) \
		--output_dir $(PPO_CKPT_DIR) \
		--total_ppo_steps $(PPO_STEPS) \
		--kl_coef $(KL_COEF) \
		--dtype $(DTYPE) \
		--seed $(SEED)
	touch $@

ppo: $(STAMP_DIR)/ppo.done

# ------------------------------------------------------------------
clean:
	rm -rf $(SFT_CKPT_DIR) $(RM_CKPT_DIR) $(PPO_CKPT_DIR) $(STAMP_DIR)
	rm -f $(GENERATED_FILE) $(PREFERENCES_FILE) $(RM_TRAIN_FILE) $(RM_VAL_FILE)

clean-stamps:
	rm -rf $(STAMP_DIR)
