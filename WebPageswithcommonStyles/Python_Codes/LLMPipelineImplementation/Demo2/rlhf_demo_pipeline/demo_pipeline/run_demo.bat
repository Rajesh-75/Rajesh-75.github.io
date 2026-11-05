@echo off
REM ==================================================================
REM RLHF pipeline demo runner for Windows cmd — no "make" required.
REM Mirrors the Makefile's tuned CPU/laptop-demo defaults exactly.
REM Run from an ordinary Command Prompt, inside this folder:
REM     run_demo.bat
REM ==================================================================

setlocal

set PYTHON=python

set BASE_MODEL=HuggingFaceTB/SmolLM2-135M-Instruct
set SFT_TRAIN_FILE=data\sft_train.jsonl
set SFT_VAL_FILE=data\sft_val.jsonl
set RL_PROMPTS_FILE=data\rl_prompts.jsonl

set SFT_CKPT_DIR=checkpoints\sft
set RM_CKPT_DIR=checkpoints\reward_model
set PPO_CKPT_DIR=checkpoints\ppo

set GENERATED_FILE=data\generated_completions.jsonl
set PREFERENCES_FILE=data\preferences.jsonl
set RM_TRAIN_FILE=data\rm_train.jsonl
set RM_VAL_FILE=data\rm_val.jsonl

set DTYPE=float32
set SFT_EPOCHS=3
set SFT_BATCH_SIZE=2
set SFT_GRAD_ACCUM=1
set NUM_SAMPLES=3
set PREF_MODE=human
set KL_COEF=0.05
set PPO_STEPS=5
set VAL_FRACTION=0.2
set SEED=42

if not exist checkpoints mkdir checkpoints
if not exist data mkdir data

echo.
echo === Stage 0: SFT ===
%PYTHON% sft_training.py --model_name %BASE_MODEL% --train_file %SFT_TRAIN_FILE% --val_file %SFT_VAL_FILE% --output_dir %SFT_CKPT_DIR% --epochs %SFT_EPOCHS% --batch_size %SFT_BATCH_SIZE% --grad_accum_steps %SFT_GRAD_ACCUM% --dtype %DTYPE% --seed %SEED%
if errorlevel 1 goto :error

echo.
echo === Stage B: response generation ===
%PYTHON% response_generation.py --model_name %SFT_CKPT_DIR%\final --prompt_file %RL_PROMPTS_FILE% --output_file %GENERATED_FILE% --num_samples_per_prompt %NUM_SAMPLES% --dtype %DTYPE% --seed %SEED%
if errorlevel 1 goto :error

echo.
echo === Stage B.5: preference collection (mode=%PREF_MODE%) ===
%PYTHON% preference_collection.py --input_file %GENERATED_FILE% --output_file %PREFERENCES_FILE% --mode %PREF_MODE% --seed %SEED%
if errorlevel 1 goto :error

echo.
echo === Splitting preferences into train/val ===
%PYTHON% split_preferences.py --preferences_file %PREFERENCES_FILE% --train_file %RM_TRAIN_FILE% --val_file %RM_VAL_FILE% --val_fraction %VAL_FRACTION% --seed %SEED%
if errorlevel 1 goto :error

echo.
echo === Stage A: reward model training ===
%PYTHON% reward_model_training.py --base_model_name %SFT_CKPT_DIR%\final --train_file %RM_TRAIN_FILE% --val_file %RM_VAL_FILE% --output_dir %RM_CKPT_DIR% --batch_size 2 --grad_accum_steps 1 --dtype %DTYPE% --seed %SEED%
if errorlevel 1 goto :error

echo.
echo === Stage C: PPO ===
%PYTHON% ppo_rlhf_training.py --policy_model_name %SFT_CKPT_DIR%\final --reward_model_path %RM_CKPT_DIR%\final --prompt_file %RL_PROMPTS_FILE% --output_dir %PPO_CKPT_DIR% --total_ppo_steps %PPO_STEPS% --kl_coef %KL_COEF% --dtype %DTYPE% --seed %SEED%
if errorlevel 1 goto :error

echo.
echo === Pipeline complete. Final PPO checkpoint: %PPO_CKPT_DIR%\final ===
goto :eof

:error
echo.
echo *** A stage failed (exit code %errorlevel%). Stopping here so earlier checkpoints are left intact. ***
exit /b 1
