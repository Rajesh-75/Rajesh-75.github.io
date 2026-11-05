"""
==================================================================
 COMPLETE SUPERVISED FINE-TUNING (SFT) ALGORITHM
==================================================================

Goal of SFT:
    Take a pretrained (base) LLM, which only knows "predict the next
    token", and teach it to follow the (instruction -> response)
    format by training on curated (prompt, response) pairs.

Key idea that trips people up:
    We do NOT compute loss on the prompt tokens. The model should
    learn to GENERATE good responses, not to memorize/predict the
    prompts. So every prompt token's label is masked to -100 (the
    value PyTorch's cross-entropy ignores by default).

    input_ids: [ <prompt tokens> <response tokens> <eos> ]
    labels:    [   -100  -100 ...  <response tokens> <eos> ]

The 6 stages below map 1:1 onto the algorithm-box structure used on
the page:
    1. Data formatting   (raw pairs -> chat-template text)
    2. Tokenization + loss masking
    3. Batching / collation (dynamic padding)
    4. Model + optimizer + LR schedule setup
    5. Training loop (forward, loss, backward, step)
    6. Evaluation + checkpointing
"""

import math
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

IGNORE_INDEX = -100  # PyTorch cross-entropy ignores labels == -100


# ------------------------------------------------------------------
# STAGE 1 + 2: Data formatting, tokenization, and loss masking
# ------------------------------------------------------------------
class SFTDataset(Dataset):
    """
    Each raw example is a dict: {"prompt": ..., "response": ...}

    We build ONE tokenized sequence = prompt_tokens + response_tokens,
    and mask the prompt tokens out of the loss.
    """

    def __init__(self, examples, tokenizer, max_length=1024):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]

        # Apply a chat template so the model sees the exact special
        # tokens / role markers it will see at inference time.
        # e.g. "<|user|>\n{prompt}\n<|assistant|>\n"
        prompt_text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": ex["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        response_text = ex["response"] + self.tokenizer.eos_token

        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        response_ids = self.tokenizer(response_text, add_special_tokens=False)["input_ids"]

        input_ids = prompt_ids + response_ids
        input_ids = input_ids[: self.max_length]

        # Labels: -100 over the prompt span, real token ids over the response span
        labels = [IGNORE_INDEX] * len(prompt_ids) + response_ids[:]
        labels = labels[: self.max_length]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


# ------------------------------------------------------------------
# STAGE 3: Batching with dynamic (per-batch) padding
# ------------------------------------------------------------------
def make_collate_fn(pad_token_id):
    def collate_fn(batch):
        max_len = max(len(x["input_ids"]) for x in batch)

        input_ids, labels, attention_mask = [], [], []
        for x in batch:
            pad_len = max_len - len(x["input_ids"])
            input_ids.append(F.pad(x["input_ids"], (0, pad_len), value=pad_token_id))
            labels.append(F.pad(x["labels"], (0, pad_len), value=IGNORE_INDEX))
            attention_mask.append(
                torch.cat([torch.ones(len(x["input_ids"])), torch.zeros(pad_len)])
            )

        return {
            "input_ids": torch.stack(input_ids),
            "labels": torch.stack(labels),
            "attention_mask": torch.stack(attention_mask).long(),
        }

    return collate_fn


# ------------------------------------------------------------------
# STAGE 6 (helper): Evaluation -> loss and perplexity on held-out data
# ------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        n_tokens = (batch["labels"] != IGNORE_INDEX).sum().item()
        total_loss += out.loss.item() * n_tokens
        total_tokens += n_tokens
    model.train()
    avg_loss = total_loss / max(total_tokens, 1)
    return avg_loss, math.exp(avg_loss)


# ------------------------------------------------------------------
# STAGES 4 + 5: Full training driver
# ------------------------------------------------------------------
def train_sft(
    model_name: str,
    train_examples: list,
    val_examples: list,
    output_dir: str = "./sft_checkpoints",
    epochs: int = 3,
    batch_size: int = 4,
    grad_accum_steps: int = 8,      # effective batch size = batch_size * grad_accum_steps
    lr: float = 2e-5,
    warmup_ratio: float = 0.03,
    max_grad_norm: float = 1.0,
    eval_every_steps: int = 200,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Stage 4a: tokenizer + model ---
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16
    ).to(device)
    model.gradient_checkpointing_enable()  # trade compute for memory
    model.train()

    # --- Data ---
    train_ds = SFTDataset(train_examples, tokenizer)
    val_ds = SFTDataset(val_examples, tokenizer)
    collate_fn = make_collate_fn(tokenizer.pad_token_id)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # --- Stage 4b: optimizer + LR schedule ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    total_optim_steps = (len(train_loader) // grad_accum_steps) * epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(warmup_ratio * total_optim_steps),
        num_training_steps=total_optim_steps,
    )

    # --- Stage 5: training loop ---
    global_step = 0
    for epoch in range(epochs):
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass. HF's CausalLM shifts labels internally and
            # applies cross-entropy ONLY where labels != -100, which is
            # exactly the loss-masking behaviour Stage 2 set up.
            outputs = model(**batch)
            loss = outputs.loss / grad_accum_steps
            loss.backward()

            if (step + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % eval_every_steps == 0:
                    val_loss, val_ppl = evaluate(model, val_loader, device)
                    print(f"[step {global_step}] val_loss={val_loss:.4f} val_ppl={val_ppl:.2f}")

        print(f"=== epoch {epoch + 1}/{epochs} complete ===")
        model.save_pretrained(f"{output_dir}/epoch_{epoch + 1}")
        tokenizer.save_pretrained(f"{output_dir}/epoch_{epoch + 1}")

    return model, tokenizer


# ------------------------------------------------------------------
# Example usage
# ------------------------------------------------------------------
if __name__ == "__main__":
    train_examples = [
        {"prompt": "What is the capital of France?", "response": "The capital of France is Paris."},
        # ... thousands more (prompt, response) pairs
    ]
    val_examples = [
        {"prompt": "What is 2 + 2?", "response": "2 + 2 equals 4."},
    ]

    train_sft(
        model_name="meta-llama/Llama-3.2-1B",
        train_examples=train_examples,
        val_examples=val_examples,
    )