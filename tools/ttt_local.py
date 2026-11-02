"""ttt_local.py - single-task LoRA TTT on local RTX 3060 (plain PEFT, no unsloth).

Mirrors upstream hyperparameters (r=256, lr 5e-5, 1 epoch, batch 1, bf16)
with gradient checkpointing + 6144 ctx to fit 12GB VRAM.

Usage:
  ./.venv-ttt/bin/python tools/ttt_local.py --task 0934a4d8 [--n 16] [--epochs 1]
Compares train loss against Kaggle magnitudes (~0.001-0.1).
"""
import argparse
import json
import sys

import numpy as np

sys.path.insert(0, "src")
from arc_loader import ArcDataset, QwenFormatter  # noqa: E402

USER_TOKEN_ID = 11
ASSISTANT_TOKEN_ID = 12
EOS_ID = 15


class CompletionOnlyCollator:
    """Port of upstream QwenDataCollatorForCompletionOnlyLM (no unsloth)."""

    def __init__(self, tokenizer):
        from transformers import DataCollatorForLanguageModeling
        self.base = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    def __call__(self, examples):
        import torch
        batch = self.base(examples)
        for i in range(len(examples)):
            labels = batch["input_ids"][i].clone()
            user_start = set(np.where(labels == USER_TOKEN_ID)[0].tolist())
            asst_start = set(np.where(labels == ASSISTANT_TOKEN_ID)[0].tolist())
            start_idx = sorted(user_start | asst_start)
            end_idx = np.where(labels == EOS_ID)[0]
            batch["labels"][i, :] = -100
            for j, (s, e) in enumerate(zip(start_idx, end_idx)):
                if j % 2 == 1:
                    batch["labels"][i, s + 2:e + 1] = labels[s + 2:e + 1]
        return batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/home/taizo/Kaggle/models/qwen3_4b_sft")
    ap.add_argument("--task", default="0934a4d8")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-len", type=int, default=6144)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
    from peft import LoraConfig, get_peft_model
    from datasets import Dataset

    tok = AutoTokenizer.from_pretrained(args.model)
    print("vocab:", tok.vocab_size, flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16).to("cuda")
    model.gradient_checkpointing_enable()
    peft_cfg = LoraConfig(
        r=256, lora_alpha=32, lora_dropout=0.0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",
                        "embed_tokens", "lm_head"])
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    fmt = QwenFormatter(tok)
    ds = ArcDataset.from_file("data/arc-agi_evaluation_challenges.json").change_keys([args.task])
    tr = ds.augment(n=args.n, shfl_keys=True, seed=1)
    tr = tr.cut_to_len(formatter=fmt, name="text", max_len=args.max_len)
    hfds = Dataset.from_list(tr.as_list(fmt))
    print("samples:", len(hfds), flush=True)

    def tok_fn(b):
        return tok(b["text"], truncation=True, max_length=args.max_len)
    hfds = hfds.map(tok_fn, batched=True, remove_columns=["train", "query", "reply", "input", "text", "key"])

    targs = TrainingArguments(
        output_dir="/tmp/ttt_out", per_device_train_batch_size=1,
        num_train_epochs=args.epochs, learning_rate=5e-5, optim="adamw_torch",
        lr_scheduler_type="cosine", warmup_steps=10, logging_steps=10,
        save_strategy="no", report_to="none", bf16=True, seed=42)
    trainer = Trainer(model=model, args=targs, train_dataset=hfds,
                      data_collator=CompletionOnlyCollator(tok))
    stats = trainer.train()
    print("FINAL train_loss:", stats.training_loss, flush=True)


if __name__ == "__main__":
    main()
