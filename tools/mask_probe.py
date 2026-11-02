"""mask_probe.py - replicate upstream stack exactly, inspect labels+loss."""
import sys

sys.path.insert(0, "src")
sys.path.insert(0, "tools")
import numpy as np  # noqa: E402
from arc_loader import ArcDataset, QwenFormatter  # noqa: E402

USER_TOKEN_ID = 11
ASSISTANT_TOKEN_ID = 12
EOS_ID = 15

from transformers import DataCollatorForLanguageModeling  # noqa: E402


class UpstreamCollator(DataCollatorForLanguageModeling):
    def torch_call(self, examples):
        import torch
        batch = super().torch_call(examples)
        for i in range(len(examples)):
            labels = batch["input_ids"][i].clone()
            user_start_idx = np.where(labels == USER_TOKEN_ID)[0].tolist()
            assistant_start_idx = np.where(labels == ASSISTANT_TOKEN_ID)[0].tolist()
            start_idx = sorted(user_start_idx + assistant_start_idx)
            end_idx = np.where(labels == EOS_ID)[0]
            batch["labels"][i, :] = -100
            for j, (start, end) in enumerate(zip(start_idx, end_idx)):
                assert start < end
                if j % 2 == 1:
                    start += 2
                    end += 1
                    batch["labels"][i, start:end] = labels[start:end]
        return batch


def main():
    import torch
    from unsloth import FastLanguageModel, UnslothTrainingArguments, UnslothTrainer
    from datasets import Dataset

    tok = AutoTokenizer = __import__("transformers").AutoTokenizer.from_pretrained(
        "/home/taizo/Kaggle/models/qwen3_4b_sft")
    fmt = QwenFormatter(tok)
    ds = ArcDataset.from_file("data/arc-agi_evaluation_challenges.json").change_keys(["0934a4d8"])
    tr = ds.augment(n=1, shfl_keys=True, seed=1).cut_to_len(
        formatter=fmt, name="text", max_len=6144)
    lst = tr.as_list(fmt)[:4]
    print("n samples:", len(lst), flush=True)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="/home/taizo/Kaggle/models/qwen3_4b_sft",
        full_finetuning=False, load_in_4bit=False,
        use_gradient_checkpointing="unsloth",
        max_seq_length=6144)
    model = FastLanguageModel.get_peft_model(
        model, r=8,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=16, lora_dropout=0, bias="none",
        use_gradient_checkpointing="unsloth", random_state=42,
        use_rslora=True, loftq_config=None)

    # inspect ONE batch exactly as trainer would build it (pre-tokenized,
    # as UnslothTrainer does internally via dataset_text_field)
    col = UpstreamCollator(tokenizer=tokenizer, mlm=False)
    ex = [dict(tok(x["text"])) for x in lst]
    batch = col(ex)
    lab = batch["labels"]
    import torch as _t
    frac = [(lab[i] != -100).float().mean().item() for i in range(lab.shape[0])]
    print("unmasked frac per sample:", [round(f, 4) for f in frac], flush=True)
    print("input lens:", batch["input_ids"].shape, flush=True)

    args = UnslothTrainingArguments(
        per_device_train_batch_size=1, num_train_epochs=1,
        warmup_steps=2, learning_rate=5e-5, optim="adamw_torch",
        lr_scheduler_type="cosine", seed=42, report_to="none",
        save_strategy="no", logging_steps=1, fp16=False, bf16=True)
    trn = UnslothTrainer(model=model, tokenizer=tokenizer,
                         data_collator=col,
                         train_dataset=Dataset.from_list(lst),
                         dataset_text_field="text",
                         max_seq_length=6144, args=args)
    stats = trn.train()
    print("FINAL:", stats.training_loss, flush=True)


if __name__ == "__main__":
    main()
