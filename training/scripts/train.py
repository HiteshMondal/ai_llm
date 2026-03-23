"""
Basic training script using HuggingFace Transformers + Trainer API.
Trains (or fine-tunes) a causal language model on a plain-text dataset.

Usage:
    python training/scripts/train.py --config training/configs/train_config.yaml
"""

import argparse
import yaml
from pathlib import Path

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="training/configs/train_config.yaml")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    cfg = load_config(args.config)

    model_name = cfg.get("model_name", "gpt2")
    dataset_path = cfg.get("dataset_path", "training/datasets/train.txt")
    output_dir = cfg.get("output_dir", "training/checkpoints")
    num_epochs = cfg.get("num_epochs", 3)
    batch_size = cfg.get("batch_size", 4)
    max_length = cfg.get("max_length", 512)
    lr = cfg.get("learning_rate", 5e-5)

    print(f"[train] Model      : {model_name}")
    print(f"[train] Dataset    : {dataset_path}")
    print(f"[train] Output dir : {output_dir}")

    # Tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Dataset
    dataset = load_dataset("text", data_files={"train": dataset_path}, split="train")

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        save_strategy="epoch",
        logging_dir=cfg.get("log_dir", "training/logs"),
        logging_steps=10,
        fp16=cfg.get("fp16", False),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    print("[train] Starting training...")
    trainer.train()

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[train] Model saved to {output_dir}")


if __name__ == "__main__":
    main()