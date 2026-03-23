"""
LoRA fine-tuning script using PEFT + HuggingFace Transformers.
Fine-tunes a causal LM on instruction-style JSONL data.

Each line in the dataset should be JSON: {"instruction": "...", "response": "..."}

Usage:
    python training/scripts/fine_tune.py --config training/configs/fine_tune_config.yaml
"""

import argparse
import json
import yaml
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import get_peft_model, LoraConfig, TaskType


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="training/configs/fine_tune_config.yaml")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def format_prompt(record: dict) -> str:
    instruction = record.get("instruction", "")
    response = record.get("response", "")
    return f"### Instruction:\n{instruction}\n\n### Response:\n{response}"


def main():
    args = parse_args()
    cfg = load_config(args.config)

    model_name = cfg.get("model_name", "gpt2")
    dataset_path = cfg.get("dataset_path", "training/datasets/finetune.jsonl")
    output_dir = cfg.get("output_dir", "training/checkpoints/lora")
    num_epochs = cfg.get("num_epochs", 3)
    batch_size = cfg.get("batch_size", 2)
    max_length = cfg.get("max_length", 512)
    lr = cfg.get("learning_rate", 2e-4)

    # LoRA config
    lora_r = cfg.get("lora_r", 8)
    lora_alpha = cfg.get("lora_alpha", 16)
    lora_dropout = cfg.get("lora_dropout", 0.05)

    print(f"[fine_tune] Model      : {model_name}")
    print(f"[fine_tune] Dataset    : {dataset_path}")
    print(f"[fine_tune] Output dir : {output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if cfg.get("fp16", False) else torch.float32,
    )

    # Apply LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Dataset
    records = load_jsonl(dataset_path)
    texts = [format_prompt(r) for r in records]
    raw_dataset = Dataset.from_dict({"text": texts})

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

    tokenized = raw_dataset.map(tokenize, batched=True, remove_columns=["text"])

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

    print("[fine_tune] Starting LoRA fine-tuning...")
    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[fine_tune] LoRA adapter saved to {output_dir}")


if __name__ == "__main__":
    main()