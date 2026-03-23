"""
Evaluation script: computes perplexity and BLEU score on a held-out dataset.

Usage:
    python training/scripts/evaluate.py --config training/configs/eval_config.yaml
"""

import argparse
import json
import math
import yaml

import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from evaluate import load as load_metric


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="training/configs/eval_config.yaml")
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


def compute_perplexity(model, tokenizer, texts: list[str], max_length: int = 512) -> float:
    model.eval()
    device = next(model.parameters()).device
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for text in texts:
            enc = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=max_length
            ).to(device)
            outputs = model(**enc, labels=enc["input_ids"])
            loss = outputs.loss.item()
            num_tokens = enc["input_ids"].numel()
            total_loss += loss * num_tokens
            total_tokens += num_tokens

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    return math.exp(avg_loss)


def compute_bleu(predictions: list[str], references: list[str]) -> float:
    bleu = load_metric("bleu")
    result = bleu.compute(
        predictions=[p.split() for p in predictions],
        references=[[r.split()] for r in references],
    )
    return result["bleu"]


def main():
    args = parse_args()
    cfg = load_config(args.config)

    model_path = cfg.get("model_path", "training/checkpoints")
    dataset_path = cfg.get("dataset_path", "training/datasets/eval.jsonl")
    max_length = cfg.get("max_length", 512)
    max_new_tokens = cfg.get("max_new_tokens", 128)

    print(f"[evaluate] Model   : {model_path}")
    print(f"[evaluate] Dataset : {dataset_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)

    records = load_jsonl(dataset_path)
    instructions = [r.get("instruction", "") for r in records]
    references = [r.get("response", "") for r in records]

    # ── Perplexity ────────────────────────────────────────────────────────────
    full_texts = [f"### Instruction:\n{i}\n\n### Response:\n{r}" for i, r in zip(instructions, references)]
    ppl = compute_perplexity(model, tokenizer, full_texts, max_length)
    print(f"[evaluate] Perplexity : {ppl:.4f}")

    # ── BLEU ──────────────────────────────────────────────────────────────────
    model.eval()
    predictions = []
    with torch.no_grad():
        for instruction in instructions:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
            enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length).to(device)
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            generated = tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
            predictions.append(generated.strip())

    bleu = compute_bleu(predictions, references)
    print(f"[evaluate] BLEU       : {bleu:.4f}")

    print("\n[evaluate] Done.")


if __name__ == "__main__":
    main()