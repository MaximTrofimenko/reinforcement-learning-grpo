"""Evaluate a base or LoRA model on a frozen split without training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from model_utils import generate_one, load_base
from task import check_answer


HERE = Path(__file__).resolve().parent


def read_tasks(split: str) -> list[dict]:
    return [json.loads(line) for line in (HERE / "data" / f"{split}.jsonl").read_text().splitlines()]


def evaluate(model, tokenizer, device: str, tasks: list[dict], output: Path) -> dict:
    rows = []
    for index, task in enumerate(tasks, 1):
        response, token_count = generate_one(model, tokenizer, task, device)
        check = check_answer(task, response)
        rows.append({"id": task["id"], "numbers": task["numbers"], "target": task["target"], "response": response, "tokens": token_count, **check})
        if index % 25 == 0 or index == len(tasks):
            print(f"Evaluated {index}/{len(tasks)}; accuracy = {sum(x['correct'] for x in rows) / len(rows):.3f}", flush=True)
    metrics = {
        "count": len(rows),
        "accuracy": sum(x["correct"] for x in rows) / len(rows),
        "format_rate": sum(x["format_ok"] for x in rows) / len(rows),
        "numbers_rate": sum(x["numbers_ok"] for x in rows) / len(rows),
        "mean_tokens": sum(x["tokens"] for x in rows) / len(rows),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"metrics": metrics, "rows": rows}, ensure_ascii=False, indent=2))
    print(metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("dev", "test", "final_test"), required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tokenizer, model, device = load_base()
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
        model.to(device)
        model.eval()
    print(f"Device: {device}; model: {model.config._name_or_path}", flush=True)
    evaluate(model, tokenizer, device, read_tasks(args.split), args.output)


if __name__ == "__main__":
    with torch.inference_mode():
        main()
