"""Recheck Countdown on 220 nontrivial tasks using saved model weights."""

import argparse
import itertools
import json
import random
import re
from pathlib import Path

from task import check_answer


HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE / "experiments/2026-11-18"
EXTRA = HERE / "experiments/countdown_check"
MODELS = {
    "base": "base_final.json",
    "kl_18": "kl_final.json", "no_kl_18": "no_kl_final.json",
    "kl_19": "kl_seed19/final.json", "no_kl_19": "no_kl_seed19/final.json",
    "kl_20": "kl_seed20/final.json", "no_kl_20": "no_kl_seed20/final.json",
}


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def key(task):
    return tuple(sorted(task["numbers"])), task["target"]


def build_extra():
    old = {key(task) for folder in (HERE / "data", EXPERIMENT / "data")
           for path in folder.glob("*.jsonl") for task in rows(path)}
    pool = set()
    for numbers in itertools.combinations_with_replacement(range(2, 13), 3):
        for a, b, c in itertools.permutations(numbers):
            for goal in (a*b+c, a*b-c, (a+b)*c):
                if 5 <= goal <= 60 and goal not in numbers and goal != sum(numbers):
                    pool.add((numbers, goal))
    selected = random.Random(20261123).sample(sorted(pool - old), 60)
    return [{"id": f"extra-{i:04d}", "numbers": list(ns), "target": goal}
            for i, (ns, goal) in enumerate(selected)]


def pdf_score(task, response):
    # The assignment allows reasoning before the answer and uses the last pair of tags.
    matches = list(re.finditer(r"<answer>(.*?)</answer>", response, re.DOTALL))
    if not matches or response[matches[-1].end():].strip():
        return False
    return check_answer(task, matches[-1].group())["correct"]


def generate_extra_answers(tasks):
    import torch
    from peft import PeftModel
    from evaluate import evaluate
    from model_utils import load_base

    for name, filename in MODELS.items():
        output = EXTRA / f"{name}.json"
        if output.exists():
            continue
        tokenizer, model, device = load_base()
        if name != "base":
            directory = Path(filename).parent
            if name == "no_kl_18":
                directory = Path("no_kl")
            model = PeftModel.from_pretrained(model, EXPERIMENT / "results" / directory / "adapter")
            model.eval()
        print(name, device, flush=True)
        with torch.inference_mode():
            evaluate(model, tokenizer, device, tasks, output)
        del model, tokenizer


def summarize(tasks):
    original = rows(EXPERIMENT / "data/final_test.jsonl")
    expected = [task for task in original if task["target"] != sum(task["numbers"])] + tasks
    assert len(expected) == len(set(map(key, expected))) == 220
    for name, filename in MODELS.items():
        saved = json.loads((EXPERIMENT / "results" / filename).read_text())["rows"]
        assert [key(r) for r in saved] == [key(t) for t in original]
        filtered = [r for r in saved if r["target"] != sum(r["numbers"])]
        added = json.loads((EXTRA / f"{name}.json").read_text())["rows"]
        assert [(r["id"], key(r)) for r in filtered + added] == [(t["id"], key(t)) for t in expected]
        counts = [sum(pdf_score(r, r["response"]) for r in group) for group in (filtered, added)]
        changed = sum(pdf_score(r, r["response"]) != r["correct"] for r in saved + added)
        print(f"{name}: {counts[0]}/160 + {counts[1]}/60 = {sum(counts)}/220; changed by tag rule: {changed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "evaluate", "summary"))
    mode = parser.parse_args().mode
    tasks = build_extra()
    path = EXTRA / "tasks.jsonl"
    if mode == "prepare" and not path.exists():
        EXTRA.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(t) + "\n" for t in tasks))
    assert rows(path) == tasks, "Extra tasks differ from the fixed selection"
    if mode == "evaluate":
        generate_extra_answers(tasks)
    if mode != "prepare":
        summarize(tasks)
