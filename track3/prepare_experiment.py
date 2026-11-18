"""Freeze a new experiment before any model evaluation or training."""

import hashlib
import itertools
import json
from pathlib import Path
import random
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE / "experiments" / "2026-11-18"
DATA_SEED = 20261118
SEEDS = [20261118, 20261119, 20261120]


def key(task):
    return tuple(task["numbers"]), task["target"]


def main():
    if EXPERIMENT.exists():
        raise SystemExit("Experiment already exists; frozen files must not be replaced.")
    # The same expression templates as before, now including repeated numbers.
    pool = set()
    for numbers in itertools.combinations_with_replacement(range(2, 13), 3):
        for a, b, c in set(itertools.permutations(numbers)):
            for target in (a + b + c, a * b + c, a * b - c, (a + b) * c):
                if 5 <= target <= 60 and target not in numbers:
                    pool.add((numbers, target))
    old = {
        key(json.loads(line))
        for path in (HERE / "data").glob("*.jsonl")
        for line in path.read_text().splitlines()
    }
    rng = random.Random(DATA_SEED)
    fresh = sorted(pool - old)
    rng.shuffle(fresh)
    final = fresh[:220]
    remaining = sorted(pool - set(final))
    rng.shuffle(remaining)
    splits = {"train": remaining[:400], "dev": remaining[400:450], "final_test": final}
    assert len(final) == 220 and not set(final) & old
    data = EXPERIMENT / "data"
    data.mkdir(parents=True)
    for split, tasks in splits.items():
        rows = [{"id": f"{split}-{i:04d}", "numbers": list(nums), "target": target}
                for i, (nums, target) in enumerate(tasks)]
        (data / f"{split}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
    files = [HERE / name for name in (
        "task.py", "model_utils.py", "train_grpo.py", "evaluate.py", "analyze.py",
        "prepare_experiment.py", "run_experiment.py", "requirements.txt",
    )] + sorted(data.glob("*.jsonl")) + sorted((HERE / "data").glob("*.jsonl"))
    protocol = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_seed": DATA_SEED, "training_seeds": SEEDS,
        "model": "Qwen/Qwen3-0.6B",
        "model_revision": "c1899de289a04d12100db370d81485cdf75e47ca",
        "steps": 200, "eval_every": 50, "group_size": 4,
        "kl_betas": [0.02, 0.0], "learning_rate": 5e-5,
        "lora": {"r": 8, "alpha": 16, "dropout": 0.0, "targets": ["q_proj", "v_proj"]},
        "generation": {"max_new_tokens": 64, "training_temperature": 0.8,
                       "training_top_p": 0.95, "evaluation_do_sample": False, "thinking": False},
        "split_sizes": {name: len(rows) for name, rows in splits.items()},
        "final_test_overlap_with_previous_data": 0,
        "baseline_accuracy_gate": [0.10, 0.50],
        "analysis": "All three paired seeds; mean and sample standard deviation. No checkpoint selection.",
        "sha256": {str(path.relative_to(HERE)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
    }
    (EXPERIMENT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(json.dumps({k: v for k, v in protocol.items() if k != "sha256"}, indent=2))


if __name__ == "__main__":
    main()
