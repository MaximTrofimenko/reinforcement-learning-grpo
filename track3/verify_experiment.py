"""Check that the saved results follow the protocol without loading a model."""

import hashlib
import json
import math
from pathlib import Path

from task import check_answer

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE / "experiments" / "2026-11-18"


def read(path):
    return json.loads(path.read_text())


def task_key(row):
    return tuple(row["numbers"]), row["target"]


def main():
    protocol = read(EXPERIMENT / "protocol.json")
    for name, digest in protocol["sha256"].items():
        assert hashlib.sha256((HERE / name).read_bytes()).hexdigest() == digest, name
    splits = {
        split: [json.loads(line) for line in (EXPERIMENT / "data" / f"{split}.jsonl").read_text().splitlines()]
        for split in protocol["split_sizes"]
    }
    keys = {split: {task_key(row) for row in rows} for split, rows in splits.items()}
    for split, rows in splits.items():
        assert len(rows) == len(keys[split]) == protocol["split_sizes"][split]
    assert not any(keys[a] & keys[b] for a in keys for b in keys if a < b)
    old = {task_key(json.loads(line)) for path in (HERE / "data").glob("*.jsonl")
           for line in path.read_text().splitlines()}
    assert not old & keys["final_test"]
    results = EXPERIMENT / "results"
    expected = [(row["id"], task_key(row)) for row in splits["final_test"]]
    paths = [results / "base_final.json"]
    task_orders = {}
    for i, seed in enumerate(protocol["training_seeds"]):
        for beta in protocol["kl_betas"]:
            condition = "kl" if beta else "no_kl"
            directory = results if i == 0 and beta else results / (condition if i == 0 else f"{condition}_seed{str(seed)[-2:]}")
            path = results / f"{condition}_final.json" if i == 0 else directory / "final.json"
            paths.append(path)
            config = read(directory / "training_config.json")
            assert config["seed"] == seed and config["kl_beta"] == beta
            assert config["steps"] == protocol["steps"]
            logs = [json.loads(line) for line in (directory / "training_log.jsonl").read_text().splitlines()]
            assert [row["step"] for row in logs] == list(range(1, protocol["steps"] + 1))
            assert all(math.isfinite(row[field]) for row in logs for field in ("loss", "mean_kl", "mean_reward"))
            order = [row["task_id"] for row in logs]
            if seed in task_orders:
                assert task_orders[seed] == order, seed
            task_orders[seed] = order
    for path in paths:
        data = read(path)
        assert [(row["id"], task_key(row)) for row in data["rows"]] == expected, path
        for row in data["rows"]:
            for field, value in check_answer(row, row["response"]).items():
                assert row[field] == value, (path, row["id"], field)
        assert data["metrics"]["count"] == len(expected)
        assert data["metrics"]["accuracy"] == sum(row["correct"] for row in data["rows"]) / len(expected)
    journal = [json.loads(line) for line in (results / "run_journal.jsonl").read_text().splitlines()]
    assert journal[0]["event"] == "protocol_verified"
    assert journal[0]["time_utc"] > protocol["frozen_at_utc"]
    assert journal[-1]["event"] == "complete"
    gate = next(row for row in journal if row["event"] == "baseline_gate")
    assert gate["passed"] and 0.1 <= gate["accuracy"] <= 0.5
    training = [row for row in journal if row.get("stage", "").startswith("train_")]
    finals = [row for row in journal if row.get("stage", "").startswith("final_")]
    assert len(training) == len(finals) == 12
    assert all(gate["time_utc"] < row["time_utc"] for row in training)
    assert max(row["time_utc"] for row in training) < min(row["time_utc"] for row in finals)
    assert all(row["returncode"] == 0 for row in journal if row["event"] == "end")
    print("Verified: frozen protocol, disjoint data, six runs, seven evaluations, rewards, metrics and stage order.")


if __name__ == "__main__":
    main()
