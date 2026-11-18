"""Run the frozen protocol and record the actual order of all stages."""

from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE / "experiments" / "2026-11-18"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENT / "results")
    args = parser.parse_args()
    protocol = json.loads((EXPERIMENT / "protocol.json").read_text())
    for name, digest in protocol["sha256"].items():
        if hashlib.sha256((HERE / name).read_bytes()).hexdigest() != digest:
            raise SystemExit(f"Frozen file changed: {name}")
    results = args.output_dir.resolve()
    results.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, TRACK3_DATA_DIR=str(EXPERIMENT / "data"), PYTHONUNBUFFERED="1")
    journal = results / "run_journal.jsonl"
    frozen_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE, text=True).strip()

    def record(**fields):
        with journal.open("a") as log:
            log.write(json.dumps({"time_utc": datetime.now(timezone.utc).isoformat(), **fields}) + "\n")

    def run(stage, *args):
        started = time.monotonic()
        command = [sys.executable, *map(str, args)]
        record(stage=stage, event="start", command=command)
        print(f"\nSTAGE {stage}", flush=True)
        completed = subprocess.run(command, cwd=HERE, env=env)
        record(stage=stage, event="end", returncode=completed.returncode,
               seconds=round(time.monotonic() - started, 2))
        if completed.returncode:
            raise SystemExit(completed.returncode)

    record(event="protocol_verified", frozen_commit=frozen_commit,
           protocol_sha256=hashlib.sha256((EXPERIMENT / "protocol.json").read_bytes()).hexdigest())
    run("base_final", "evaluate.py", "--split", "final_test", "--output", results / "base_final.json")
    accuracy = json.loads((results / "base_final.json").read_text())["metrics"]["accuracy"]
    lower, upper = protocol["baseline_accuracy_gate"]
    record(event="baseline_gate", accuracy=accuracy, passed=lower <= accuracy <= upper)
    if not lower <= accuracy <= upper:
        raise SystemExit("Base accuracy is outside 10–50%; no training was started.")
    run("base_dev", "evaluate.py", "--split", "dev", "--output", results / "base_dev.json")
    jobs = []
    for i, seed in enumerate(protocol["training_seeds"]):
        for beta in protocol["kl_betas"]:
            condition = "kl" if beta else "no_kl"
            directory = results if i == 0 and beta else results / (condition if i == 0 else f"{condition}_seed{str(seed)[-2:]}")
            final = results / f"{condition}_final.json" if i == 0 else directory / "final.json"
            jobs.append((seed, beta, directory, final))
    # Finish all training before opening any tuned model's final-test results.
    for seed, beta, directory, _ in jobs:
        config = {"seed": seed, "kl_beta": beta, "steps": protocol["steps"], "frozen_commit": frozen_commit}
        directory.mkdir(exist_ok=True)
        (directory / "planned_config.json").write_text(json.dumps(config, indent=2))
        run(f"train_seed{seed}_beta{beta}", "train_grpo.py", "--seed", seed, "--kl-beta", beta,
            "--steps", protocol["steps"], "--eval-every", protocol["eval_every"], "--output", directory)
    for seed, beta, directory, final in jobs:
        run(f"final_seed{seed}_beta{beta}", "evaluate.py", "--split", "final_test",
            "--adapter", directory / "adapter", "--output", final)
    run("analysis", "analyze.py", "--results-dir", results, "--seeds", *protocol["training_seeds"])
    record(event="complete")


if __name__ == "__main__":
    main()
