"""Create and freeze train, development and test tasks before model training."""

from __future__ import annotations

import json
import random
from pathlib import Path

from task import task_pool


DATA_DIR = Path(__file__).resolve().parent / "data"
SPLIT_SIZES = {"train": 400, "dev": 50, "test": 220, "final_test": 220}
DATA_SEED = 20261114


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    paths = {split: DATA_DIR / f"{split}.jsonl" for split in SPLIT_SIZES}
    tasks = task_pool()
    if len(tasks) < sum(SPLIT_SIZES.values()):
        raise RuntimeError(f"Need {sum(SPLIT_SIZES.values())} tasks, got {len(tasks)}")
    random.Random(DATA_SEED).shuffle(tasks)
    offset = 0
    for split, size in SPLIT_SIZES.items():
        contents = "".join(
            json.dumps({"id": f"{split}-{index:04d}", **task}) + "\n"
            for index, task in enumerate(tasks[offset : offset + size])
        )
        if paths[split].exists():
            if paths[split].read_text(encoding="utf-8") != contents:
                raise RuntimeError(f"Existing split has changed: {paths[split]}")
        else:
            paths[split].write_text(contents, encoding="utf-8")
        offset += size
        print(f"{split}: {size} tasks -> {paths[split]}")


if __name__ == "__main__":
    main()
