"""Make four learning curves and summarize three seeds per condition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
STEPS = [0, 50, 100, 150, 200]
SEEDS = [20261114, 20261115, 20261116]
COLORS = {"kl": "#1768ac", "no_kl": "#d67b20"}
LABELS = {"kl": "GRPO, KL = 0,02", "no_kl": "GRPO, KL = 0"}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def moving_average(values: list[float], window: int = 20) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return np.array([array[max(0, i - window + 1) : i + 1].mean() for i in range(len(array))])


def mean_sd(values: list[float]) -> dict:
    return {"mean": float(np.mean(values)), "sd": float(np.std(values, ddof=1)), "by_seed": dict(zip(SEEDS, values))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=HERE / "results")
    args = parser.parse_args()
    results = args.results_dir
    run_dirs = {
        "kl": [results, results / "kl_seed15", results / "kl_seed16"],
        "no_kl": [results / "no_kl", results / "no_kl_seed15", results / "no_kl_seed16"],
    }
    logs = {
        key: [[json.loads(line) for line in (directory / "training_log.jsonl").read_text().splitlines()] for directory in dirs]
        for key, dirs in run_dirs.items()
    }
    base = read_json(results / "base_final.json")
    finals = {
        "kl": [read_json(results / "kl_final.json")]
        + [read_json(directory / "final.json") for directory in run_dirs["kl"][1:]],
        "no_kl": [read_json(results / "no_kl_final.json")]
        + [read_json(directory / "final.json") for directory in run_dirs["no_kl"][1:]],
    }
    base_dev = read_json(results / "base_dev.json")["metrics"]["accuracy"]
    dev_accuracy = {
        key: [
            [base_dev] + [read_json(directory / f"dev_step_{step}.json")["metrics"]["accuracy"] for step in STEPS[1:]]
            for directory in dirs
        ]
        for key, dirs in run_dirs.items()
    }

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4))
    curves = [
        (axes[0, 0], "mean_reward", "Награда за ответ", "Награда (среднее за 20 шагов)"),
        (axes[1, 0], "mean_kl", "KL к исходной модели", "Оценка KL (среднее за 20 шагов)"),
        (axes[1, 1], "mean_tokens", "Длина ответа", "Токены (среднее за 20 шагов)"),
    ]
    for key, run_logs in logs.items():
        for ax, field, _, _ in curves:
            smoothed = np.array([moving_average([row[field] for row in rows]) for rows in run_logs])
            x = np.arange(1, smoothed.shape[1] + 1)
            center = smoothed.mean(axis=0)
            spread = smoothed.std(axis=0, ddof=1)
            ax.plot(x, center, label=LABELS[key], color=COLORS[key], linewidth=2)
            ax.fill_between(x, center - spread, center + spread, color=COLORS[key], alpha=0.15)
        values = np.array(dev_accuracy[key]) * 100
        center = values.mean(axis=0)
        spread = values.std(axis=0, ddof=1)
        axes[0, 1].plot(STEPS, center, "o-", label=LABELS[key], color=COLORS[key], linewidth=2)
        axes[0, 1].fill_between(STEPS, center - spread, center + spread, color=COLORS[key], alpha=0.15)
    for ax, _, title, ylabel in curves:
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Шаг GRPO")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    axes[0, 1].set_title("Точность на 50 проверочных задачах")
    axes[0, 1].set_xlabel("Шаг GRPO")
    axes[0, 1].set_ylabel("Точные решения, %")
    axes[0, 1].set_ylim(0, 45)
    axes[0, 1].grid(alpha=0.25)
    axes[0, 1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(results / "learning_curves.png", dpi=170)
    plt.close(fig)

    base_ids = [row["id"] for row in base["rows"]]
    for condition_runs in finals.values():
        for run in condition_runs:
            assert [row["id"] for row in run["rows"]] == base_ids
    summary = {
        "seeds": SEEDS,
        "base_final": base["metrics"],
        "final_test": {
            key: {field: mean_sd([run["metrics"][field] for run in condition_runs]) for field in ("accuracy", "format_rate", "numbers_rate", "mean_tokens")}
            for key, condition_runs in finals.items()
        },
        "dev_accuracy": {
            key: {str(step): mean_sd([values[i] for values in runs]) for i, step in enumerate(STEPS)}
            for key, runs in dev_accuracy.items()
        },
        "last_50_steps": {
            key: {field: mean_sd([float(np.mean([row[field] for row in rows[-50:]])) for rows in run_logs]) for field in ("mean_reward", "mean_kl", "mean_tokens")}
            for key, run_logs in logs.items()
        },
        "accuracy_differences": {
            key: mean_sd([run["metrics"]["accuracy"] - base["metrics"]["accuracy"] for run in condition_runs])
            for key, condition_runs in finals.items()
        },
        "paired_condition_difference": mean_sd(
            [b["metrics"]["accuracy"] - a["metrics"]["accuracy"] for a, b in zip(finals["kl"], finals["no_kl"])]
        ),
        "error_reasons": {
            key: {
                reason: mean_sd([sum(row["reason"] == reason for row in run["rows"]) for run in condition_runs])
                for reason in ("correct", "target", "numbers", "format")
            }
            for key, condition_runs in finals.items()
        },
    }
    (results / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
