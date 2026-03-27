"""Append-only TSV experiment log — one row per training run.

Every run (train.py, validate_fm.py) appends a row with timing, loss, eval metrics.
Human-readable, grep-able, easy to paste into spreadsheets.

File: experiments.tsv (project root)
"""

from __future__ import annotations

import datetime
from pathlib import Path

from loguru import logger

LOG_PATH = Path("experiments.tsv")

COLUMNS = [
    "timestamp",
    "env",
    "policy",
    "backbone",
    "norm",
    "steps",
    "batch_size",
    "n_params",
    "final_loss",
    "success_rate",
    "avg_reward",
    "train_s",
    "steps_per_s",
    "samples_per_s",
    "gpu",
    "notes",
]


def log_experiment(
    *,
    env: str,
    policy: str,
    backbone: str = "",
    norm: str = "",
    steps: int = 0,
    batch_size: int = 0,
    n_params: int = 0,
    final_loss: float = 0.0,
    success_rate: float = 0.0,
    avg_reward: float = 0.0,
    train_s: float = 0.0,
    gpu: str = "",
    notes: str = "",
):
    """Append one row to experiments.tsv."""
    steps_per_s = steps / train_s if train_s > 0 else 0
    samples_per_s = steps_per_s * batch_size

    row = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "env": env,
        "policy": policy,
        "backbone": backbone,
        "norm": norm,
        "steps": steps,
        "batch_size": batch_size,
        "n_params": n_params,
        "final_loss": f"{final_loss:.6f}",
        "success_rate": f"{success_rate:.1%}",
        "avg_reward": f"{avg_reward:.3f}",
        "train_s": f"{train_s:.0f}",
        "steps_per_s": f"{steps_per_s:.1f}",
        "samples_per_s": f"{samples_per_s:.0f}",
        "gpu": gpu,
        "notes": notes,
    }

    write_header = not LOG_PATH.exists()
    with open(LOG_PATH, "a") as f:
        if write_header:
            f.write("\t".join(COLUMNS) + "\n")
        f.write("\t".join(str(row[c]) for c in COLUMNS) + "\n")

    logger.info(f"Experiment logged to {LOG_PATH}")
