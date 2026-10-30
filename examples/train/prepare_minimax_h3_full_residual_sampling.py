#!/usr/bin/env python3
"""Create a repeat-aware index over every approved train row without touching cache."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random

import pyarrow.parquet as pq


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditioned-dir", type=Path, required=True)
    parser.add_argument("--train-indices", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--non-multicam-repeats", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261003)
    args = parser.parse_args()
    if args.non_multicam_repeats < 1:
        raise ValueError("non-multicam-repeats must be positive")
    approved = set(json.loads(args.train_indices.read_text()))
    all_rows: list[int] = []
    source_counts: Counter[str] = Counter()
    draws: list[int] = []
    row_number = 0
    for shard in sorted(args.conditioned_dir.rglob("*.parquet")):
        for sample_id in pq.read_table(shard, columns=["id"]).column("id").to_pylist():
            if row_number in approved:
                source = sample_id.split("/", 1)[0]
                repeats = 1 if source == "train_v2_multicam_video" else args.non_multicam_repeats
                draws.extend([row_number] * repeats)
                all_rows.append(row_number)
                source_counts[source] += 1
            row_number += 1
    if len(all_rows) != len(approved) or len(all_rows) != len(set(all_rows)) or row_number != 65389:
        raise ValueError("Approved train indices do not match the verified 65,389-row cache")
    while len(draws) % 8:
        # Pad with repeats, never remove an approved unique row.
        draws.append(all_rows[len(draws) % len(all_rows)])
    random.Random(args.seed).shuffle(draws)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "sampling_indices.json").write_text(json.dumps(draws) + "\n")
    report = {
        "approved_unique_train_rows": len(all_rows),
        "all_conditioned_rows": row_number,
        "draws_per_epoch": len(draws),
        "source_counts_unique": dict(sorted(source_counts.items())),
        "multicam_draw_fraction": source_counts["train_v2_multicam_video"] / len(draws),
        "non_multicam_repeats": args.non_multicam_repeats,
    }
    (args.output_dir / "sampling_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
