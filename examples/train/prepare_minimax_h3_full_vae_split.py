#!/usr/bin/env python3
"""Freeze scene-disjoint row indices for the fully conditioned H3 VAE cache."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq


def _jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditioned-dir", type=Path, required=True)
    parser.add_argument("--source-mapping", type=Path, required=True)
    parser.add_argument("--train-ids", type=Path, required=True)
    parser.add_argument("--validation-ids", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=65389)
    parser.add_argument("--seed", type=int, default=20260927)
    args = parser.parse_args()

    splits = {}
    for split, path in (("train", args.train_ids), ("validation", args.validation_ids)):
        for item in _jsonl(path):
            global_id = item["global_id"]
            if global_id in splits:
                raise ValueError(f"Duplicate or cross-split global ID: {global_id}")
            splits[global_id] = split

    source_to_split = {}
    for item in _jsonl(args.source_mapping):
        source_id = f"{item['source_name']}/{item['sample_id']}"
        if source_id in source_to_split:
            raise ValueError(f"Duplicate source ID: {source_id}")
        source_to_split[source_id] = splits[item["global_id"]]

    paths = sorted(args.conditioned_dir.rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No conditioned shards in {args.conditioned_dir}")
    indices = {"train": [], "validation": []}
    seen = set()
    source_counts = Counter()
    row_index = 0
    for path in paths:
        ids = pq.read_table(path, columns=["id"]).column("id").to_pylist()
        if not 0 < len(ids) <= 16:
            raise ValueError(f"Unexpected row count in {path}: {len(ids)}")
        for source_id in ids:
            if source_id in seen:
                raise ValueError(f"Duplicate conditioned ID: {source_id}")
            seen.add(source_id)
            try:
                split = source_to_split[source_id]
            except KeyError as exc:
                raise ValueError(f"Conditioned ID absent from approved split: {source_id}") from exc
            indices[split].append(row_index)
            source_counts[source_id.split("/", 1)[0]] += 1
            row_index += 1

    if row_index != args.expected_rows or seen != set(source_to_split):
        raise ValueError(
            f"Conditioned split incomplete: rows={row_index}, expected={args.expected_rows}, "
            f"missing={len(set(source_to_split) - seen)}, extra={len(seen - set(source_to_split))}"
        )
    random.Random(args.seed).shuffle(indices["train"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, values in indices.items():
        (args.output_dir / f"{split}_indices.json").write_text(json.dumps(values) + "\n", encoding="utf-8")
    report = {
        "conditioned_dir": str(args.conditioned_dir.resolve()),
        "total_rows": row_index,
        "train_rows": len(indices["train"]),
        "validation_rows": len(indices["validation"]),
        "source_counts": dict(sorted(source_counts.items())),
        "shards": len(paths),
    }
    (args.output_dir / "split_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
