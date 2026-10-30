#!/usr/bin/env python3
"""Freeze a direction-balanced pilot from already completed H3 VAE shards."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq


ACTIONS = "WASDIJKL"
SPEED_BINS = ("slow", "medium", "fast")


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def stable_key(identity: str) -> str:
    return hashlib.sha256(f"20260926:{identity}".encode()).hexdigest()


def select_balanced(rows: list[dict], count: int) -> list[dict]:
    by_speed: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        action = next(iter(row["actions"].values()))
        by_speed[action["speed_bin"]].append(row)
    for speed in SPEED_BINS:
        by_speed[speed].sort(key=lambda row: stable_key(row["global_id"]))
    chosen = []
    while len(chosen) < count:
        progressed = False
        for speed in SPEED_BINS:
            if len(chosen) >= count:
                break
            if by_speed[speed]:
                chosen.append(by_speed[speed].pop())
                progressed = True
        if not progressed:
            raise ValueError(f"only {len(chosen)} available, need {count}")
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-plan", type=Path, required=True)
    parser.add_argument("--source-mapping", type=Path, required=True)
    parser.add_argument("--vae-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-per-direction", type=int, default=128)
    parser.add_argument("--validation-per-direction", type=int, default=6)
    args = parser.parse_args()

    paths = sorted(args.vae_root.rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no completed VAE shards under {args.vae_root}")
    available = set()
    for path in paths:
        ids = pq.read_table(path, columns=["id"])["id"].to_pylist()
        if available.intersection(ids):
            raise ValueError(f"duplicate VAE IDs in {path}")
        available.update(ids)

    mapping = {
        row["global_id"]: f"{row['source_name']}/{row['sample_id']}"
        for row in read_jsonl(args.source_mapping)
    }
    result = {}
    speed_counts = Counter()
    source_counts = Counter()
    scene_groups = {}
    for split, per_direction in (("train", args.train_per_direction),
                                 ("validation", args.validation_per_direction)):
        eligible = defaultdict(list)
        for row in read_jsonl(args.original_plan / f"{split}_ids.jsonl"):
            if len(row["actions"]) != 1 or mapping.get(row["global_id"]) not in available:
                continue
            action = next(iter(row["actions"]))
            eligible[action].append(row)
        selected = []
        for action in ACTIONS:
            if len(eligible[action]) < per_direction:
                raise ValueError(f"{split}/{action}: {len(eligible[action])} < {per_direction}")
            selected.extend(select_balanced(eligible[action], per_direction))
        selected.sort(key=lambda row: stable_key(row["global_id"]))
        result[split] = selected
        scene_groups[split] = {row["scene_key"] for row in selected}
        for row in selected:
            action, label = next(iter(row["actions"].items()))
            speed_counts[(split, action, label["speed_bin"])] += 1
            source_counts[(split, row["source_family"])] += 1
    if scene_groups["train"] & scene_groups["validation"]:
        raise ValueError("scene leakage between training and validation")
    if {row["global_id"] for row in result["train"]} & {row["global_id"] for row in result["validation"]}:
        raise ValueError("ID leakage between training and validation")

    args.output.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation"):
        write_jsonl(args.output / f"pilot_{split}_ids.jsonl", result[split])
    report = {
        "status": "completed_vae_shards_only_automatic_qc_pass_human_review_pending",
        "vae_shards_seen": len(paths),
        "available_vae_rows": len(available),
        "pilot_train": len(result["train"]),
        "pilot_validation": len(result["validation"]),
        "speed_bins": {"/".join(key): value for key, value in sorted(speed_counts.items())},
        "source_counts": {"/".join(key): value for key, value in sorted(source_counts.items())},
        "scene_groups": {split: len(groups) for split, groups in scene_groups.items()},
    }
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
