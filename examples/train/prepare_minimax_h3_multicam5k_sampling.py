#!/usr/bin/env python3
"""Freeze a 5,000-multicam subset of the verified H3 training cache.

All approved non-multicam training rows are retained. The conditioned Parquet,
source videos, camera paths, and VAE latents are read-only.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import json
from pathlib import Path
import random

import pyarrow.parquet as pq


DIRECTIONS = "WASDIJKL"
SPEEDS = ("slow", "medium", "fast")
MULTICAM = "train_v2_multicam_video"


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def strong_labels(actions: dict) -> set[tuple[str, str]]:
    return {
        (direction, action["speed_bin"])
        for direction, action in actions.items()
        if direction in DIRECTIONS and action.get("speed_bin") in SPEEDS
        and float(action.get("axis_fraction", 0)) >= 0.35
        and float(action.get("direction_consistency", 0)) >= 0.65
    }


def select_multicam(rows: list[dict], initial_counts: Counter, count: int,
                    rng: random.Random) -> list[dict]:
    buckets: dict[tuple[str, str], deque] = {}
    for label in ((direction, speed) for direction in DIRECTIONS for speed in SPEEDS):
        by_scene = defaultdict(list)
        for row in rows:
            if label in row["labels"]:
                by_scene[row["scene"]].append(row)
        scenes = sorted(by_scene)
        rng.shuffle(scenes)
        for scene in scenes:
            rng.shuffle(by_scene[scene])
        ordered = []
        while scenes:
            next_scenes = []
            for scene in scenes:
                ordered.append(by_scene[scene].pop())
                if by_scene[scene]:
                    next_scenes.append(scene)
            scenes = next_scenes
        buckets[label] = deque(ordered)

    selected = []
    selected_indices = set()
    counts = Counter(initial_counts)
    scene_counts = Counter()
    while len(selected) < count:
        choice = None
        for scene_limit in (1, 2, 4, 8, 16, float("inf")):
            for label in sorted(buckets, key=lambda item: (counts[item], item)):
                bucket = buckets[label]
                for _ in range(len(bucket)):
                    row = bucket.popleft()
                    if row["index"] in selected_indices:
                        continue
                    if scene_counts[row["scene"]] >= scene_limit:
                        bucket.append(row)
                        continue
                    choice = row
                    break
                if choice is not None:
                    break
            if choice is not None:
                break
        if choice is None:
            raise ValueError(f"Only {len(selected)} eligible multicam rows, need {count}")
        selected.append(choice)
        selected_indices.add(choice["index"])
        scene_counts[choice["scene"]] += 1
        counts.update(choice["labels"])
    return selected


def label_report(rows: list[dict]) -> dict:
    counts = Counter(label for row in rows for label in row["labels"])
    return {
        "direction_occurrences": {
            direction: sum(counts[direction, speed] for speed in SPEEDS)
            for direction in DIRECTIONS
        },
        "direction_speed_occurrences": {
            f"{direction}/{speed}": counts[direction, speed]
            for direction in DIRECTIONS for speed in SPEEDS
        },
        "unlabeled_rows": sum(not row["labels"] for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditioned-dir", type=Path, required=True)
    parser.add_argument("--source-mapping", type=Path, required=True)
    parser.add_argument("--train-indices", type=Path, required=True)
    parser.add_argument("--train-ids", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--multicam-count", type=int, default=5000)
    parser.add_argument("--global-batch", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20261008)
    args = parser.parse_args()
    if args.multicam_count <= 0 or args.global_batch <= 0:
        raise ValueError("multicam-count and global-batch must be positive")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite frozen selection: {args.output_dir}")

    rng = random.Random(args.seed)
    mapping = {}
    for row in read_jsonl(args.source_mapping):
        parquet_id = f"{row['source_name']}/{row['sample_id']}"
        if parquet_id in mapping or row.get("gpu_qc_pass") is not True:
            raise ValueError(f"Duplicate or non-QC-passing source: {parquet_id}")
        mapping[parquet_id] = row
    scenes = {row["global_id"]: row["scene_key"] for row in read_jsonl(args.train_ids)}
    approved = set(json.loads(args.train_indices.read_text(encoding="utf-8")))
    multicam = []
    nonmulticam = []
    row_index = 0
    for shard in sorted(args.conditioned_dir.rglob("*.parquet")):
        for parquet_id in pq.read_table(shard, columns=["id"]).column("id").to_pylist():
            if row_index in approved:
                source = mapping[parquet_id]
                global_id = source["global_id"]
                row = {
                    "index": row_index,
                    "global_id": global_id,
                    "source": source["source_name"],
                    "scene": scenes[global_id],
                    "labels": strong_labels(source["actions"]),
                }
                (multicam if row["source"] == MULTICAM else nonmulticam).append(row)
            row_index += 1
    if row_index != len(mapping) or len(approved) != len(multicam) + len(nonmulticam):
        raise ValueError("Conditioned cache or approved index count differs from the verified source")
    if len(multicam) < args.multicam_count:
        raise ValueError(f"Only {len(multicam)} multicam rows available")

    multicam_candidates = [row for row in multicam if row["labels"]]
    initial_counts = Counter(label for row in nonmulticam for label in row["labels"])
    chosen_multi = select_multicam(multicam_candidates, initial_counts, args.multicam_count, rng)
    selected = nonmulticam + chosen_multi
    if len({row["index"] for row in selected}) != len(selected):
        raise ValueError("Selected row indices are not unique")
    rng.shuffle(selected)
    draws = [row["index"] for row in selected]
    pad = (-len(draws)) % args.global_batch
    draws.extend(row["index"] for row in nonmulticam[:pad])
    if len(draws) % args.global_batch:
        raise AssertionError("Global batch padding failed")

    source_counts = Counter(row["source"] for row in selected)
    scene_counts = Counter(row["scene"] for row in selected)
    report = {
        "quality_gate": "automatic GPU-QC pass; not human approval",
        "source_cache_rows": row_index,
        "source_approved_train_rows": len(approved),
        "available_multicam": len(multicam),
        "available_nonmulticam": len(nonmulticam),
        "selected_multicam": len(chosen_multi),
        "selected_nonmulticam": len(nonmulticam),
        "selected_unique": len(selected),
        "padded_draws": len(draws),
        "padding_repeats": pad,
        "global_batch": args.global_batch,
        "steps_per_pass": len(draws) // args.global_batch,
        "multicam_draw_fraction": len(chosen_multi) / len(draws),
        "source_counts": dict(sorted(source_counts.items())),
        "unique_scenes": len(scene_counts),
        "max_windows_per_scene": max(scene_counts.values()),
        "all_selected_labels": label_report(selected),
        "nonmulticam_labels": label_report(nonmulticam),
        "selected_multicam_labels": label_report(chosen_multi),
        "seed": args.seed,
        "selection_note": "Labels are multi-label window pose metadata, not pure per-frame actions.",
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "sampling_indices.json").write_text(json.dumps(draws) + "\n", encoding="utf-8")
    with (args.output_dir / "train_ids.jsonl").open("w", encoding="utf-8") as handle:
        for row in selected:
            output = {key: value for key, value in row.items() if key != "labels"}
            output["labels"] = sorted(row["labels"])
            handle.write(json.dumps(output) + "\n")
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
