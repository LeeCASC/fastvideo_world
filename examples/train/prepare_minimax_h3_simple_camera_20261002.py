#!/usr/bin/env python3
"""Freeze a small, direction-balanced camera curriculum from verified H3 rows.

This is a pose-metadata screen, not proof that each video's visible background
motion agrees with its pose. Source records and existing latents are read-only.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import re

import pyarrow.parquet as pq


DIRECTIONS = "WASDIJKL"
SPEEDS = ("medium", "fast")


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def intended_js_direction(global_id: str) -> str | None:
    tokens = re.findall(r"_([WASDIJKL])(?:_|$)", global_id)
    return tokens[-1] if tokens and len(set(tokens)) == 1 else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditioned-dir", type=Path, required=True)
    parser.add_argument("--source-mapping", type=Path, required=True)
    parser.add_argument("--train-indices", type=Path, required=True)
    parser.add_argument("--train-ids", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-bucket", type=int, default=32)
    parser.add_argument("--max-multicam-fraction", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=20261002)
    args = parser.parse_args()
    if args.per_bucket <= 0 or not 0 <= args.max_multicam_fraction < 1:
        raise ValueError("per-bucket target and multicam fraction are invalid")

    rng = random.Random(args.seed)
    mapping = {}
    for row in read_jsonl(args.source_mapping):
        source_id = f"{row['source_name']}/{row['sample_id']}"
        if source_id in mapping or row.get("gpu_qc_pass") is not True:
            raise ValueError(f"duplicate or non-QC-passing source: {source_id}")
        mapping[source_id] = row
    scene_by_id = {row["global_id"]: row["scene_key"] for row in read_jsonl(args.train_ids)}
    allowed = set(json.loads(args.train_indices.read_text(encoding="utf-8")))

    buckets = defaultdict(list)
    row_index = 0
    rejected_js_intent = 0
    for shard in sorted(args.conditioned_dir.rglob("*.parquet")):
        ids = pq.read_table(shard, columns=["id"]).column("id").to_pylist()
        for source_id in ids:
            if row_index in allowed:
                source = mapping[source_id]
                strong = [(key, item) for key, item in source["actions"].items()
                          if key in DIRECTIONS and item.get("speed_bin") in SPEEDS
                          and float(item.get("axis_fraction", 0)) >= 0.8
                          and float(item.get("direction_consistency", 0)) >= 0.9]
                weak = [key for key, item in source["actions"].items()
                        if key in DIRECTIONS and float(item.get("axis_fraction", 0)) >= 0.35
                        and float(item.get("direction_consistency", 0)) >= 0.65]
                if len(strong) == len(weak) == 1 and strong[0][0] == weak[0]:
                    key, action = strong[0]
                    intended = (intended_js_direction(source["global_id"])
                                if source["source_name"].startswith("h3_js_easy") else None)
                    if intended is not None and intended != key:
                        rejected_js_intent += 1
                    else:
                        scene = scene_by_id.get(source["global_id"])
                        if scene is None:
                            raise ValueError(f"missing training scene for {source['global_id']}")
                        buckets[key, action["speed_bin"]].append({
                            "index": row_index,
                            "global_id": source["global_id"],
                            "source": source["source_name"],
                            "scene": scene,
                            "direction": key,
                            "speed": action["speed_bin"],
                            "axis_fraction": float(action["axis_fraction"]),
                            "direction_consistency": float(action["direction_consistency"]),
                            "multicam": source["source_name"] == "train_v2_multicam_video",
                        })
            row_index += 1
    if row_index != len(mapping):
        raise ValueError(f"conditioned rows {row_index} do not match source mapping {len(mapping)}")

    selected = []
    scene_counts = Counter()
    candidate_counts = {}
    for key in DIRECTIONS:
        for speed in SPEEDS:
            label = key, speed
            available = buckets[label]
            candidate_counts[f"{key}/{speed}"] = len(available)
            if len(available) < args.per_bucket:
                raise ValueError(f"only {len(available)} candidates for {label}, need {args.per_bucket}")
            nonmulti = [row for row in available if not row["multicam"]]
            multicam = [row for row in available if row["multicam"]]
            rng.shuffle(nonmulti)
            rng.shuffle(multicam)
            chosen = []
            for pool in (nonmulti, multicam):
                for scene_limit in (1, 2, 4, 8, 1000000):
                    for row in pool:
                        if len(chosen) == args.per_bucket:
                            break
                        if row in chosen or scene_counts[row["scene"]] >= scene_limit:
                            continue
                        chosen.append(row)
                        scene_counts[row["scene"]] += 1
                    if len(chosen) == args.per_bucket:
                        break
            if len(chosen) != args.per_bucket:
                raise ValueError(f"selection failed for {label}: {len(chosen)}/{args.per_bucket}")
            buckets[label] = chosen

    # Repeatable round-robin order: every 16 consecutive draws cover every
    # direction and both retained speed classes once.
    for offset in range(args.per_bucket):
        for key in DIRECTIONS:
            for speed in SPEEDS:
                selected.append(buckets[key, speed][offset])
    multicam_count = sum(row["multicam"] for row in selected)
    multicam_fraction = multicam_count / len(selected)
    if multicam_fraction > args.max_multicam_fraction:
        raise ValueError(f"multicam share {multicam_fraction:.3f} exceeds {args.max_multicam_fraction:.3f}")
    if len({row["index"] for row in selected}) != len(selected):
        raise ValueError("selected conditioned row indices are not unique")

    report = {
        "quality_gate": "strict pose metadata and automatic GPU-QC pass; visual pose agreement not yet verified",
        "source_cache_rows": row_index,
        "selected_rows": len(selected),
        "directions": {key: sum(row["direction"] == key for row in selected) for key in DIRECTIONS},
        "direction_speed": {f"{key}/{speed}": len(buckets[key, speed])
                            for key in DIRECTIONS for speed in SPEEDS},
        "source_counts": dict(sorted(Counter(row["source"] for row in selected).items())),
        "multicam_fraction": multicam_fraction,
        "unique_scenes": len(scene_counts),
        "maximum_windows_per_scene": max(scene_counts.values()),
        "rejected_js_filename_pose_disagreement": rejected_js_intent,
        "candidate_counts": candidate_counts,
        "seed": args.seed,
        "note": "One strong metadata direction is not proof that every frame is a pure action.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "train_indices.json").write_text(json.dumps([row["index"] for row in selected]) + "\n")
    with (args.output_dir / "train_ids.jsonl").open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row) + "\n")
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
