#!/usr/bin/env python3
"""Freeze strict GPU-QC IDs and scene-held-out action-only training plans.

This is metadata-only. It never edits a source record or starts preprocessing.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re


ACTION_KEYS = "WASDIJKL"
DEFAULT_CANDIDATES = Path(
    "/datacc05/lizihao/worldmodel/fastvideowm_data_audit_20260924/"
    "combined_cpu_screen_complete/cpu_candidate_ids.jsonl"
)
DEFAULT_PASS = Path(
    "/datacc05/lizihao/worldmodel/fastvideowm_full_gpu_qc_20260925/"
    "gpu_qc_pass_ids.jsonl"
)


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def scene_key(row: dict) -> str:
    family = row["source_family"]
    hint = str(row.get("group_hint") or row["global_id"])
    if family == "js_refine_data" and row["subset"] == "realworld":
        hint = Path(row["video"]).parent.name.split("_", 1)[0]
    elif family == "h3_js_easy":
        match = re.search(r"_js_\d+", hint)
        if match:
            hint = hint[:match.end()]
    elif family == "train_v2" and not row.get("group_hint"):
        hint = re.sub(r"_\d{5}_\d{5}$", "", hint)
    return f"{family}:{hint}"


def stable_value(key: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--gpu-pass", type=Path, default=DEFAULT_PASS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--pilot-per-direction", type=int, default=256)
    parser.add_argument("--pilot-validation-per-direction", type=int, default=8)
    args = parser.parse_args()
    if not 0.0 < args.validation_fraction < 0.5:
        raise ValueError("validation fraction must be in (0, 0.5)")

    pass_ids = {str(row["global_id"]) for row in read_jsonl(args.gpu_pass)}
    if not pass_ids:
        raise ValueError("strict GPU pass file is empty")
    selected = []
    for row in read_jsonl(args.candidates):
        if row["global_id"] not in pass_ids:
            continue
        key = scene_key(row)
        split = ("validation" if stable_value(key, args.seed) < args.validation_fraction else "train")
        selected.append({
            "global_id": row["global_id"],
            "source_family": row["source_family"],
            "subset": row["subset"],
            "scene_key": key,
            "split": split,
            "video": row["video"],
            "pose_path": row["pose_path"],
            "cache_path": row.get("cache_path"),
            "actions": row["actions"],
            "translation_scale_xyz": row["translation_scale_xyz"],
            "gpu_qc_pass": True,
            "quality_approved": False,
        })
    if len(selected) != len(pass_ids):
        raise ValueError(f"GPU pass/candidate ID mismatch: {len(selected)} != {len(pass_ids)}")
    if len({row["global_id"] for row in selected}) != len(selected):
        raise ValueError("duplicate candidate IDs")
    train_scenes = {row["scene_key"] for row in selected if row["split"] == "train"}
    validation_scenes = {row["scene_key"] for row in selected if row["split"] == "validation"}
    if train_scenes & validation_scenes:
        raise RuntimeError("scene leakage across train/validation")

    pilot: dict[str, list[dict]] = {"train": [], "validation": []}
    pilot_counts = {"train": args.pilot_per_direction, "validation": args.pilot_validation_per_direction}
    for split in ("train", "validation"):
        for action in ACTION_KEYS:
            candidates = [
                row for row in selected
                if row["split"] == split and not row["cache_path"]
                and set(row["actions"]) == {action}
            ]
            candidates.sort(key=lambda row: stable_value(row["global_id"], args.seed + 1))
            if len(candidates) < pilot_counts[split]:
                raise ValueError(
                    f"insufficient {split} uncached single-{action} rows: "
                    f"{len(candidates)} < {pilot_counts[split]}"
                )
            pilot[split].extend(candidates[:pilot_counts[split]])
        pilot[split].sort(key=lambda row: stable_value(row["global_id"], args.seed + 2))

    args.output.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation"):
        write_jsonl(args.output / f"{split}_ids.jsonl", [row for row in selected if row["split"] == split])
        write_jsonl(args.output / f"pilot_{split}_ids.jsonl", pilot[split])
    counts = Counter((row["split"], row["source_family"]) for row in selected)
    directions = Counter((row["split"], key) for row in selected for key in row["actions"])
    speed_bins = Counter(
        (row["split"], key, value["speed_bin"])
        for row in selected for key, value in row["actions"].items()
    )
    report = {
        "status": "gpu_qc_pass_scene_split_frozen_human_review_pending",
        "total": len(selected),
        "train": sum(row["split"] == "train" for row in selected),
        "validation": sum(row["split"] == "validation" for row in selected),
        "cached": sum(bool(row["cache_path"]) for row in selected),
        "uncached": sum(not row["cache_path"] for row in selected),
        "pilot_train": len(pilot["train"]),
        "pilot_validation": len(pilot["validation"]),
        "counts_by_split_source": {f"{s}/{f}": n for (s, f), n in sorted(counts.items())},
        "action_occurrences": {f"{s}/{a}": n for (s, a), n in sorted(directions.items())},
        "speed_bin_occurrences": {f"{s}/{a}/{b}": n for (s, a, b), n in sorted(speed_bins.items())},
        "scene_groups": {"train": len(train_scenes), "validation": len(validation_scenes)},
        "pilot_policy": "uncached, single action, balanced eight directions, scene held out",
    }
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
