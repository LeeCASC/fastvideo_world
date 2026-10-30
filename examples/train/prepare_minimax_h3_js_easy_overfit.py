#!/usr/bin/env python3
"""Freeze a pose-verified, paired JS-easy diagnostic sampling plan.

This is a diagnostic overfit plan, not a quality-approved production split.
It selects *measured-pose* records already admitted to the FastVideoWM audit;
the analytic 45-degree benchmark trajectories are deliberately never read.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import re
from typing import Any

PAIR_ORDER = (("J", "L"), ("I", "K"), ("A", "D"), ("W", "S"))
SOURCE_PATTERN = re.compile(
    r"^h3_js_easy:h3_i2v_js_easy_124_js_(\d{5})_([WASDIJKL])_(slow|fast)_(camera_only|scene_camera)$"
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_frozen(path: Path, content: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise ValueError(f"refusing to change an existing frozen JS-easy plan: {path}")
        return
    path.write_text(content, encoding="utf-8")


def eligible_groups(
    rows: list[dict[str, Any]],
    train_ids: set[str],
    *,
    min_axis_fraction: float,
    min_direction_consistency: float,
) -> dict[tuple[str, str, str], dict[str, dict[str, Any]]]:
    """Keep only nominal directions corroborated by the measured camera path."""
    groups: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        identity = str(row["global_id"])
        match = SOURCE_PATTERN.fullmatch(identity)
        if match is None or identity not in train_ids or row.get("gpu_qc_pass") is not True:
            continue
        scene, direction, speed, variant = match.groups()
        action = row.get("actions", {}).get(direction)
        if not isinstance(action, dict):
            continue
        if (float(action.get("axis_fraction", 0.0)) < min_axis_fraction
                or float(action.get("direction_consistency", 0.0)) < min_direction_consistency):
            continue
        key = (scene, speed, variant)
        if direction in groups[key]:
            raise ValueError(f"duplicate direction {direction} in JS-easy group {key}")
        groups[key][direction] = row
    return dict(groups)


def select_pairs(
    groups: dict[tuple[str, str, str], dict[str, dict[str, Any]]],
    directions: tuple[str, str],
    *,
    limit: int,
    seed: int,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Cover distinct scenes before taking a second speed/variant from one scene."""
    if limit < 1:
        raise ValueError("pair limit must be positive")
    by_scene: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for group, records in groups.items():
        if all(direction in records for direction in directions):
            by_scene[group[0]].append(group)
    rng = random.Random(seed)
    scenes = sorted(by_scene)
    rng.shuffle(scenes)
    for scene in scenes:
        rng.shuffle(by_scene[scene])
    selected: list[tuple[dict[str, Any], dict[str, Any]]] = []
    while scenes and len(selected) < limit:
        remaining = []
        for scene in scenes:
            group = by_scene[scene].pop()
            records = groups[group]
            selected.append((records[directions[0]], records[directions[1]]))
            if by_scene[scene]:
                remaining.append(scene)
            if len(selected) == limit:
                break
        scenes = remaining
    return selected


def parquet_indices(root: Path) -> dict[str, int]:
    """Use the same sorted-file/row order as the FastVideoWM map-style loader."""
    import pyarrow.parquet as pq

    files = sorted(root.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no completed conditioned parquet rows under {root}")
    result: dict[str, int] = {}
    index = 0
    for path in files:
        for identity in pq.read_table(path, columns=["id"])["id"].to_pylist():
            if identity in result:
                raise ValueError(f"duplicate conditioned parquet ID: {identity}")
            result[identity] = index
            index += 1
    return result


def freeze_plan(
    groups: dict[tuple[str, str, str], dict[str, dict[str, Any]]],
    *,
    pair_limit: int,
    seed: int,
    parquet_ids: dict[str, int] | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[int]], dict[str, Any]]:
    phases: dict[str, list[dict[str, Any]]] = {}
    indices: dict[str, list[int]] = {}
    report: dict[str, Any] = {"eligible_rows": sum(map(len, groups.values())), "available_pairs": {}}
    selected: list[dict[str, Any]] = []
    for pair_number, directions in enumerate(PAIR_ORDER):
        available = sum(all(direction in records for direction in directions) for records in groups.values())
        report["available_pairs"]["".join(directions)] = available
        pairs = select_pairs(groups, directions, limit=pair_limit, seed=seed + pair_number)
        if len(pairs) < pair_limit:
            raise ValueError(f"only {len(pairs)} eligible {''.join(directions)} pairs; need {pair_limit}")
        selected.extend(row for pair in pairs for row in pair)
        phase = ("stage1_jl", "stage2_jlik", "stage3_all8", "stage3_all8")[pair_number]
        if pair_number == 2:
            # A/D alone is not an eight-direction stage; finish with W/S below.
            continue
        if pair_number == 3 or pair_number < 2:
            frozen = [
                {
                    "global_id": row["global_id"],
                    "parquet_id": f"{row['source_name']}/{row['sample_id']}",
                    "nominal_direction": SOURCE_PATTERN.fullmatch(row["global_id"]).group(2),
                    "quality_approved": bool(row.get("quality_approved", False)),
                }
                for row in selected
            ]
            if len({row["global_id"] for row in frozen}) != len(frozen):
                raise ValueError(f"duplicate video within {phase}")
            phases[phase] = frozen
            if parquet_ids is not None:
                missing = [row["parquet_id"] for row in frozen if row["parquet_id"] not in parquet_ids]
                if missing:
                    raise ValueError(f"{phase}: {len(missing)} conditioned rows missing; first={missing[:3]}")
                indices[phase] = [parquet_ids[row["parquet_id"]] for row in frozen]
    report["phase_rows"] = {name: len(rows) for name, rows in phases.items()}
    report["note"] = "GPU-QC pass is not manual quality approval; diagnostic overfit only."
    return phases, indices, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-mapping", required=True, type=Path)
    parser.add_argument("--train-ids", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--parquet-data", type=Path, help="completed conditioned train parquet; omit to plan now")
    parser.add_argument("--pair-limit", type=int, default=10)
    parser.add_argument("--min-axis-fraction", type=float, default=0.8)
    parser.add_argument("--min-direction-consistency", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=20260926)
    args = parser.parse_args()
    if not 0.0 <= args.min_axis_fraction <= 1.0 or not 0.0 <= args.min_direction_consistency <= 1.0:
        raise ValueError("direction thresholds must be in [0,1]")
    train_ids = {str(row["global_id"]) for row in read_jsonl(args.train_ids)}
    groups = eligible_groups(
        read_jsonl(args.source_mapping),
        train_ids,
        min_axis_fraction=args.min_axis_fraction,
        min_direction_consistency=args.min_direction_consistency,
    )
    ids = parquet_indices(args.parquet_data) if args.parquet_data else None
    phases, indices, report = freeze_plan(groups, pair_limit=args.pair_limit, seed=args.seed, parquet_ids=ids)
    report.update({
        "source_mapping": str(args.source_mapping.resolve()),
        "train_ids": str(args.train_ids.resolve()),
        "pair_limit_per_axis": args.pair_limit,
        "min_axis_fraction": args.min_axis_fraction,
        "min_direction_consistency": args.min_direction_consistency,
        "seed": args.seed,
    })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for phase, rows in phases.items():
        write_frozen(args.output_dir / f"{phase}_ids.jsonl",
                     "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    for phase, ordered_indices in indices.items():
        write_frozen(args.output_dir / f"{phase}_indices.json", json.dumps(ordered_indices, indent=2) + "\n")
    write_frozen(args.output_dir / "report.json", json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
