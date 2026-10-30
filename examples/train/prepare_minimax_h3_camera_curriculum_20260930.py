#!/usr/bin/env python3
"""Freeze a balanced, geometry-corrected JS-easy camera-control diagnostic.

The existing H3 video/first-frame/text latents are copied unchanged. Only the
derived camera matrices are changed: the old per-axis translation multipliers
are inverted, leaving one isotropically normalized measured camera path.
Source records and the 65,389-row conditioned cache are read-only.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import re

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


PATTERN = re.compile(
    r"^h3_js_easy:h3_i2v_js_easy_124_js_(\d{5})_([WASDIJKL])_(slow|fast)_(camera_only|scene_camera)$"
)
DIRECTIONS = "WASDIJKL"


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-mapping", type=Path, required=True)
    parser.add_argument("--train-ids", type=Path, required=True)
    parser.add_argument("--conditioned-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260930)
    args = parser.parse_args()
    output = args.output
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Refusing to overwrite existing diagnostic data: {output}")

    train_ids = {row["global_id"] for row in read_jsonl(args.train_ids)}
    selected = {}
    for row in read_jsonl(args.source_mapping):
        match = PATTERN.fullmatch(row["global_id"])
        if match is None or row["global_id"] not in train_ids or row.get("gpu_qc_pass") is not True:
            continue
        direction = match.group(2)
        action = row.get("actions", {}).get(direction, {})
        if (float(action.get("axis_fraction", 0)) < 0.8
                or float(action.get("direction_consistency", 0)) < 0.8):
            continue
        selected[f"{row['source_name']}/{row['sample_id']}"] = (row, direction)
    if len(selected) < 100:
        raise SystemExit(f"Too few strictly screened examples: {len(selected)}")

    files = sorted(args.conditioned_data.rglob("*.parquet"))
    if not files:
        raise SystemExit("No conditioned H3 parquet found")
    found = {}
    for path in files:
        ids = pq.read_table(path, columns=["id"])["id"].to_pylist()
        hits = [(index, identity) for index, identity in enumerate(ids) if identity in selected]
        if not hits:
            continue
        table = pq.read_table(path)
        for index, identity in hits:
            if identity in found:
                raise SystemExit(f"Duplicate conditioned ID: {identity}")
            source, direction = selected[identity]
            sample = table.slice(index, 1).to_pylist()[0]
            shape = tuple(sample["camera_to_world_shape"])
            dtype = np.dtype(sample["camera_to_world_dtype"])
            camera = np.frombuffer(sample["camera_to_world_bytes"], dtype=dtype).reshape(shape).copy()
            if camera.shape != (37, 4, 4) or not np.isfinite(camera).all():
                raise SystemExit(f"Invalid camera path for {identity}: {camera.shape}")
            if not np.allclose(camera[0], np.eye(4), atol=1e-3):
                raise SystemExit(f"First-frame camera is not identity: {identity}")
            factors = np.asarray(source["translation_scale_xyz"], dtype=np.float32)
            if factors.shape != (3,) or not np.isfinite(factors).all() or np.any(factors <= 0):
                raise SystemExit(f"Invalid old translation scale: {identity}: {factors}")
            camera[:, :3, 3] /= factors[None, :]
            camera[0] = np.eye(4, dtype=camera.dtype)
            if not np.isfinite(camera).all():
                raise SystemExit(f"Non-finite corrected camera: {identity}")
            # The old pipeline divided measured displacement by its maximum
            # norm before multiplying the three axes independently. Therefore
            # the inverse must recover a path with norm <=~1. The 37 latent
            # frames can miss the peak among the 124 decoded source frames.
            maximum = np.linalg.norm(camera[:, :3, 3], axis=1).max()
            if maximum < 0.01 or maximum > 1.1:
                raise SystemExit(f"Unexpected inverse scale for {identity}: max_norm={maximum}")
            sample["camera_to_world_bytes"] = camera.tobytes()
            found[identity] = (sample, direction, source, float(maximum))
    missing = sorted(set(selected) - set(found))
    if missing:
        raise SystemExit(f"Missing {len(missing)} conditioned rows; first={missing[:3]}")

    output.mkdir(parents=True)
    data = output / "data"
    data.mkdir()
    ordered = sorted(found)
    schema = pq.read_schema(files[0])
    # Bounded output files keep the map-style loader and validation fast.
    for start in range(0, len(ordered), 16):
        rows = [found[key][0] for key in ordered[start:start + 16]]
        pq.write_table(pa.Table.from_pylist(rows, schema=schema), data / f"data_{start // 16:04d}.parquet")

    by_direction = {direction: [] for direction in DIRECTIONS}
    manifest = []
    for index, identity in enumerate(ordered):
        _, direction, source, maximum = found[identity]
        by_direction[direction].append(index)
        manifest.append({
            "id": identity, "global_id": source["global_id"], "direction": direction,
            "scene": PATTERN.fullmatch(source["global_id"]).group(1),
            "speed": PATTERN.fullmatch(source["global_id"]).group(3),
            "axis_fraction": source["actions"][direction]["axis_fraction"],
            "direction_consistency": source["actions"][direction]["direction_consistency"],
            "old_translation_scale_xyz": source["translation_scale_xyz"],
            "corrected_max_translation_norm": maximum,
            "parquet_index": index,
        })
    rng = random.Random(args.seed)
    target = max(map(len, by_direction.values()))
    balanced = []
    for direction in DIRECTIONS:
        entries = by_direction[direction]
        rng.shuffle(entries)
        balanced.extend(entries)
        balanced.extend(entries[i % len(entries)] for i in range(target - len(entries)))
    rng.shuffle(balanced)
    (output / "train_indices.json").write_text(json.dumps(balanced, indent=2) + "\n")
    (output / "ids.jsonl").write_text("".join(json.dumps(row) + "\n" for row in manifest))
    report = {
        "unique_samples": len(ordered),
        "unique_direction_counts": dict(Counter(row["direction"] for row in manifest)),
        "balanced_draws_per_direction": target,
        "balanced_draws_total": len(balanced),
        "geometry": "undo old per-axis XYZ translation scale; preserve existing rotations, K, and all latents",
        "source_conditioned_data": str(args.conditioned_data),
        "source_mapping": str(args.source_mapping),
        "quality_scope": "existing GPU-QC pass and measured-pose directional screen; not manual video/pose review",
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
