#!/usr/bin/env python3
"""Fail closed on JS-easy curriculum/held-out scene leakage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def verify(plan: Path, registry: Path, mapping: Path, validation: Path) -> dict:
    planned = {row["global_id"] for row in read_jsonl(plan)}
    if len(planned) != 80:
        raise ValueError(f"expected 80 unique diagnostic train IDs, got {len(planned)}")
    scene_by_global = {}
    split_by_global = {}
    for split in ("train", "validation"):
        for row in read_jsonl(registry / f"{split}_ids.jsonl"):
            identity = row["global_id"]
            if identity in scene_by_global:
                raise ValueError(f"duplicate registry ID: {identity}")
            scene_by_global[identity] = row["scene_key"]
            split_by_global[identity] = split
    if any(split_by_global.get(identity) != "train" for identity in planned):
        raise ValueError("diagnostic plan contains non-training registry IDs")
    source_to_global = {}
    for row in read_jsonl(mapping):
        identity = f"{row['source_name']}/{row['sample_id']}"
        if identity in source_to_global:
            raise ValueError(f"duplicate mapped parquet ID: {identity}")
        source_to_global[identity] = row["global_id"]
    validation_ids = []
    for path in sorted(validation.rglob("*.parquet")):
        validation_ids.extend(pq.read_table(path, columns=["id"])["id"].to_pylist())
    if len(validation_ids) != 48 or len(set(validation_ids)) != 48:
        raise ValueError(f"expected 48 unique held-out rows, got {len(validation_ids)}")
    validation_global = {source_to_global.get(identity) for identity in validation_ids}
    if None in validation_global or any(split_by_global.get(identity) != "validation"
                                        for identity in validation_global):
        raise ValueError("held-out parquet includes unknown or training registry IDs")
    train_scenes = {scene_by_global[identity] for identity in planned}
    validation_scenes = {scene_by_global[identity] for identity in validation_global}
    overlap = train_scenes & validation_scenes
    if overlap:
        raise ValueError(f"train/validation scene leakage: {sorted(overlap)[:3]}")
    return {"train_rows": 80, "validation_rows": 48, "train_scenes": len(train_scenes),
            "validation_scenes": len(validation_scenes), "overlap": 0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.plan, args.registry, args.mapping, args.validation), indent=2))


if __name__ == "__main__":
    main()
