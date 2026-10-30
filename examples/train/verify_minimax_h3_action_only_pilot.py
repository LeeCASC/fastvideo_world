#!/usr/bin/env python3
"""Fail closed before starting the first action-only MiniMax-H3 training run."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from fastvideo.dataset.dataloader.schema import pyarrow_schema_minimax_h3_worldmodel_prope


def rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()

    mapping = {row["global_id"]: f"{row['source_name']}/{row['sample_id']}" for row in rows(args.mapping)}
    expected = {
        split: {mapping[row["global_id"]] for row in rows(args.plan / f"pilot_{split}_ids.jsonl")}
        for split in ("train", "validation")
    }
    if expected["train"] & expected["validation"]:
        raise ValueError("train/validation IDs overlap")
    observed = {"train": set(), "validation": set()}
    vision_tokens = Counter()
    translation_norms = []
    for split in ("train", "validation"):
        paths = sorted((args.data_root / "preprocessed" / split).rglob("*.parquet"))
        if not paths:
            raise FileNotFoundError(f"no conditioned parquet for {split}")
        for path in paths:
            if not pq.read_schema(path).equals(pyarrow_schema_minimax_h3_worldmodel_prope, check_metadata=False):
                raise ValueError(f"wrong schema or audio field: {path}")
            table = pq.read_table(path, schema=pyarrow_schema_minimax_h3_worldmodel_prope)
            for row in table.to_pylist():
                identity = row["id"]
                if identity not in expected[split] or identity in observed[split]:
                    raise ValueError(f"unexpected or duplicate {split} row: {identity}")
                observed[split].add(identity)
                video = np.frombuffer(row["vae_latent_bytes"], dtype=np.float32).reshape(row["vae_latent_shape"])
                first = np.frombuffer(row["first_frame_latent_bytes"], dtype=np.float32).reshape(
                    row["first_frame_latent_shape"]
                )
                camera = np.frombuffer(row["camera_to_world_bytes"], dtype=np.float32).reshape(
                    row["camera_to_world_shape"]
                )
                tags = np.frombuffer(row["text_token_tags_bytes"], dtype=np.float32).reshape(
                    row["text_token_tags_shape"]
                )
                if video.shape != (24, 37, 30, 52) or first.shape != (24, 1, 30, 52):
                    raise ValueError(f"wrong H3 latent shape: {identity}")
                if camera.shape != (37, 4, 4) or tags.ndim != 1 or not np.any(tags == 0):
                    raise ValueError(f"missing camera or first-frame vision tokens: {identity}")
                if not all(np.isfinite(value).all() for value in (video, first, camera)):
                    raise ValueError(f"nonfinite H3 latent or camera: {identity}")
                if not np.allclose(camera[0], np.eye(4), atol=5.0e-3):
                    raise ValueError(f"camera was not rebased to first frame: {identity}")
                vision_tokens[split] += int(np.count_nonzero(tags == 0))
                translation_norms.append(float(np.linalg.norm(camera[-1, :3, 3])))
        if observed[split] != expected[split]:
            missing = sorted(expected[split] - observed[split])
            raise ValueError(f"missing {split} rows: {len(missing)}, first={missing[:5]}")
    if max(translation_norms) - min(translation_norms) < 0.05:
        raise ValueError("camera translation scale appears constant across pilot rows")
    print(json.dumps({
        "status": "action_only_pilot_ready",
        "train": len(observed["train"]),
        "validation": len(observed["validation"]),
        "vision_tokens": dict(vision_tokens),
        "translation_norm_range": [min(translation_norms), max(translation_norms)],
        "audio_fields": 0,
    }))


if __name__ == "__main__":
    main()
