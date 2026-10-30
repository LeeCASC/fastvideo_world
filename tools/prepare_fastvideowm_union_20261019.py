#!/usr/bin/env python3
"""Audit and assemble the old + new MiniMax-H3 Parquet cache without copying data."""
from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


OLD = Path("/datacc05/lizihao/worldmodel/fastvideowm_full_65389_20260927/preprocessed/all")
NEW1 = Path("/datacc05/lizihao/worldmodel/vwm_unified_124_243_20261016/124f/vae_latents/minimax_h3_prope")
NEW2 = Path("/datacc05/lizihao/worldmodel/vwm_unified_124_243_20261016/124f/vae_latents/minimax_h3_prope_phase2_8gpu")
OUT = Path("/datacc05/lizihao/worldmodel/fastvideowm_union_115988_20261019")

EXPECTED_COLUMNS = {
    "id", "vae_latent_bytes", "vae_latent_shape", "vae_latent_dtype",
    "first_frame_latent_bytes", "first_frame_latent_shape", "first_frame_latent_dtype",
    "action_continuous_bytes", "action_continuous_shape", "action_continuous_dtype",
    "action_valid_mask_bytes", "action_valid_mask_shape", "action_valid_mask_dtype",
    "source_name", "caption", "media_type", "width", "height", "num_frames", "duration_sec",
    "fps", "source_fps", "source_frame_indices", "camera_to_world_bytes", "camera_to_world_shape",
    "camera_to_world_dtype", "camera_intrinsics_bytes", "camera_intrinsics_shape",
    "camera_intrinsics_dtype", "first_frame_png",
}


def shape(row: dict, key: str) -> tuple[int, ...]:
    value = row[key]
    if isinstance(value, (list, tuple)):
        return tuple(int(x) for x in value)
    raise ValueError(f"{key} is not a shape list")


def bytes_ok(row: dict, data_key: str, shape_key: str, dtype_key: str) -> bool:
    dtype = np.dtype(str(row[dtype_key]))
    expected = int(np.prod(shape(row, shape_key), dtype=np.int64)) * dtype.itemsize
    return len(row[data_key]) == expected


def audit_root(root: Path, label: str, ids: set[str], source_counts: Counter, files_out: list[dict]) -> int:
    files = sorted(root.rglob("*.parquet"))
    rows_total = 0
    for path in files:
        schema_names = set(pq.read_schema(path).names)
        if schema_names != EXPECTED_COLUMNS:
            raise RuntimeError(f"schema mismatch in {path}: {sorted(schema_names ^ EXPECTED_COLUMNS)}")
        table = pq.read_table(path).to_pydict()
        count = len(table["id"])
        rows_total += count
        files_out.append({"label": label, "path": str(path), "rows": count})
        for i in range(count):
            row = {key: values[i] for key, values in table.items()}
            sample_id = str(row["id"])
            if sample_id in ids:
                raise RuntimeError(f"duplicate id: {sample_id}")
            ids.add(sample_id)
            source_counts[str(row["source_name"])] += 1
            if shape(row, "vae_latent_shape") != (24, 37, 30, 52):
                raise RuntimeError(f"bad VAE shape in {path} row {i}: {row['vae_latent_shape']}")
            if shape(row, "first_frame_latent_shape") != (24, 1, 30, 52):
                raise RuntimeError(f"bad first-frame shape in {path} row {i}")
            if shape(row, "action_continuous_shape") != (37, 6):
                raise RuntimeError(f"bad action shape in {path} row {i}")
            if shape(row, "action_valid_mask_shape") != (37,):
                raise RuntimeError(f"bad action mask shape in {path} row {i}")
            if shape(row, "camera_to_world_shape") != (37, 4, 4):
                raise RuntimeError(f"bad C2W shape in {path} row {i}")
            if shape(row, "camera_intrinsics_shape") not in {(3, 3), (37, 3, 3)}:
                raise RuntimeError(f"bad K shape in {path} row {i}")
            for data_key, shape_key, dtype_key in (
                ("vae_latent_bytes", "vae_latent_shape", "vae_latent_dtype"),
                ("first_frame_latent_bytes", "first_frame_latent_shape", "first_frame_latent_dtype"),
                ("action_continuous_bytes", "action_continuous_shape", "action_continuous_dtype"),
                ("action_valid_mask_bytes", "action_valid_mask_shape", "action_valid_mask_dtype"),
                ("camera_to_world_bytes", "camera_to_world_shape", "camera_to_world_dtype"),
                ("camera_intrinsics_bytes", "camera_intrinsics_shape", "camera_intrinsics_dtype"),
            ):
                if not bytes_ok(row, data_key, shape_key, dtype_key):
                    raise RuntimeError(f"byte length mismatch in {path} row {i}: {data_key}")
            if not row["first_frame_png"]:
                raise RuntimeError(f"empty first_frame_png in {path} row {i}")
    return rows_total


def symlink_tree(source: Path, target: Path) -> int:
    files = sorted(source.rglob("*.parquet"))
    for src in files:
        rel = src.relative_to(source)
        dst = target / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink():
            raise RuntimeError(f"refusing to overwrite {dst}")
        dst.symlink_to(src)
    return len(files)


def main() -> None:
    if OUT.exists():
        raise RuntimeError(f"output already exists: {OUT}")
    OUT.mkdir(parents=True)
    ids: set[str] = set()
    sources: Counter = Counter()
    files: list[dict] = []
    counts = {
        "old": audit_root(OLD, "old", ids, sources, files),
        "new_phase1": audit_root(NEW1 / "shards", "new_phase1", ids, sources, files),
        "new_phase2": audit_root(NEW2 / "shards", "new_phase2", ids, sources, files),
    }
    if sum(counts.values()) != 115988:
        raise RuntimeError(f"unexpected union size: {counts}")
    link_counts = {
        "old": symlink_tree(OLD, OUT / "preprocessed/all/old"),
        "new_phase1": symlink_tree(NEW1 / "shards", OUT / "preprocessed/all/new_phase1"),
        "new_phase2": symlink_tree(NEW2 / "shards", OUT / "preprocessed/all/new_phase2"),
    }
    (OUT / "training_files.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in files), encoding="utf-8"
    )
    (OUT / "data_summary.json").write_text(json.dumps({
        "total_rows": sum(counts.values()),
        "counts_by_partition": counts,
        "parquet_files_by_partition": link_counts,
        "unique_ids": len(ids),
        "source_counts": dict(sources.most_common()),
        "old_cache": str(OLD),
        "new_phase1": str(NEW1),
        "new_phase2": str(NEW2),
        "symlinks_only": True,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"counts": counts, "unique_ids": len(ids), "sources": dict(sources)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
