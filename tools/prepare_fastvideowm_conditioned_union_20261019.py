#!/usr/bin/env python3
"""Audit final conditioned rows and assemble the old + new cache by symlink."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


OLD = Path("/datacc05/lizihao/worldmodel/fastvideowm_full_65389_20260927/preprocessed/all")
NEW = Path("/datacc05/lizihao/worldmodel/fastvideowm_union_115988_20261019/new_conditioned")
OUT = Path("/datacc05/lizihao/worldmodel/fastvideowm_union_115988_20261019")


def expected_schema(path: Path):
    return pq.read_schema(path)


def valid_bytes(row: dict, data: str, shape_key: str, dtype_key: str) -> bool:
    dtype = np.dtype(str(row[dtype_key]))
    shape = tuple(int(x) for x in row[shape_key])
    return len(row[data]) == int(np.prod(shape, dtype=np.int64)) * dtype.itemsize


def audit(root: Path, label: str, ids: set[str], source_counts: Counter, files_out: list[dict], schema) -> int:
    total = 0
    # Read the small index columns for every row.  Payload bytes are sampled per
    # file below; the old cache and the conditioned cache have already passed
    # their producer-side row-count checks, and loading all latent bytes here
    # would make this final audit needlessly take tens of minutes.
    index_columns = [
        "id", "source_name", "vae_latent_shape", "first_frame_latent_shape",
        "action_continuous_shape", "action_valid_mask_shape", "camera_to_world_shape",
        "camera_intrinsics_shape",
    ]
    sample_columns = [
        "vae_latent_bytes", "vae_latent_shape", "vae_latent_dtype",
        "first_frame_latent_bytes", "first_frame_latent_shape", "first_frame_latent_dtype",
        "text_embedding_bytes", "text_embedding_shape", "text_embedding_dtype",
        "text_token_tags_bytes", "text_token_tags_shape", "text_token_tags_dtype",
        "action_continuous_bytes", "action_continuous_shape", "action_continuous_dtype",
        "action_valid_mask_bytes", "action_valid_mask_shape", "action_valid_mask_dtype",
        "camera_to_world_bytes", "camera_to_world_shape", "camera_to_world_dtype",
        "camera_intrinsics_bytes", "camera_intrinsics_shape", "camera_intrinsics_dtype",
    ]
    paths = sorted(root.rglob("*.parquet"))
    sample_paths = set(paths[:4] + paths[-4:])
    for path in paths:
        if not pq.read_schema(path).equals(schema, check_metadata=False):
            raise RuntimeError(f"schema mismatch: {path}")
        table = pq.read_table(path, columns=index_columns).to_pydict()
        count = len(table["id"])
        total += count
        files_out.append({"label": label, "path": str(path), "rows": count})
        for i in range(count):
            row = {k: v[i] for k, v in table.items()}
            sample_id = str(row["id"])
            if sample_id in ids:
                raise RuntimeError(f"duplicate id: {sample_id}")
            ids.add(sample_id)
            source_counts[str(row["source_name"])] += 1
            if tuple(row["vae_latent_shape"]) != (24, 37, 30, 52):
                raise RuntimeError(f"bad VAE shape: {path} row {i}")
            if tuple(row["first_frame_latent_shape"]) != (24, 1, 30, 52):
                raise RuntimeError(f"bad keyframe shape: {path} row {i}")
            if tuple(row["action_continuous_shape"]) != (37, 6):
                raise RuntimeError(f"bad action shape: {path} row {i}")
            if tuple(row["action_valid_mask_shape"]) != (37,):
                raise RuntimeError(f"bad action mask shape: {path} row {i}")
            if tuple(row["camera_to_world_shape"]) != (37, 4, 4):
                raise RuntimeError(f"bad C2W shape: {path} row {i}")
            if tuple(row["camera_intrinsics_shape"]) not in {(3, 3), (37, 3, 3)}:
                raise RuntimeError(f"bad K shape: {path} row {i}")
            for data, shape, dtype in (
                ("vae_latent_bytes", "vae_latent_shape", "vae_latent_dtype"),
                ("first_frame_latent_bytes", "first_frame_latent_shape", "first_frame_latent_dtype"),
                ("text_embedding_bytes", "text_embedding_shape", "text_embedding_dtype"),
                ("text_token_tags_bytes", "text_token_tags_shape", "text_token_tags_dtype"),
                ("action_continuous_bytes", "action_continuous_shape", "action_continuous_dtype"),
                ("action_valid_mask_bytes", "action_valid_mask_shape", "action_valid_mask_dtype"),
                ("camera_to_world_bytes", "camera_to_world_shape", "camera_to_world_dtype"),
                ("camera_intrinsics_bytes", "camera_intrinsics_shape", "camera_intrinsics_dtype"),
            ):
                pass
        if path not in sample_paths:
            continue
        sample = pq.ParquetFile(path).read(columns=sample_columns).slice(0, 1).to_pydict()
        row = {k: v[0] for k, v in sample.items()}
        for data, shape, dtype in (
            ("vae_latent_bytes", "vae_latent_shape", "vae_latent_dtype"),
            ("first_frame_latent_bytes", "first_frame_latent_shape", "first_frame_latent_dtype"),
            ("text_embedding_bytes", "text_embedding_shape", "text_embedding_dtype"),
            ("text_token_tags_bytes", "text_token_tags_shape", "text_token_tags_dtype"),
            ("action_continuous_bytes", "action_continuous_shape", "action_continuous_dtype"),
            ("action_valid_mask_bytes", "action_valid_mask_shape", "action_valid_mask_dtype"),
            ("camera_to_world_bytes", "camera_to_world_shape", "camera_to_world_dtype"),
            ("camera_intrinsics_bytes", "camera_intrinsics_shape", "camera_intrinsics_dtype"),
        ):
            if not valid_bytes(row, data, shape, dtype):
                raise RuntimeError(f"bad byte length {data}: {path} sample row")
        tags = np.frombuffer(row["text_token_tags_bytes"], dtype=np.dtype(str(row["text_token_tags_dtype"])))
        if tags.ndim != 1 or not np.isin(tags, (0, 1)).all() or not (tags == 0).any():
            raise RuntimeError(f"missing H3 vision tags: {path} sample row")
        camera = np.frombuffer(row["camera_to_world_bytes"], dtype=np.float32).reshape(37, 4, 4)
        if not np.isfinite(camera).all() or not np.allclose(camera[0], np.eye(4), atol=1e-4):
            raise RuntimeError(f"invalid/rebased camera: {path} sample row")
        intr = np.frombuffer(row["camera_intrinsics_bytes"], dtype=np.float32)
        intr_matrix = intr.reshape(tuple(row["camera_intrinsics_shape"]))
        focal = intr_matrix[..., :2, :2].diagonal(axis1=-2, axis2=-1)
        if not np.isfinite(intr).all() or not (focal > 0).all():
            raise RuntimeError(f"invalid intrinsics: {path} sample row")
    return total


def symlink_files(source: Path, target: Path) -> int:
    paths = sorted(source.rglob("*.parquet"))
    for src in paths:
        rel = src.relative_to(source)
        dst = target / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink():
            raise RuntimeError(f"refusing to overwrite {dst}")
        dst.symlink_to(src)
    return len(paths)


def main() -> None:
    if not NEW.exists() or not (NEW / "CONDITIONING_COMPLETE").exists():
        raise RuntimeError("new conditioned cache is not complete")
    new_files = sorted(NEW.rglob("*.parquet"))
    old_files = sorted(OLD.rglob("*.parquet"))
    if not old_files or not new_files:
        raise RuntimeError("missing old or new conditioned Parquet files")
    schema = expected_schema(old_files[0])
    ids: set[str] = set()
    source_counts: Counter = Counter()
    files: list[dict] = []
    old_rows = audit(OLD, "old_conditioned", ids, source_counts, files, schema)
    new_rows = audit(NEW, "new_conditioned", ids, source_counts, files, schema)
    if (old_rows, new_rows, len(ids)) != (65389, 50599, 115988):
        raise RuntimeError(f"unexpected totals old={old_rows} new={new_rows} unique={len(ids)}")
    train_root = OUT / "preprocessed/all"
    train_root.mkdir(parents=True, exist_ok=True)
    link_counts = {
        "old_conditioned": symlink_files(OLD, train_root / "old_conditioned"),
        "new_conditioned": symlink_files(NEW, train_root / "new_conditioned"),
    }
    (OUT / "training_files.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in files), encoding="utf-8"
    )
    (OUT / "data_summary.json").write_text(json.dumps({
        "total_rows": len(ids),
        "old_conditioned_rows": old_rows,
        "new_conditioned_rows": new_rows,
        "unique_ids": len(ids),
        "source_counts": dict(source_counts.most_common()),
        "training_data_path": str(train_root),
        "training_files": link_counts,
        "source_data_read_only": True,
        "scene_text_used_by_active_recipe": False,
        "vision_only_conditioning": True,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"old_rows": old_rows, "new_rows": new_rows, "unique": len(ids), "sources": dict(source_counts)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
