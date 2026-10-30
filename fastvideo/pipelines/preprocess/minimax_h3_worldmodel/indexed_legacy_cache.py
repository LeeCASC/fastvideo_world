# SPDX-License-Identifier: Apache-2.0
"""Convert an audited indexed H3 cache into audio-free PRoPE parquet rows.

The input JSONL is a frozen, one-row-per-window manifest. Each row must give
the exact H3 cache, target-aligned 124-frame camera, original intrinsics, and
calibration factors. This deliberately does not discover or silently include
unreviewed source videos.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from vwm_datasets.transforms import transform_intrinsic_for_resize_center_crop

from fastvideo.dataset.dataloader.record_schema import (
    minimax_h3_worldmodel_prope_record_from_additive,
    minimax_h3_worldmodel_record_creator,
)
from fastvideo.dataset.dataloader.schema import pyarrow_schema_minimax_h3_worldmodel_prope
from fastvideo.dataset.vwm_h3 import build_local_camera_actions, h3_latent_source_frame_indices
from fastvideo.pipelines.preprocess.minimax_h3_worldmodel.legacy_cache import (
    LEGACY_H3_FPS,
    LEGACY_H3_HEIGHT,
    LEGACY_H3_NUM_FRAMES,
    LEGACY_H3_WIDTH,
    extract_legacy_h3_cache_payload,
)


def calibrated_camera(camera: np.ndarray, row: dict[str, Any]) -> np.ndarray:
    """Rebase, canonicalize OpenCV poses, and retain calibrated motion scale."""
    camera = np.asarray(camera, dtype=np.float64)
    if camera.shape != (LEGACY_H3_NUM_FRAMES, 4, 4) or not np.isfinite(camera).all():
        raise ValueError("camera must have 124 finite homogeneous 4x4 poses")
    relative = np.linalg.inv(camera[0]) @ camera
    convention = row["camera_convention"]
    if convention == "opencv_y_down":
        basis = np.diag([1.0, -1.0, 1.0, 1.0])
        relative = basis @ relative @ basis
    elif convention != "canonical_y_up":
        raise ValueError(f"unknown camera convention: {convention}")
    rotation = relative[:, :3, :3]
    if np.max(np.abs(rotation @ rotation.transpose(0, 2, 1) - np.eye(3))) > 0.05:
        raise ValueError("camera rotations are not rigid")
    translation = relative[:, :3, 3]
    maximum = float(np.linalg.norm(translation, axis=-1).max())
    if maximum > 1.0e-8:
        translation = translation / maximum
    factors = np.asarray(row["translation_scale_xyz"], dtype=np.float64)
    if factors.shape != (3, ) or not np.isfinite(factors).all() or np.any(factors <= 0):
        raise ValueError("translation_scale_xyz must have three positive finite factors")
    if np.any(factors > 4.0):
        raise ValueError("translation_scale_xyz must not exceed the audited factor cap 4")
    relative[:, :3, 3] = translation * factors
    result = relative.astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("calibrated camera is non-finite")
    return result


def aligned_intrinsics(row: dict[str, Any]) -> np.ndarray:
    """Map source/target VGGT or VWM K to the 124-frame H3 crop."""
    with np.load(row["intrinsics_path"], allow_pickle=False) as payload:
        source = np.asarray(payload[row["intrinsics_key"]], dtype=np.float32)
    if source.shape == (3, 3):
        source = source[None]
    if source.ndim != 3 or source.shape[1:] != (3, 3):
        raise ValueError("intrinsics must be [T,3,3] or [3,3]")
    if len(source) == 1:
        selected = np.repeat(source, LEGACY_H3_NUM_FRAMES, axis=0)
    elif row["intrinsics_alignment"] == "target":
        if len(source) != LEGACY_H3_NUM_FRAMES:
            raise ValueError("target-aligned intrinsics must have 124 frames")
        selected = source
    elif row["intrinsics_alignment"] == "source":
        indices = np.asarray(row["source_frame_indices"], dtype=np.int64)
        if indices.shape != (LEGACY_H3_NUM_FRAMES, ) or indices.min() < 0 or indices.max() >= len(source):
            raise ValueError("source intrinsics and source_frame_indices do not align")
        selected = source[indices]
    else:
        raise ValueError("intrinsics_alignment must be source or target")
    src_h, src_w = (int(value) for value in row["intrinsics_source_hw"])
    if min(src_h, src_w) <= 0:
        raise ValueError("intrinsics_source_hw must be positive")
    if row["intrinsics_units"] == "pixel":
        normalized = selected.copy()
        normalized[:, 0, :] /= src_w
        normalized[:, 1, :] /= src_h
    elif row["intrinsics_units"] == "normalized":
        normalized = selected
    else:
        raise ValueError("intrinsics_units must be pixel or normalized")
    output = np.stack([
        transform_intrinsic_for_resize_center_crop(
            intrinsic,
            src_size=(src_h, src_w),
            dst_size=(LEGACY_H3_HEIGHT, LEGACY_H3_WIDTH),
        ) for intrinsic in normalized
    ])
    if not np.isfinite(output).all() or np.any(output[:, (0, 1), (0, 1)] <= 0):
        raise ValueError("invalid transformed intrinsics")
    return output


def convert_indexed_cache_row(row: dict[str, Any]) -> dict[str, Any]:
    if not row.get("quality_approved", False):
        raise ValueError("only quality-approved rows may enter FastVideoWM training")
    source_indices = np.asarray(row["source_frame_indices"], dtype=np.int64)
    if source_indices.shape != (LEGACY_H3_NUM_FRAMES, ) or np.any(np.diff(source_indices) < 0):
        raise ValueError("source_frame_indices must be a monotonic 124-frame plan")
    with np.load(row["camera_path"], allow_pickle=False) as payload:
        target_camera = np.asarray(payload[row.get("camera_key", "camera_to_world")], dtype=np.float32)
    camera = calibrated_camera(target_camera, row)
    intrinsics = aligned_intrinsics(row)
    latent_indices = h3_latent_source_frame_indices(LEGACY_H3_NUM_FRAMES)
    latent_camera = camera[latent_indices]
    latent_intrinsics = intrinsics[latent_indices]
    actions = build_local_camera_actions(latent_camera)
    actions[:, 3:] /= 180.0
    video, first, text, tags = extract_legacy_h3_cache_payload(Path(row["cache_path"]))
    additive = minimax_h3_worldmodel_record_creator(
        sample_id=str(row["global_id"]),
        source_name=str(row["source_name"]),
        caption=str(row["prompt"]),
        vae_latent=video.numpy(),
        first_frame_latent=first.numpy(),
        text_embedding=text.numpy(),
        text_token_tags=tags.numpy(),
        action_continuous=actions,
        action_valid_mask=np.ones(len(latent_indices), dtype=np.float32),
        source_frame_indices=source_indices,
        width=LEGACY_H3_WIDTH,
        height=LEGACY_H3_HEIGHT,
        fps=LEGACY_H3_FPS,
        source_fps=float(row["source_fps"]),
    )
    record = minimax_h3_worldmodel_prope_record_from_additive(
        additive,
        camera_to_world=latent_camera,
        camera_intrinsics=latent_intrinsics,
    )
    if set(record) != set(pyarrow_schema_minimax_h3_worldmodel_prope.names):
        raise RuntimeError("FastVideoWM PRoPE record schema changed")
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--shard-size", type=int, default=8)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.shard_size < 1:
        raise ValueError("shard-size must be positive")
    rows = [json.loads(line) for line in args.manifest.open() if line.strip()]
    if args.limit is not None:
        rows = rows[:args.limit]
    ids = [row["global_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate global IDs in frozen manifest")
    if not all(row.get("quality_approved") is True for row in rows):
        raise ValueError("frozen manifest contains rows without final quality approval")
    for row in rows:
        for field in ("cache_path", "camera_path", "intrinsics_path"):
            if not Path(row[field]).is_file():
                raise FileNotFoundError(f"missing {field} for {row['global_id']}: {row[field]}")
    args.output.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(rows), args.shard_size):
        group = rows[start:start + args.shard_size]
        path = args.output / f"data_{start // args.shard_size:06d}.parquet"
        if path.is_file():
            if not pq.read_schema(path).equals(pyarrow_schema_minimax_h3_worldmodel_prope, check_metadata=False):
                raise ValueError(f"existing shard schema changed: {path}")
            if pq.read_table(path, columns=["id"]).column("id").to_pylist() != [r["global_id"] for r in group]:
                raise ValueError(f"existing shard IDs changed: {path}")
            continue
        records = [convert_indexed_cache_row(row) for row in group]
        temporary = path.with_suffix(f".parquet.tmp.{os.getpid()}")
        try:
            pq.write_table(pa.Table.from_pylist(records, schema=pyarrow_schema_minimax_h3_worldmodel_prope), temporary)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        print(f"converted {start + len(group)}/{len(rows)} windows: {path}", flush=True)


if __name__ == "__main__":
    main()
