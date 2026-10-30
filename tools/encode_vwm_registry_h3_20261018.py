#!/usr/bin/env python3
"""Encode a corrected VWM registry with the native MiniMax-H3 VAE.

This consumes only rows that pass the automatic GPU-QC predicate. Source media
and source pose archives remain read-only; output is resumable Parquet shards.
"""

from __future__ import annotations

import argparse
from io import BytesIO
import json
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from decord import VideoReader, cpu

from fastvideo.dataset.dataloader.parquet_io import records_to_table
from fastvideo.dataset.dataloader.record_schema import minimax_h3_worldmodel_prope_vae_record_creator
from fastvideo.dataset.dataloader.schema import pyarrow_schema_minimax_h3_worldmodel_prope_vae
from fastvideo.dataset.vwm_h3 import (
    build_normalized_local_camera_actions,
    canonicalize_source_intrinsics,
    h3_latent_source_frame_indices,
    rebase_camera_to_first_frame,
)
from fastvideo.models.loader.component_loader import PipelineComponentLoader
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.configs.pipelines.minimax_h3 import MiniMaxH3PipelineConfig
from fastvideo.pipelines.preprocess.minimax_h3_worldmodel.encoding import (
    encode_h3_worldmodel_video,
)
from fastvideo.pipelines.preprocess.minimax_h3_worldmodel.preprocess import (
    _encode_png,
    _fastvideo_args,
    _initialize_worker,
    _load_component,
    _load_model_index,
)
from vwm_datasets.transforms import resize_center_crop
from vwm_datasets import VWMSample


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def gpu_pass(row: dict) -> bool:
    assessment = row.get("assessment") or {}
    if row.get("error") or row.get("parse_error"):
        return False
    if assessment.get("uncertain") is True or assessment.get("visible_camera_motion_match") != "yes":
        return False
    return not any(
        assessment.get(key) is True
        for key in ("blur", "texture_breakdown", "subject_deformation", "translation_jitter")
    )


def _cache_key(value: str) -> str:
    """Normalize old cache ids and the new registry ids to one dedupe key."""
    value = str(value)
    # Old records use source/sample_id while the new registry uses source:sample_id.
    if ":" in value:
        source, sample = value.split(":", 1)
        return f"{source}/{sample}"
    return value


def existing_cache_keys(cache_roots: list[Path]) -> set[str]:
    keys: set[str] = set()
    for root in cache_roots:
        if not root.exists():
            continue
        for path in root.rglob("*.parquet"):
            try:
                table = pq.read_table(path, columns=["id"])
            except Exception:
                continue
            keys.update(_cache_key(x) for x in table["id"].to_pylist())
    return keys


def load_rows(root: Path, frames: int, skip_keys: set[str] | None = None) -> list[dict]:
    base = root / "124f"
    registry = {row["global_id"]: row for row in read_jsonl(
        base / "registry" / f"windows_{frames}_candidates.jsonl"
    )}
    qc_rows = []
    for path in sorted((base / "gpu_qc" / "results").glob("*.jsonl")):
        qc_rows.extend(read_jsonl(path))
    selected = [registry[row["global_id"]] for row in qc_rows if gpu_pass(row)]
    if skip_keys:
        selected = [
            row for row in selected
            if _cache_key(f"{row['canonical_source']}/{row['sample_id']}") not in skip_keys
        ]
    selected.sort(key=lambda row: str(row["global_id"]))
    if len(selected) != len({row["global_id"] for row in selected}):
        raise ValueError("duplicate selected GPU-QC IDs")
    return selected


def load_video_and_geometry(row: dict, *, height: int = 480, width: int = 832):
    indices = np.asarray(row["source_frame_indices"], dtype=np.int64)
    reader = VideoReader(str(row["video"]), ctx=cpu(0), num_threads=2)
    try:
        frames = reader.get_batch(indices.tolist()).asnumpy().astype(np.uint8)
    finally:
        del reader
    with np.load(row["camera"], allow_pickle=False) as payload:
        key = "camera_to_world" if "camera_to_world" in payload else "c2w"
        camera = np.asarray(payload[key], dtype=np.float32)
        intrinsics = np.asarray(payload["intrinsics"], dtype=np.float32)
    camera = camera[indices]
    if intrinsics.ndim == 3:
        intrinsics = intrinsics[indices]
    camera = rebase_camera_to_first_frame(camera)
    intrinsics = canonicalize_source_intrinsics(
        intrinsics,
        source_name=str(row["canonical_source"]),
        image_width=int(row["width"]),
        image_height=int(row["height"]),
    )
    sample = VWMSample(
        sample_id=str(row["sample_id"]),
        source_name=str(row["canonical_source"]),
        video=frames,
        camera_to_world=camera,
        intrinsics=intrinsics,
        source_frame_indices=indices,
        fps=24.0,
        text=Path(row["prompt_source"]).read_text(encoding="utf-8").strip(),
        width=int(row["width"]),
        height=int(row["height"]),
    )
    return resize_center_crop(sample, (height, width))


def atomic_write(records: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + f".tmp.{os.getpid()}")
    pq.write_table(records_to_table(records, pyarrow_schema_minimax_h3_worldmodel_prope_vae), temporary, compression="zstd")
    os.replace(temporary, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("/datacc05/lizihao_wm/worldmodel/ckpts/MiniMax-H3"))
    parser.add_argument("--worker-index", type=int, required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--shard-size", type=int, default=8)
    parser.add_argument(
        "--existing-cache", type=Path, action="append", default=[],
        help="Previous parquet cache root(s); rows with the same source/sample id are skipped.",
    )
    parser.add_argument("--max-rows", type=int, default=0, help="Optional smoke-test limit after deduplication.")
    args = parser.parse_args()
    if not 0 <= args.worker_index < args.num_workers:
        raise ValueError("worker-index must be in [0,num-workers)")
    device = _initialize_worker(args.device_index, 29800 + args.worker_index)
    model_index = _load_model_index(args.model)
    fastvideo_args = _fastvideo_args(args.model)
    vae = _load_component("vae", args.model, model_index, fastvideo_args)
    vae.eval()
    skip_keys = existing_cache_keys(args.existing_cache)
    rows = load_rows(args.root, 124, skip_keys=skip_keys)
    if args.max_rows > 0:
        rows = rows[: args.max_rows]
    print(
        f"worker={args.worker_index} selected={len(rows)} skipped_existing={len(skip_keys)}",
        flush=True,
    )
    rows = rows[args.worker_index :: args.num_workers]
    args.output.mkdir(parents=True, exist_ok=True)
    failed_path = args.output / f"failed_worker_{args.worker_index:02d}.jsonl"
    failed = failed_path.open("a", encoding="utf-8")
    try:
        for shard_no in range(0, len(rows), args.shard_size):
            chunk = rows[shard_no : shard_no + args.shard_size]
            out = args.output / "shards" / f"worker_{args.worker_index:02d}" / f"data_{shard_no // args.shard_size:06d}.parquet"
            if out.exists():
                continue
            records = []
            for row in chunk:
                try:
                    sample = load_video_and_geometry(row)
                    video = torch.from_numpy(np.ascontiguousarray(sample.video)).permute(3, 0, 1, 2).contiguous()
                    target, first = encode_h3_worldmodel_video(vae, video, device=device)
                    latent_indices = h3_latent_source_frame_indices(124)
                    camera_latent = sample.camera_to_world[latent_indices]
                    if sample.intrinsics.ndim == 2:
                        intr_latent = np.broadcast_to(sample.intrinsics, (len(latent_indices), 3, 3)).copy()
                    else:
                        intr_latent = sample.intrinsics[latent_indices]
                    actions = build_normalized_local_camera_actions(camera_latent)
                    records.append(minimax_h3_worldmodel_prope_vae_record_creator(
                        first_frame_png=_encode_png(video[:, 0]),
                        sample_id=str(row["global_id"]),
                        source_name=str(row["canonical_source"]),
                        caption=sample.text or "A video scene observed from a moving camera.",
                        vae_latent=target.numpy(),
                        first_frame_latent=first.numpy(),
                        action_continuous=actions,
                        action_valid_mask=np.ones(actions.shape[0], dtype=np.bool_),
                        source_frame_indices=np.asarray(row["source_frame_indices"], dtype=np.int64),
                        width=sample.width,
                        height=sample.height,
                        fps=24.0,
                        source_fps=float(row["source_fps"]),
                        camera_to_world=camera_latent,
                        camera_intrinsics=intr_latent,
                    ))
                except Exception as error:
                    failed.write(json.dumps({"global_id": row["global_id"], "error": f"{type(error).__name__}: {error}"}) + "\n")
                    failed.flush()
            if records:
                atomic_write(records, out)
            print(f"worker={args.worker_index} processed={shard_no + len(chunk)}/{len(rows)}", flush=True)
    finally:
        failed.close()


if __name__ == "__main__":
    main()
