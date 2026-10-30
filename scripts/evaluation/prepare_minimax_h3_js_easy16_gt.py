# SPDX-License-Identifier: Apache-2.0
"""Encode the 16 JS-easy GT clips on H3's fixed 124-frame time grid.

This is a progress-aligned comparison to 61-frame GT, not an original-FPS
comparison. The exact benchmark PNG's keyframe and Qwen condition are reused
from prepare_minimax_h3_js_easy_benchmark_20261008.py; only GT video latent and
its matched numeric camera path are replaced here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
from vwm_datasets.transforms import transform_intrinsic_for_resize_center_crop

from fastvideo.dataset.dataloader.schema import pyarrow_schema_minimax_h3_worldmodel_prope
from fastvideo.dataset.vwm_h3 import (
    build_normalized_local_camera_actions,
    h3_latent_source_frame_indices,
    rebase_camera_to_first_frame,
)
from fastvideo.pipelines.preprocess.minimax_h3_worldmodel.encoding import encode_h3_worldmodel_video
from fastvideo.pipelines.preprocess.minimax_h3_worldmodel.preprocess import (
    _fastvideo_args,
    _initialize_worker,
    _load_component,
    _load_model_index,
)


SOURCE_FRAMES = 61
TARGET_FRAMES = 124
TARGET_HW = (480, 832)
CASE_IDS = tuple(f"js_{index:05d}" for index in range(1, 17))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_indices() -> np.ndarray:
    """Nearest GT frames on a fixed normalized-progress grid."""
    return np.rint(np.linspace(0, SOURCE_FRAMES - 1, TARGET_FRAMES)).astype(np.int64)


def resize_center_crop_rgb(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    scale = max(TARGET_HW[0] / height, TARGET_HW[1] / width)
    resized_h = max(TARGET_HW[0], round(height * scale))
    resized_w = max(TARGET_HW[1], round(width * scale))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(frame, (resized_w, resized_h), interpolation=interpolation)
    top = (resized_h - TARGET_HW[0]) // 2
    left = (resized_w - TARGET_HW[1]) // 2
    return np.ascontiguousarray(resized[top:top + TARGET_HW[0], left:left + TARGET_HW[1]])


def read_gt_video(path: Path, *, pixels: bool) -> tuple[np.ndarray | None, float, tuple[int, int]]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Could not open GT video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = []
    source_hw = None
    try:
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            source_hw = bgr.shape[:2]
            if pixels:
                frames.append(resize_center_crop_rgb(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
            else:
                frames.append(None)
    finally:
        capture.release()
    if len(frames) != SOURCE_FRAMES or source_hw is None or fps <= 0:
        raise ValueError(f"Expected {SOURCE_FRAMES} decoded GT frames with positive FPS: {path}")
    return (np.stack(frames) if pixels else None), fps, source_hw


def read_geometry(case_dir: Path, source_hw: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    metadata = json.loads((case_dir / "sample_001.json").read_text(encoding="utf-8"))
    mouse = metadata["frame_actions"]["mouse"]
    camera = np.asarray([entry["c2w"] for entry in mouse], dtype=np.float32)
    if camera.shape != (SOURCE_FRAMES, 4, 4):
        raise ValueError(f"Expected 61 corrected C2W poses: {case_dir}")
    pose_metadata = json.loads((case_dir / "sample_001_pose.json").read_text(encoding="utf-8"))
    calibration_hw = tuple(int(value) for value in pose_metadata["processed_hw"])
    with np.load(case_dir / "sample_001_pose.npz") as pose:
        pixel_k = np.asarray(pose["intrinsics"], dtype=np.float32)
    if pixel_k.shape != (SOURCE_FRAMES, 3, 3):
        raise ValueError(f"Expected 61 pixel-space intrinsics: {case_dir}")
    normalized_k = pixel_k.copy()
    normalized_k[:, 0, :] /= calibration_hw[1]
    normalized_k[:, 1, :] /= calibration_hw[0]
    indices = source_indices()
    anchors = h3_latent_source_frame_indices(TARGET_FRAMES)
    latent_camera = rebase_camera_to_first_frame(camera[indices][anchors])
    latent_k = np.stack([
        transform_intrinsic_for_resize_center_crop(value, src_size=source_hw, dst_size=TARGET_HW)
        for value in normalized_k[indices][anchors]
    ]).astype(np.float32)
    if latent_camera.shape != (37, 4, 4) or latent_k.shape != (37, 3, 3):
        raise ValueError(f"Unexpected latent geometry: {case_dir}")
    orthogonality = latent_camera[:, :3, :3] @ np.swapaxes(latent_camera[:, :3, :3], 1, 2)
    if (not np.isfinite(latent_camera).all() or not np.isfinite(latent_k).all()
            or not np.allclose(latent_camera[0], np.eye(4), atol=1e-4)
            or not np.allclose(orthogonality, np.eye(3), atol=2e-2)
            or not np.allclose(latent_camera[:, 3], [0, 0, 0, 1], atol=1e-4)
            or np.any(latent_k[:, 0, 0] <= 0) or np.any(latent_k[:, 1, 1] <= 0)
            or np.any(latent_k[:, 0, 0] > 32) or np.any(latent_k[:, 1, 1] > 32)
            or np.any(np.abs(latent_k[:, :2, 2]) > 2)):
        raise ValueError(f"Invalid camera geometry or unnormalized K: {case_dir}")
    return latent_camera, latent_k


def _set_float32_tensor(row: dict, name: str, value: np.ndarray) -> None:
    array = np.ascontiguousarray(value, dtype=np.float32)
    row[f"{name}_bytes"] = array.tobytes()
    row[f"{name}_shape"] = list(array.shape)
    row[f"{name}_dtype"] = "float32"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--first-frame-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--simple-output-root", type=Path,
                        help="Write exact-PNG synthetic shape carriers with each case's calibrated K")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--master-port", type=int, default=29983)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--simple-only", action="store_true",
                        help="CPU-only staging of calibrated synthetic-action rows")
    args = parser.parse_args()
    if args.simple_only and args.simple_output_root is None:
        parser.error("--simple-only requires --simple-output-root")

    pending = []
    for case_id in CASE_IDS:
        case_dir = args.benchmark_root / case_id
        video_path = case_dir / "sample_001_gt.mp4"
        first_path = case_dir / "sample_001_first_frame.png"
        frames, source_fps, source_hw = read_gt_video(video_path, pixels=False)
        del frames
        camera, intrinsics = read_geometry(case_dir, source_hw)
        print(f"validated {case_id}: 61 -> 124 frames, {source_fps:g} source FPS, K={intrinsics[0,0,0]:.3f}",
              flush=True)
        if args.validate_only:
            continue
        first_row_path = args.first_frame_root / case_id / "data_000000.parquet"
        source_row = pq.read_table(first_row_path).to_pylist()[0]
        if source_row["id"] != f"js_easy_benchmark/{case_id}":
            raise ValueError(f"Unexpected exact-PNG condition: {first_row_path}")
        out_dir = args.output_root / case_id
        out_path = out_dir / "data_000000.parquet"
        fingerprints = {path.name: _sha256(path) for path in (
            video_path, first_path, case_dir / "sample_001.json", case_dir / "sample_001_pose.npz"
        )}
        if args.simple_output_root is not None:
            simple_dir = args.simple_output_root / case_id
            simple_path = simple_dir / "data_000000.parquet"
            if simple_path.exists():
                prior_simple = json.loads((simple_dir / "source.json").read_text(encoding="utf-8"))
                if prior_simple.get("source_sha256") != fingerprints:
                    raise ValueError(f"Existing simple condition does not match benchmark sources: {simple_dir}")
            else:
                simple_row = dict(source_row)
                simple_row["id"] = f"js_easy_benchmark/{case_id}/synthetic_shape_carrier"
                simple_row["source_frame_indices"] = source_indices().tolist()
                simple_row["source_fps"] = source_fps
                simple_row["fps"] = 24.0
                simple_row["num_frames"] = TARGET_FRAMES
                simple_row["duration_sec"] = TARGET_FRAMES / 24.0
                _set_float32_tensor(simple_row, "camera_to_world", np.broadcast_to(np.eye(4), (37, 4, 4)))
                _set_float32_tensor(simple_row, "camera_intrinsics", intrinsics)
                _set_float32_tensor(simple_row, "action_continuous", np.zeros((37, 6), dtype=np.float32))
                _set_float32_tensor(simple_row, "action_valid_mask", np.zeros(37, dtype=np.float32))
                simple_dir.mkdir(parents=True, exist_ok=True)
                simple_temporary = simple_dir / "data_000000.parquet.tmp"
                pq.write_table(pa.Table.from_pylist([simple_row], schema=pyarrow_schema_minimax_h3_worldmodel_prope),
                               simple_temporary, compression="zstd")
                os.replace(simple_temporary, simple_path)
                (simple_dir / "source.json").write_text(json.dumps({
                    "case_id": case_id,
                    "source_sha256": fingerprints,
                    "first_frame_condition": str((args.first_frame_root / case_id).resolve()),
                    "first_frame_from_exact_png": True,
                    "latent_target_used_for_shape_only": True,
                    "camera_intrinsics_from_this_case": True,
                }, indent=2) + "\n", encoding="utf-8")
                print(f"wrote calibrated simple condition: {case_id}", flush=True)
        if out_path.exists():
            prior = json.loads((out_dir / "alignment.json").read_text(encoding="utf-8"))
            if prior.get("source_sha256") != fingerprints or pq.ParquetFile(out_path).metadata.num_rows != 1:
                raise ValueError(f"Existing GT condition does not match benchmark sources: {out_dir}")
            print(f"reusing GT condition: {case_id}", flush=True)
            continue
        pending.append((case_id, source_row, source_fps, source_hw, camera, intrinsics, fingerprints))

    if args.simple_only or not pending:
        return
    device = _initialize_worker(args.device_index, args.master_port)
    model_path = args.model_path.resolve()
    vae = _load_component("vae", model_path, _load_model_index(model_path), _fastvideo_args(model_path))
    vae.eval()
    try:
        for case_id, row, source_fps, source_hw, camera, intrinsics, fingerprints in pending:
            source_video = args.benchmark_root / case_id / "sample_001_gt.mp4"
            frames, _, decoded_hw = read_gt_video(source_video, pixels=True)
            assert frames is not None and decoded_hw == source_hw
            aligned = np.ascontiguousarray(frames[source_indices()])
            video = torch.from_numpy(aligned).permute(3, 0, 1, 2).contiguous()
            target_latent, _ = encode_h3_worldmodel_video(vae, video, device=device)
            if tuple(target_latent.shape) != (24, 37, 30, 52):
                raise ValueError(f"Unexpected GT VAE latent: {case_id} {tuple(target_latent.shape)}")
            row = dict(row)
            row["id"] = f"js_easy_benchmark/{case_id}/gt_aligned"
            row["source_frame_indices"] = source_indices().tolist()
            row["source_fps"] = source_fps
            row["fps"] = 24.0
            row["num_frames"] = TARGET_FRAMES
            row["duration_sec"] = TARGET_FRAMES / 24.0
            _set_float32_tensor(row, "vae_latent", target_latent.numpy())
            _set_float32_tensor(row, "camera_to_world", camera)
            _set_float32_tensor(row, "camera_intrinsics", intrinsics)
            _set_float32_tensor(row, "action_continuous", build_normalized_local_camera_actions(camera))
            _set_float32_tensor(row, "action_valid_mask", np.ones(37, dtype=np.float32))
            out_dir = args.output_root / case_id
            out_dir.mkdir(parents=True, exist_ok=True)
            temporary = out_dir / "data_000000.parquet.tmp"
            out_path = out_dir / "data_000000.parquet"
            pq.write_table(pa.Table.from_pylist([row], schema=pyarrow_schema_minimax_h3_worldmodel_prope),
                           temporary, compression="zstd")
            os.replace(temporary, out_path)
            alignment = {
                "case_id": case_id,
                "source_video": str(source_video.resolve()),
                "source_first_frame_png": str((args.benchmark_root / case_id / "sample_001_first_frame.png").resolve()),
                "source_sha256": fingerprints,
                "source_frames": SOURCE_FRAMES,
                "source_fps": source_fps,
                "source_hw": list(source_hw),
                "target_frames": TARGET_FRAMES,
                "target_fps": 24.0,
                "target_hw": list(TARGET_HW),
                "temporal_alignment": "nearest_frame_at_equal_normalized_progress",
                "source_frame_indices": source_indices().tolist(),
                "latent_anchor_indices": h3_latent_source_frame_indices(TARGET_FRAMES).tolist(),
                "pose_source": "sample_001.json frame_actions.mouse[].c2w (corrected)",
                "intrinsics_source": "sample_001_pose.npz pixel K normalized by pose processed_hw before crop",
                "first_frame_condition": str((args.first_frame_root / case_id).resolve()),
                "first_frame_from_exact_png": True,
                "gt_first_frame_may_differ_by_video_compression": True,
            }
            (out_dir / "alignment.json").write_text(json.dumps(alignment, indent=2) + "\n", encoding="utf-8")
            print(f"encoded aligned GT: {case_id}", flush=True)
            del aligned, frames, video, target_latent
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
