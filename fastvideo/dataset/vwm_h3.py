# SPDX-License-Identifier: Apache-2.0
"""MiniMax-H3 adapter for canonical VWM v2 clips.

``vwm_datasets`` owns manifest parsing, clip decoding, camera validation, and
deterministic resize/crop. This module only owns H3-specific temporal sampling,
tensor layout, train/validation selection, and camera-action construction.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
import torch
from torch.utils.data import Dataset
from vwm_datasets import VWMDataset, derive_frame_pyr_from_view_se3
from vwm_datasets.transforms import resize_center_crop

from fastvideo.pipelines.basic.minimax_h3.packing import (
    MINIMAX_H3_FPS,
    MINIMAX_H3_FRAMES_PER_CHUNK,
    MINIMAX_H3_LATENTS_PER_CHUNK,
    MINIMAX_H3_ROPE_FRAMES_PER_LATENT,
    video_latent_num_frames,
)

H3_CAMERA_ACTION_NAMES = (
    "local_forward",
    "local_right",
    "local_up",
    "delta_pitch_half_turn",
    "delta_yaw_half_turn",
    "delta_roll_half_turn",
)


def canonicalize_source_intrinsics(
    intrinsics: np.ndarray,
    *,
    source_name: str,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    """Convert the known JS-easy VGGT pixel-K convention before image cropping.

    JS-easy videos are 1344x768, but their VGGT calibration was performed at
    672x384. Other VWM sources already store normalized K. Reject any other
    pixel-valued K rather than silently inventing a calibration resolution.
    """
    value = np.asarray(intrinsics, dtype=np.float32).copy()
    if value.shape[-2:] != (3, 3) or value.ndim not in (2, 3) or not np.isfinite(value).all():
        raise ValueError("camera intrinsics must be finite [3,3] or [T,3,3]")
    if source_name in {
        "h3_js_easy_fasth3_t2v_js_1000_124",
        "h3_js_easy_fasth3_t2v_js_1000_long",
        "h3_js_easy_h3_i2v_js_easy_124",
        "h3_js_easy_h3_i2v_js_easy_long",
    }:
        if (image_width, image_height) != (1344, 768):
            raise ValueError("JS-easy VGGT intrinsics require 1344x768 source video")
        if not (np.allclose(value[..., 0, 2], 336.0, atol=0.5)
                and np.allclose(value[..., 1, 2], 192.0, atol=0.5)):
            raise ValueError("JS-easy VGGT calibration must have a 672x384 centered principal point")
        value[..., 0, :] /= 672.0
        value[..., 1, :] /= 384.0
    if (np.any(value[..., 0, 0] <= 0.0) or np.any(value[..., 1, 1] <= 0.0)
            or np.any(value[..., 0, 0] > 32.0) or np.any(value[..., 1, 1] > 32.0)
            or np.any(np.abs(value[..., :2, 2]) > 2.0)):
        raise ValueError(f"{source_name}: camera intrinsics are not normalized")
    return value


def rebase_camera_to_first_frame(camera_to_world: np.ndarray) -> np.ndarray:
    """Express a sampled camera path in its first frame's coordinate system."""
    poses = np.asarray(camera_to_world, dtype=np.float32)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or not len(poses):
        raise ValueError(f"camera_to_world must have shape [T,4,4], got {poses.shape}")
    if not np.isfinite(poses).all():
        raise ValueError("camera_to_world contains nonfinite values")
    origin_inverse = np.linalg.inv(poses[0])
    return np.ascontiguousarray(origin_inverse @ poses, dtype=np.float32)


class _ManifestRecord(Protocol):
    source_name: str
    sample_id: str


def _stable_fraction(value: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{value}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def source_balanced_manifest_indices(
    records: Sequence[_ManifestRecord],
    selected_indices: Sequence[int],
    seed: int,
) -> list[int]:
    """Interleave sources while deterministically shuffling within each source.

    VWM train_v2 is grouped by source in its manifest. Interleaving makes every
    finite preprocessing prefix useful for pilot training without changing the
    membership or source frequencies of the complete dataset.
    """
    by_source: dict[str, list[int]] = defaultdict(list)
    for index in selected_indices:
        record = records[index]
        source_name = str(record.source_name)
        by_source[source_name].append(index)

    for source_name, indices in by_source.items():
        indices.sort(
            key=lambda index: (
                _stable_fraction(
                    f"{source_name}/{records[index].sample_id}",
                    seed,
                ),
                index,
            )
        )

    balanced: list[int] = []
    offsets = {source_name: 0 for source_name in by_source}
    active_sources = sorted(by_source)
    while active_sources:
        next_active: list[str] = []
        for source_name in active_sources:
            offset = offsets[source_name]
            indices = by_source[source_name]
            if offset < len(indices):
                balanced.append(indices[offset])
                offsets[source_name] = offset + 1
            if offsets[source_name] < len(indices):
                next_active.append(source_name)
        active_sources = next_active
    return balanced


def source_limited_manifest_indices(
    records: Sequence[_ManifestRecord],
    source_limits: dict[str, int],
    seed: int,
) -> list[int]:
    """Select deterministic per-source subsets while retaining every uncapped source."""
    if any(int(limit) <= 0 for limit in source_limits.values()):
        raise ValueError("source limits must be positive")
    by_source: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_source[str(record.source_name)].append(index)
    unknown_sources = set(source_limits) - set(by_source)
    if unknown_sources:
        raise ValueError(f"source limits reference unknown sources: {sorted(unknown_sources)}")

    selected: list[int] = []
    for source_name, indices in by_source.items():
        limit = source_limits.get(source_name)
        if limit is not None and limit < len(indices):
            indices = sorted(
                indices,
                key=lambda index: (
                    _stable_fraction(f"{source_name}/{records[index].sample_id}", seed),
                    index,
                ),
            )[:limit]
        selected.extend(indices)
    return sorted(selected)


def h3_latent_source_frame_indices(num_frames: int) -> np.ndarray:
    """Return the raw-frame anchor represented by each H3 video latent."""
    latent_frames = video_latent_num_frames(int(num_frames))
    spans = np.asarray(
        [MINIMAX_H3_ROPE_FRAMES_PER_LATENT[index % len(MINIMAX_H3_ROPE_FRAMES_PER_LATENT)]
         for index in range(latent_frames)],
        dtype=np.int64,
    )
    if int(spans.sum()) != int(num_frames):
        raise ValueError(
            f"H3 temporal spans cover {int(spans.sum())} frames, expected {num_frames}."
        )
    return np.cumsum(spans) - 1


def resample_source_frame_indices(
    *,
    source_num_frames: int,
    source_fps: float,
    target_num_frames: int,
    target_fps: float = MINIMAX_H3_FPS,
    sample_key: str = "",
    seed: int = 0,
    min_source_duration_seconds: float = 4.5,
) -> np.ndarray:
    """Select one deterministic source window on the H3 24-FPS time grid.

    Clips long enough for the requested H3 duration preserve wall-clock time.
    Slightly shorter train-v2 clips are stretched over the complete source clip;
    clips below ``min_source_duration_seconds`` are rejected.
    """
    source_num_frames = int(source_num_frames)
    target_num_frames = int(target_num_frames)
    source_fps = float(source_fps)
    target_fps = float(target_fps)
    if source_num_frames <= 0 or target_num_frames <= 0:
        raise ValueError("source_num_frames and target_num_frames must be positive")
    if source_fps <= 0.0 or target_fps <= 0.0:
        raise ValueError("source_fps and target_fps must be positive")
    if target_num_frames % MINIMAX_H3_FRAMES_PER_CHUNK != MINIMAX_H3_LATENTS_PER_CHUNK:
        raise ValueError(
            f"target_num_frames must have the H3 form 17 * n + 5, got {target_num_frames}."
        )

    source_duration = (source_num_frames - 1) / source_fps
    target_duration = (target_num_frames - 1) / target_fps
    if source_duration < float(min_source_duration_seconds):
        raise ValueError(
            f"source clip is {source_duration:.3f}s; at least "
            f"{float(min_source_duration_seconds):.3f}s is required"
        )

    if source_duration >= target_duration:
        max_start_seconds = source_duration - target_duration
        start_seconds = _stable_fraction(sample_key, seed) * max_start_seconds
        source_positions = (start_seconds + np.arange(target_num_frames) / target_fps) * source_fps
    else:
        source_positions = np.linspace(0.0, source_num_frames - 1, target_num_frames)

    indices = np.rint(source_positions).astype(np.int64)
    return np.clip(indices, 0, source_num_frames - 1)


def build_local_camera_actions(camera_to_world: np.ndarray) -> np.ndarray:
    """Build ``[forward,right,up,pitch,yaw,roll]`` transitions from c2w poses.

    Translations are expressed in the previous camera frame. Rotations use the
    canonical VWM X-right/Y-up/Z-forward pitch/yaw/roll convention in degrees.
    Action zero is the idle transition associated with the conditioning frame.
    """
    camera_to_world = np.asarray(camera_to_world, dtype=np.float32)
    if camera_to_world.ndim != 3 or camera_to_world.shape[1:] != (4, 4):
        raise ValueError(f"camera_to_world must have shape [T,4,4], got {camera_to_world.shape}")

    actions = np.zeros((camera_to_world.shape[0], len(H3_CAMERA_ACTION_NAMES)), dtype=np.float32)
    if camera_to_world.shape[0] <= 1:
        return actions

    rotations = derive_frame_pyr_from_view_se3(camera_to_world)
    for index in range(1, camera_to_world.shape[0]):
        delta_world = camera_to_world[index, :3, 3] - camera_to_world[index - 1, :3, 3]
        local_right, local_up, local_forward = camera_to_world[index - 1, :3, :3].T @ delta_world
        rotation_delta = [
            ((float(rotations[index, axis] - rotations[index - 1, axis]) + 180.0) % 360.0) - 180.0
            for axis in range(3)
        ]
        actions[index] = np.asarray(
            [local_forward, local_right, local_up, *rotation_delta],
            dtype=np.float32,
        )
    return actions


def normalize_camera_trajectory_translation(camera_to_world: np.ndarray) -> np.ndarray:
    """Remove source units using Stage-0 relative-max-norm normalization.

    Rotations are preserved. Translation starts at the origin, and the maximum
    displacement from the first camera is one. A stationary trajectory remains
    exactly zero.
    """
    camera_to_world = np.asarray(camera_to_world, dtype=np.float32)
    if camera_to_world.ndim != 3 or camera_to_world.shape[1:] != (4, 4):
        raise ValueError(f"camera_to_world must have shape [T,4,4], got {camera_to_world.shape}")
    normalized = camera_to_world.copy()
    relative = normalized[:, :3, 3] - normalized[:1, :3, 3]
    scale = float(np.linalg.norm(relative, axis=1).max(initial=0.0))
    if scale >= 1e-8:
        relative /= scale
    normalized[:, :3, 3] = relative
    return normalized


def build_normalized_local_camera_actions(camera_to_world: np.ndarray) -> np.ndarray:
    """Build unitless local translation and half-turn rotation increments."""
    normalized_poses = normalize_camera_trajectory_translation(camera_to_world)
    actions = build_local_camera_actions(normalized_poses)
    actions[:, 3:] /= 180.0
    return actions


@dataclass(frozen=True)
class H3VWMDataConfig:
    """Deterministic raw-data policy used before offline H3 encoding."""

    manifest_path: str | Path
    split: Literal["train", "validation", "all"] = "train"
    validation_fraction: float = 0.01
    split_seed: int = 20260828
    height: int = 480
    width: int = 832
    num_frames: int = 124
    target_fps: float = MINIMAX_H3_FPS
    min_source_duration_seconds: float = 4.5
    source_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.split not in {"train", "validation", "all"}:
            raise ValueError(f"Unsupported split: {self.split!r}")
        if not 0.0 < float(self.validation_fraction) < 1.0:
            raise ValueError("validation_fraction must be between zero and one")
        if float(self.min_source_duration_seconds) < 0.0:
            raise ValueError("min_source_duration_seconds must be non-negative")
        if self.height <= 0 or self.width <= 0:
            raise ValueError("height and width must be positive")
        video_latent_num_frames(self.num_frames)


class H3VWMRawDataset(Dataset):
    """Adapt canonical VWM clips to raw H3 world-model preprocessing samples."""

    def __init__(self, config: H3VWMDataConfig) -> None:
        self.config = config
        self.vwm_dataset = VWMDataset(config.manifest_path)
        allowed_sources = set(config.source_names)
        selected_indices: list[int] = []
        for index, record in enumerate(self.vwm_dataset.records):
            if allowed_sources and record.source_name not in allowed_sources:
                continue
            duration = (record.num_frames - 1) / record.fps
            if duration < config.min_source_duration_seconds:
                continue
            validation = _stable_fraction(
                f"{record.source_name}/{record.sample_id}", config.split_seed
            ) < config.validation_fraction
            if config.split == "train" and validation:
                continue
            if config.split == "validation" and not validation:
                continue
            selected_indices.append(index)
        if not selected_indices:
            raise ValueError(f"No VWM records satisfy the {config.split!r} H3 data policy")
        self.selected_indices = source_balanced_manifest_indices(
            self.vwm_dataset.records,
            selected_indices,
            config.split_seed,
        )

    def __len__(self) -> int:
        return len(self.selected_indices)

    def __getitem__(self, index: int) -> dict[str, object]:
        source_index = self.selected_indices[index]
        record = self.vwm_dataset.records[source_index]
        sample_key = f"{record.source_name}/{record.sample_id}"
        source_frame_indices = resample_source_frame_indices(
            source_num_frames=record.num_frames,
            source_fps=record.fps,
            target_num_frames=self.config.num_frames,
            target_fps=self.config.target_fps,
            sample_key=sample_key,
            seed=self.config.split_seed,
            min_source_duration_seconds=self.config.min_source_duration_seconds,
        )
        sample = self.vwm_dataset.materialize(source_index, source_frame_indices)
        sample = replace(
            sample,
            intrinsics=canonicalize_source_intrinsics(
                sample.intrinsics,
                source_name=sample.source_name,
                image_width=sample.width,
                image_height=sample.height,
            ),
        )
        sample = resize_center_crop(sample, (self.config.height, self.config.width))

        latent_frame_indices = h3_latent_source_frame_indices(self.config.num_frames)
        camera_to_world = rebase_camera_to_first_frame(sample.camera_to_world)
        latent_camera_to_world = camera_to_world[latent_frame_indices]
        sample_intrinsics = np.asarray(sample.intrinsics, dtype=np.float32)
        if sample_intrinsics.shape == (3, 3):
            latent_intrinsics = np.broadcast_to(
                sample_intrinsics,
                (len(latent_frame_indices), 3, 3),
            ).copy()
        elif sample_intrinsics.shape == (len(sample.camera_to_world), 3, 3):
            latent_intrinsics = sample_intrinsics[latent_frame_indices].copy()
        else:
            raise ValueError(
                "VWM intrinsics must be static [3,3] or aligned [T,3,3], "
                f"got {sample_intrinsics.shape}"
            )
        action_continuous = build_normalized_local_camera_actions(latent_camera_to_world)
        video = torch.from_numpy(np.ascontiguousarray(sample.video)).permute(3, 0, 1, 2).contiguous()
        return {
            # sample_id is unique only within one VWM source. The compound key
            # is globally unique across the merged train-v2 manifest.
            "id": sample_key,
            "source_name": sample.source_name,
            "video": video,
            "caption": sample.text,
            "camera_to_world": torch.from_numpy(camera_to_world.copy()),
            "intrinsics": torch.from_numpy(sample.intrinsics.copy()),
            "latent_camera_to_world": torch.from_numpy(latent_camera_to_world.copy()),
            "latent_intrinsics": torch.from_numpy(latent_intrinsics.copy()),
            "source_frame_indices": torch.from_numpy(sample.source_frame_indices.copy()),
            "latent_frame_indices": torch.from_numpy(latent_frame_indices),
            "action_continuous": torch.from_numpy(action_continuous),
            "action_valid_mask": torch.ones(len(latent_frame_indices), dtype=torch.bool),
            "width": sample.width,
            "height": sample.height,
            "num_frames": self.config.num_frames,
            "fps": float(self.config.target_fps),
            "source_fps": float(record.fps),
        }


__all__ = [
    "H3_CAMERA_ACTION_NAMES",
    "H3VWMDataConfig",
    "H3VWMRawDataset",
    "build_local_camera_actions",
    "build_normalized_local_camera_actions",
    "h3_latent_source_frame_indices",
    "normalize_camera_trajectory_translation",
    "resample_source_frame_indices",
    "source_balanced_manifest_indices",
]
