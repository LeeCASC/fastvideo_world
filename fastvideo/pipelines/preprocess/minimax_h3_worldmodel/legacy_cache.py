# SPDX-License-Identifier: Apache-2.0
"""Adapt legacy DiffSynth MiniMax-H3 caches to FastVideo world-model records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from collections.abc import Sequence

import numpy as np
import torch
from vwm_datasets import VWMDataset
from vwm_datasets.transforms import transform_intrinsic_for_resize_center_crop

from fastvideo.dataset.dataloader.record_schema import (
    minimax_h3_worldmodel_prope_record_from_additive,
    minimax_h3_worldmodel_record_creator,
)
from fastvideo.dataset.vwm_h3 import (
    build_normalized_local_camera_actions,
    h3_latent_source_frame_indices,
)
from fastvideo.pipelines.basic.minimax_h3.packing import (
    MINIMAX_H3_KEYFRAME_ENCODE_SEED,
    patchify_video_latents,
    unpatchify_video_tokens,
)


LEGACY_H3_HEIGHT = 480
LEGACY_H3_WIDTH = 832
LEGACY_H3_NUM_FRAMES = 124
LEGACY_H3_FPS = 24.0
LEGACY_H3_VIDEO_SHAPE = (24, 37, 30, 52)


def legacy_h3_source_frame_indices(
    *,
    source_num_frames: int,
    source_fps: float,
    target_num_frames: int = LEGACY_H3_NUM_FRAMES,
    target_fps: float = LEGACY_H3_FPS,
) -> np.ndarray:
    """Reproduce DiffSynth's fixed-rate frame mapping, including Python rounding."""
    source_num_frames = int(source_num_frames)
    source_fps = float(source_fps)
    if source_num_frames <= 0 or source_fps <= 0.0:
        raise ValueError("source_num_frames and source_fps must be positive")
    indices = [
        min(int(round(frame / float(target_fps) * source_fps)), source_num_frames - 1)
        for frame in range(int(target_num_frames))
    ]
    return np.asarray(indices, dtype=np.int64)


def recover_legacy_h3_keyframe_latent(
    keyframe_anchor: torch.Tensor,
    video_latent_shape: Sequence[int],
    *,
    noise_strength: float,
    seed: int = MINIMAX_H3_KEYFRAME_ENCODE_SEED,
) -> torch.Tensor:
    """Undo the legacy 0.999 keyframe-noise blend and return ``[24,1,H,W]``."""
    channels, video_frames, latent_height, latent_width = (
        int(value) for value in video_latent_shape
    )
    expected_rows = (latent_height // 2) * (latent_width // 2)
    expected_width = channels * 4
    if keyframe_anchor.shape != (expected_rows, expected_width):
        raise ValueError(
            "legacy keyframe anchor has an unexpected shape: "
            f"expected {(expected_rows, expected_width)}, got {tuple(keyframe_anchor.shape)}"
        )
    if not 0.0 < float(noise_strength) <= 1.0:
        raise ValueError("legacy keyframe noise strength must be in (0, 1]")

    anchor = keyframe_anchor.detach().cpu().contiguous()
    strength = torch.tensor(float(noise_strength), dtype=anchor.dtype)
    if float(strength) == 1.0:
        clean_rows = anchor.float()
    else:
        # DiffSynth generated a full target-plus-keyframe tensor before taking
        # temporal index zero. Reproducing the complete shape is part of its RNG
        # contract; generating only one frame changes later channels' samples.
        generator = torch.Generator("cpu").manual_seed(int(seed))
        noise = torch.randn(
            (1, channels, video_frames + 1, latent_height, latent_width),
            generator=generator,
            dtype=anchor.dtype,
        )[:, :, :1]
        noise_rows = patchify_video_latents(noise, (1, 2, 2)).float()
        noise_weight = (torch.ones((), dtype=anchor.dtype) - strength).float()
        clean_rows = (anchor.float() - noise_weight * noise_rows) / strength.float()

    keyframe = unpatchify_video_tokens(
        clean_rows,
        1,
        latent_height,
        latent_width,
        channels,
        (1, 2, 2),
    )[0].float().contiguous()
    if not torch.isfinite(keyframe).all():
        raise ValueError("recovered legacy H3 keyframe contains non-finite values")
    return keyframe


def extract_legacy_h3_cache_payload(
    cache_path: Path,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract audio-free native tensors and remove legacy action-text rows."""
    payload = torch.load(cache_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, tuple) or len(payload) != 3:
        raise ValueError(f"legacy H3 cache must contain a three-item tuple: {cache_path}")
    shared, positive, negative = payload
    if not all(isinstance(value, dict) for value in payload) or negative:
        raise ValueError(f"legacy H3 cache has an unexpected payload layout: {cache_path}")

    input_latents = shared.get("input_latents")
    keyframe_anchor = shared.get("keyframe_cond_anchor")
    prompt_embeds = positive.get("prompt_embeds")
    packed = positive.get("packed")
    if not all(torch.is_tensor(value) for value in (input_latents, keyframe_anchor, prompt_embeds)):
        raise ValueError(f"legacy H3 cache is missing required tensors: {cache_path}")
    if not isinstance(packed, dict):
        raise ValueError(f"legacy H3 cache is missing packed metadata: {cache_path}")
    if tuple(input_latents.shape) != (1, *LEGACY_H3_VIDEO_SHAPE):
        raise ValueError(
            f"legacy H3 video latent must have shape {(1, *LEGACY_H3_VIDEO_SHAPE)}, "
            f"got {tuple(input_latents.shape)} in {cache_path}"
        )

    spans = packed.get("action_text_spans_local")
    if not isinstance(spans, list) or len(spans) != LEGACY_H3_VIDEO_SHAPE[1]:
        raise ValueError(f"legacy H3 cache has no complete per-latent action spans: {cache_path}")
    head_length = int(spans[0][0])
    if head_length <= 0 or int(spans[-1][1]) != int(prompt_embeds.shape[0]):
        raise ValueError(f"legacy H3 action spans do not partition the prompt tail: {cache_path}")

    text_pos = packed.get("text_pos")
    token_tags = packed.get("token_tags")
    if not torch.is_tensor(text_pos) or not torch.is_tensor(token_tags):
        raise ValueError(f"legacy H3 cache has no text positions/token tags: {cache_path}")
    text_tags = token_tags.index_select(0, text_pos.to(torch.long))[:head_length].to(torch.int64)
    if not torch.isin(text_tags, torch.tensor([0, 1], dtype=text_tags.dtype)).all():
        raise ValueError(f"legacy H3 visual/text head contains invalid token tags: {cache_path}")

    video_latent = input_latents[0].float().contiguous()
    first_frame_latent = recover_legacy_h3_keyframe_latent(
        keyframe_anchor,
        video_latent.shape,
        noise_strength=float(shared.get("imgvid_cond_noise_aug", 1.0)),
    )
    text_embedding = prompt_embeds[:head_length].float().contiguous()
    tensors = (video_latent, first_frame_latent, text_embedding, text_tags.contiguous())
    if not all(torch.isfinite(value).all() for value in tensors):
        raise ValueError(f"legacy H3 cache contains non-finite training tensors: {cache_path}")
    return tensors


class LegacyH3WorldModelCache:
    """Validated indexed view of the legacy PTH cache and canonical VWM geometry."""

    def __init__(
        self,
        *,
        metadata_path: Path,
        cache_dir: Path,
        vwm_manifest_path: Path,
    ) -> None:
        self.metadata_path = metadata_path.resolve(strict=True)
        self.cache_dir = cache_dir.resolve(strict=True)
        self.rows = [
            json.loads(line)
            for line in self.metadata_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not self.rows:
            raise ValueError(f"legacy H3 metadata is empty: {self.metadata_path}")

        cache_paths: dict[int, Path] = {}
        for path in self.cache_dir.rglob("*.pth"):
            try:
                index = int(path.stem)
            except ValueError as error:
                raise ValueError(f"legacy H3 cache filename is not numeric: {path}") from error
            if index in cache_paths:
                raise ValueError(f"duplicate legacy H3 cache index {index}")
            cache_paths[index] = path
        expected_indices = set(range(len(self.rows)))
        if set(cache_paths) != expected_indices:
            missing = sorted(expected_indices - set(cache_paths))[:8]
            extra = sorted(set(cache_paths) - expected_indices)[:8]
            raise ValueError(f"legacy H3 cache indices are incomplete: missing={missing}, extra={extra}")
        self.cache_paths = cache_paths

        self.vwm_dataset = VWMDataset(vwm_manifest_path)
        self.vwm_indices: dict[str, int] = {}
        for index, record in enumerate(self.vwm_dataset.records):
            identity = f"{record.source_name}/{record.sample_id}"
            if identity in self.vwm_indices:
                raise ValueError(f"duplicate compound VWM identity: {identity}")
            self.vwm_indices[identity] = index

        identities = [self.identity(index) for index in range(len(self.rows))]
        if len(identities) != len(set(identities)):
            raise ValueError("legacy H3 metadata contains duplicate compound identities")
        missing_geometry = [identity for identity in identities if identity not in self.vwm_indices]
        if missing_geometry:
            raise KeyError(f"legacy H3 records are absent from the VWM manifest: {missing_geometry[:8]}")

    def __len__(self) -> int:
        return len(self.rows)

    def identity(self, index: int) -> str:
        row = self.rows[index]
        return f"{row['source_name']}/{row['sample_id']}"

    def record(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        identity = self.identity(index)
        clip = self.vwm_dataset[self.vwm_indices[identity]]
        source_num_frames = int(row["source_num_frames"])
        source_fps = float(row["source_fps"])
        if source_num_frames != int(clip.num_frames) or not np.isclose(source_fps, float(clip.fps)):
            raise ValueError(
                f"legacy metadata and canonical VWM geometry disagree for {identity}: "
                f"frames={source_num_frames}/{clip.num_frames}, fps={source_fps}/{clip.fps}"
            )
        if int(row.get("target_num_frames", LEGACY_H3_NUM_FRAMES)) != LEGACY_H3_NUM_FRAMES:
            raise ValueError(f"legacy H3 target frame count changed for {identity}")
        if not np.isclose(float(row.get("target_fps", LEGACY_H3_FPS)), LEGACY_H3_FPS):
            raise ValueError(f"legacy H3 target FPS changed for {identity}")

        source_indices = legacy_h3_source_frame_indices(
            source_num_frames=source_num_frames,
            source_fps=source_fps,
        )
        latent_indices = h3_latent_source_frame_indices(LEGACY_H3_NUM_FRAMES)
        camera_to_world = np.asarray(clip.camera_to_world, dtype=np.float32)[source_indices][latent_indices]
        raw_intrinsics = np.asarray(clip.intrinsics, dtype=np.float32)
        if raw_intrinsics.shape == (3, 3):
            source_intrinsics = np.broadcast_to(raw_intrinsics, (len(latent_indices), 3, 3))
        elif raw_intrinsics.shape == (source_num_frames, 3, 3):
            source_intrinsics = raw_intrinsics[source_indices][latent_indices]
        else:
            raise ValueError(f"unsupported VWM intrinsics shape for {identity}: {raw_intrinsics.shape}")
        camera_intrinsics = np.stack([
            transform_intrinsic_for_resize_center_crop(
                value,
                src_size=(int(clip.height), int(clip.width)),
                dst_size=(LEGACY_H3_HEIGHT, LEGACY_H3_WIDTH),
            )
            for value in source_intrinsics
        ])
        actions = build_normalized_local_camera_actions(camera_to_world)
        video, first_frame, text_embedding, text_tags = extract_legacy_h3_cache_payload(
            self.cache_paths[index]
        )

        additive = minimax_h3_worldmodel_record_creator(
            sample_id=identity,
            source_name=str(row["source_name"]),
            caption=str(row["prompt"]),
            vae_latent=video.numpy(),
            first_frame_latent=first_frame.numpy(),
            text_embedding=text_embedding.numpy(),
            text_token_tags=text_tags.numpy(),
            action_continuous=actions,
            action_valid_mask=np.ones(len(latent_indices), dtype=np.float32),
            source_frame_indices=source_indices,
            width=LEGACY_H3_WIDTH,
            height=LEGACY_H3_HEIGHT,
            fps=LEGACY_H3_FPS,
            source_fps=source_fps,
        )
        return minimax_h3_worldmodel_prope_record_from_additive(
            additive,
            camera_to_world=camera_to_world,
            camera_intrinsics=camera_intrinsics,
        )


__all__ = [
    "LegacyH3WorldModelCache",
    "extract_legacy_h3_cache_payload",
    "legacy_h3_source_frame_indices",
    "recover_legacy_h3_keyframe_latent",
]
