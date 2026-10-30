# SPDX-License-Identifier: Apache-2.0
"""Checkpoint-compatible MiniMax-H3 world-model encoding boundaries."""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image
import torch

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.pipelines import ForwardBatch
from fastvideo.pipelines.basic.minimax_h3.packing import MINIMAX_H3_KEYFRAME_ENCODE_SEED
from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_conditioning import (
    MINIMAX_H3_TEXT_TOKEN_TAGS_KEY,
    MiniMaxH3ConditioningStage,
)
from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_input_preparation import MINIMAX_H3_KEYFRAMES_KEY


def _sample_posterior(posterior: Any) -> torch.Tensor:
    generator = torch.Generator("cpu").manual_seed(MINIMAX_H3_KEYFRAME_ENCODE_SEED)
    return posterior.sample(generator=generator)


@torch.no_grad()
def encode_h3_worldmodel_video(
    vae: Any,
    video: torch.Tensor,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode H3 target video and its native first-frame VAE condition.

    ``video`` is canonical uint8 RGB ``[3,T,H,W]``. Returned tensors are
    normalized contiguous float32 CPU latents shaped ``[24,T',H',W']`` and
    ``[24,1,H',W']`` respectively.
    """
    if video.ndim != 4 or video.shape[0] != 3 or video.shape[1] <= 0:
        raise ValueError(f"video must have shape [3,T,H,W] with T > 0, got {tuple(video.shape)}")
    if video.dtype != torch.uint8:
        raise ValueError(f"video must use uint8 RGB pixels, got {video.dtype}")

    # H3's native helper streams causal VAE chunks from CPU instead of keeping
    # the complete float32 video on the accelerator.
    target_posterior = vae.encode_pixels(video.unsqueeze(0).cpu()).latent_dist
    target = vae.normalize_latents(_sample_posterior(target_posterior))

    first_pixels = video[:, :1].unsqueeze(0).to(device=device, dtype=torch.float32).div_(255.0)
    first_posterior = vae.encode_keyframe(vae.normalize_pixels(first_pixels)).latent_dist
    # Match MiniMaxH3LatentPreparationStage._encode_keyframe_latents exactly:
    # released FL2VA conditioning preserves this seeded FP16 round-trip before
    # applying the per-channel latent normalization.
    first_sample = _sample_posterior(first_posterior).to(torch.float16).float()
    first_frame = vae.normalize_latents(first_sample)
    target = target.squeeze(0).float().cpu().contiguous()
    first_frame = first_frame.squeeze(0).float().cpu().contiguous()

    if target.ndim != 4 or target.shape[0] != 24:
        raise ValueError(f"H3 VAE target must have shape [24,T,H,W], got {tuple(target.shape)}")
    if first_frame.shape != (24, 1, target.shape[2], target.shape[3]):
        raise ValueError(
            "H3 keyframe latent must have shape [24,1,H,W] matching the target, "
            f"got {tuple(first_frame.shape)}"
        )
    return target, first_frame


@torch.no_grad()
def encode_h3_worldmodel_condition(
    stage: MiniMaxH3ConditioningStage,
    fastvideo_args: FastVideoArgs,
    *,
    caption: str,
    first_frame: Image.Image | np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode the native FL2VA first-frame presentation and caption.

    The released H3 conditioning stage owns tokenization, Qwen3-VL vision
    processing, hidden-state selection, and semantic video/text token tags.
    """
    if not str(caption).strip():
        raise ValueError("caption must be non-empty")
    if isinstance(first_frame, np.ndarray):
        array = np.asarray(first_frame)
        if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
            raise ValueError("first_frame array must be uint8 RGB [H,W,3]")
        image = Image.fromarray(array)
    elif isinstance(first_frame, Image.Image):
        image = first_frame.convert("RGB")
    else:
        raise TypeError(f"first_frame must be a PIL image or NumPy array, got {type(first_frame).__name__}")

    batch = ForwardBatch(data_type="video", prompt=str(caption).strip())
    batch.extra[MINIMAX_H3_KEYFRAMES_KEY] = [image]
    result = stage.forward(batch, fastvideo_args)
    if len(result.prompt_embeds) != 1:
        raise RuntimeError("MiniMax-H3 conditioning must return exactly one prompt embedding")
    embedding = result.prompt_embeds[0]
    tags = result.extra.get(MINIMAX_H3_TEXT_TOKEN_TAGS_KEY)
    if not isinstance(tags, torch.Tensor):
        raise RuntimeError("MiniMax-H3 conditioning returned no semantic token tags")
    if embedding.ndim != 3 or embedding.shape[0] != 1 or embedding.shape[-1] != 5120:
        raise ValueError(f"H3 condition embedding must have shape [1,L,5120], got {tuple(embedding.shape)}")
    if tags.shape != (embedding.shape[1],):
        raise ValueError("H3 semantic token tags must align with the condition embedding")
    return embedding.squeeze(0).float().cpu().contiguous(), tags.to(torch.int64).cpu().contiguous()
