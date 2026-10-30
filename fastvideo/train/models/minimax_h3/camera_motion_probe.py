# SPDX-License-Identifier: Apache-2.0
"""Parameter-free latent camera-motion observer for H3 supervision."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def latent_soft_translation(
    video: torch.Tensor,
    *,
    max_shift: int = 4,
    temperature: float = 0.02,
) -> torch.Tensor:
    """Estimate differentiable adjacent-frame global translation.

    For every candidate integer displacement, the observer computes cosine
    similarity over the valid spatial overlap, then returns the softmax
    expectation in latent-pixel ``[dx,dy]`` units. It has no trainable state,
    so generator gradients cannot make the observer co-adapt or forget what
    motion means.
    """
    if video.ndim != 5 or video.shape[1] < 2:
        raise ValueError(f"video must have shape [B,T,C,H,W] with T>=2, got {tuple(video.shape)}")
    if max_shift <= 0 or 2 * max_shift >= min(video.shape[-2:]):
        raise ValueError("max_shift must be positive and smaller than half the latent spatial size")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    previous = video[:, :-1].float()
    current = video[:, 1:].float()
    correlations: list[torch.Tensor] = []
    offsets: list[tuple[int, int]] = []
    for delta_y in range(-max_shift, max_shift + 1):
        previous_y = slice(max(0, -delta_y), min(video.shape[-2], video.shape[-2] - delta_y))
        current_y = slice(max(0, delta_y), min(video.shape[-2], video.shape[-2] + delta_y))
        for delta_x in range(-max_shift, max_shift + 1):
            previous_x = slice(max(0, -delta_x), min(video.shape[-1], video.shape[-1] - delta_x))
            current_x = slice(max(0, delta_x), min(video.shape[-1], video.shape[-1] + delta_x))
            previous_crop = previous[..., previous_y, previous_x].flatten(2)
            current_crop = current[..., current_y, current_x].flatten(2)
            correlations.append(F.cosine_similarity(previous_crop, current_crop, dim=-1))
            offsets.append((delta_x, delta_y))
    scores = torch.stack(correlations, dim=-1)
    probability = torch.softmax(scores / float(temperature), dim=-1)
    offset_tensor = video.new_tensor(offsets, dtype=torch.float32)
    return torch.einsum("btn,nd->btd", probability, offset_tensor)


def masked_motion_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Smooth-L1 motion error averaged across valid transitions."""
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("prediction and target must share shape [B,T,D]")
    if valid_mask.shape != prediction.shape[:2]:
        raise ValueError("valid_mask must have shape [B,T]")
    per_transition = F.smooth_l1_loss(prediction.float(), target.float(), reduction="none").mean(dim=-1)
    weights = valid_mask.float()
    return (per_transition * weights).sum() / weights.sum().clamp_min(1.0)


__all__ = ["latent_soft_translation", "masked_motion_loss"]
