# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for the parameter-free H3 camera-motion observer."""

import pytest
import torch

from fastvideo.train.models.minimax_h3.camera_motion_probe import (
    latent_soft_translation,
    masked_motion_loss,
)


def test_latent_soft_translation_recovers_known_shift_and_input_gradient():
    torch.manual_seed(7)
    first = torch.randn(1, 1, 3, 12, 14)
    second = torch.zeros_like(first)
    second[..., 1:, 2:] = first[..., :-1, :-2]
    video = torch.cat((first, second), dim=1).requires_grad_()

    motion = latent_soft_translation(video, max_shift=3, temperature=0.01)
    motion.square().mean().backward()

    torch.testing.assert_close(motion, torch.tensor([[[2.0, 1.0]]]), atol=0.1, rtol=0.0)
    assert video.grad is not None and bool((video.grad != 0).any())


def test_latent_soft_translation_reports_zero_for_static_video():
    frame = torch.randn(1, 1, 2, 10, 10)
    video = frame.expand(-1, 3, -1, -1, -1).clone()

    motion = latent_soft_translation(video, max_shift=2, temperature=0.01)

    torch.testing.assert_close(motion, torch.zeros_like(motion), atol=1.0e-4, rtol=0.0)


def test_masked_motion_loss_ignores_invalid_transitions():
    prediction = torch.zeros(1, 2, 2)
    target = torch.zeros_like(prediction)
    target[:, 1] = 100.0

    loss = masked_motion_loss(prediction, target, torch.tensor([[True, False]]))

    torch.testing.assert_close(loss, torch.tensor(0.0))
    with pytest.raises(ValueError, match="valid_mask"):
        masked_motion_loss(prediction, target, torch.ones(1, 3, dtype=torch.bool))
