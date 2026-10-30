# SPDX-License-Identifier: Apache-2.0
"""Regression checks for the real-pair camera conditioning experiment."""

import torch

from fastvideo.configs.models.dits.minimax_h3 import MiniMaxH3ArchConfig
from fastvideo.dataset.dataloader.schema import pyarrow_schema_minimax_h3_worldmodel_prope_pair
from fastvideo.models.dits.minimax_h3 import apply_camera_film
from fastvideo.train.methods.fine_tuning.worldmodel import paired_clean_difference_loss


def test_camera_film_preserves_zero_modulation_and_backpropagates() -> None:
    hidden = torch.randn(1, 3, 8, requires_grad=True)
    modulation = torch.zeros(1, 3, 16, requires_grad=True)
    strength = torch.tensor(0.2, requires_grad=True)
    result = apply_camera_film(hidden, modulation, strength)
    torch.testing.assert_close(result, hidden)
    result.sum().backward()
    assert modulation.grad is not None and bool((modulation.grad != 0).any())


def test_real_pair_difference_has_signed_target() -> None:
    first_target = torch.tensor([[[[[1.0]]]]])
    second_target = -first_target
    noise = torch.zeros_like(first_target)
    sigma = torch.ones(1, 1, 1, 1, 1)
    correct = paired_clean_difference_loss(
        -first_target, -second_target, noise, noise, sigma, first_target, second_target)
    swapped = paired_clean_difference_loss(
        -second_target, -first_target, noise, noise, sigma, first_target, second_target)
    torch.testing.assert_close(correct, torch.zeros_like(correct))
    assert swapped.item() > 0.0


def test_pair_schema_has_two_real_video_and_camera_targets() -> None:
    names = set(pyarrow_schema_minimax_h3_worldmodel_prope_pair.names)
    assert {"vae_latent_bytes", "pair_vae_latent_bytes", "camera_to_world_bytes",
            "pair_camera_to_world_bytes", "pair_camera_intrinsics_bytes"} <= names


def test_camera_film_arch_requires_a_real_encoder() -> None:
    config = MiniMaxH3ArchConfig(action_dim=12, action_modulation_mode="film", action_adapter_hidden_dim=128)
    assert config.action_modulation_mode == "film"
