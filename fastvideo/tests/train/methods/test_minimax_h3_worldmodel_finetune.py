# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for MiniMax-H3 world-model auxiliary objectives."""

from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from fastvideo.train.methods.fine_tuning.worldmodel import (
    MiniMaxH3WorldModelFineTuneMethod,
    lowpass_video,
    masked_motion_direction_loss,
    predicted_clean_video,
    temporal_delta_error,
)
from fastvideo.train.utils.dataloader import build_motion_balanced_indices


class _StaticPredictionStudent:
    def __init__(self, prediction: torch.Tensor):
        self.prediction = prediction
        self.cameras: list[torch.Tensor] = []
        self.noisy_inputs: list[torch.Tensor] = []

    def predict_noise(self, *_args, **_kwargs):
        self.cameras.append(_args[2].input_kwargs["camera_to_world"].clone())
        self.noisy_inputs.append(_args[0].clone())
        return self.prediction


def test_predicted_clean_video_inverts_rectified_flow_velocity():
    clean = torch.randn(1, 3, 2, 2, 2)
    noise = torch.randn_like(clean)
    sigma = torch.tensor([0.7]).view(1, 1, 1, 1, 1)
    noisy = (1.0 - sigma) * clean + sigma * noise
    velocity = noise - clean
    batch = SimpleNamespace(
        noisy_model_input=noisy.permute(0, 2, 1, 3, 4),
        sigmas=sigma,
    )

    reconstructed = predicted_clean_video(velocity, batch)

    torch.testing.assert_close(reconstructed, clean)


def test_temporal_delta_error_is_zero_for_matching_motion_and_masks_invalid_transition():
    clean = torch.zeros(1, 3, 1, 1, 1)
    clean[:, 1] = 1.0
    clean[:, 2] = 5.0
    prediction = clean.clone()
    prediction[:, 2] = 100.0
    valid = torch.tensor([[True, True, False]])

    error = temporal_delta_error(prediction, clean, valid)

    torch.testing.assert_close(error, torch.tensor(0.0))


def test_temporal_delta_error_rejects_bad_mask_shape():
    video = torch.zeros(1, 3, 1, 1, 1)
    with pytest.raises(ValueError, match="validity mask"):
        temporal_delta_error(video, video, torch.ones(1, 2, dtype=torch.bool))


def test_lowpass_direction_loss_rejects_reverse_motion():
    checkerboard = torch.tensor([[1.0, -1.0], [-1.0, 1.0]]).repeat(1, 2, 1, 1, 1)
    torch.testing.assert_close(lowpass_video(checkerboard, 2), torch.zeros(1, 2, 1, 1, 1))
    target = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    valid = torch.ones(1, 2, dtype=torch.bool)
    torch.testing.assert_close(
        masked_motion_direction_loss(target, target, valid),
        torch.tensor(0.0),
        atol=1.0e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(
        masked_motion_direction_loss(-target, target, valid),
        torch.tensor(2.0),
        atol=1.0e-6,
        rtol=0.0,
    )


def test_worldmodel_loss_adds_ranking_margin_and_restores_real_camera():
    method = MiniMaxH3WorldModelFineTuneMethod.__new__(MiniMaxH3WorldModelFineTuneMethod)
    method.temporal_delta_weight = 0.25
    method.temporal_delta_warmup_steps = 10
    method.camera_ranking_weight = 0.1
    method.camera_ranking_margin_ratio = 0.03
    method.camera_ranking_every_microbatches = 4
    method.camera_counterfactual_variants = ("static",)
    method.camera_counterfactual_same_input = True
    method.camera_static_paired_target = False
    method.precondition_outputs = False
    method._worldmodel_microbatch = 3
    method._attn_kind = "dense"
    prediction = torch.zeros(1, 3, 1, 1, 1)
    method.student = _StaticPredictionStudent(prediction)
    clean = torch.tensor([0.0, 1.0, 2.0]).view(1, 3, 1, 1, 1)
    camera = torch.eye(4).repeat(1, 3, 1, 1)
    camera[:, 1:, 0, 3] = torch.tensor([1.0, 2.0])
    batch = SimpleNamespace(
        latents=clean,
        noisy_model_input=torch.zeros(1, 1, 3, 1, 1),
        noise=torch.zeros(1, 1, 3, 1, 1),
        sigmas=torch.ones(1, 1, 1, 1, 1),
        timesteps=torch.ones(1),
        input_kwargs={
            "action_mask": torch.ones(1, 3, dtype=torch.bool),
            "camera_to_world": camera,
        },
    )

    losses = method.augment_loss_map(
        {"total_loss": torch.tensor(1.0), "finetune_loss": torch.tensor(5.0 / 3.0)},
        prediction,
        batch,
        iteration=10,
    )

    assert losses["temporal_delta_loss"] > 0
    assert losses["camera_ranking_loss"] > 0
    torch.testing.assert_close(losses["camera_static_flow_loss"], torch.tensor(5.0 / 3.0))
    assert losses["total_loss"] > 1.0
    torch.testing.assert_close(batch.input_kwargs["camera_to_world"], camera)


def test_worldmodel_latent_motion_loss_backpropagates_through_parameter_free_observer():
    method = MiniMaxH3WorldModelFineTuneMethod.__new__(MiniMaxH3WorldModelFineTuneMethod)
    method.temporal_delta_weight = 0.0
    method.temporal_delta_warmup_steps = 0
    method.camera_ranking_weight = 0.0
    method.camera_ranking_margin_ratio = 0.0
    method.camera_ranking_every_microbatches = 1
    method.camera_counterfactual_variants = ("static", "reverse")
    method.latent_motion_weight = 1.0
    method.static_latent_motion_weight = 1.0
    method.latent_motion_max_shift = 1
    method.latent_motion_temperature = 0.1
    method.precondition_outputs = False
    method._worldmodel_microbatch = 0
    method._attn_kind = "dense"
    prediction = torch.randn(1, 3, 1, 4, 4, requires_grad=True)
    method.student = _StaticPredictionStudent(prediction)
    camera = torch.eye(4).repeat(1, 3, 1, 1)
    camera[:, 1:, 0, 3] = torch.tensor([1.0, 2.0])
    actions = torch.zeros(1, 3, 6)
    actions[:, 1:] = 1.0
    batch = SimpleNamespace(
        latents=torch.randn_like(prediction),
        noisy_model_input=torch.zeros(1, 1, 3, 4, 4),
        noise=torch.zeros(1, 1, 3, 4, 4),
        sigmas=torch.ones(1, 1, 1, 1, 1),
        timesteps=torch.ones(1),
        input_kwargs={
            "actions": actions,
            "action_mask": torch.ones(1, 3, dtype=torch.bool),
            "camera_to_world": camera,
        },
    )

    losses = method.augment_loss_map(
        {"total_loss": prediction.sum() * 0.0, "finetune_loss": prediction.sum() * 0.0},
        prediction,
        batch,
        iteration=1,
    )
    losses["total_loss"].backward()

    assert losses["latent_motion_loss"] > 0
    assert prediction.grad is not None and bool((prediction.grad != 0).any())
    torch.testing.assert_close(batch.input_kwargs["camera_to_world"], camera)


def test_worldmodel_counterfactual_alternates_static_and_reverse_targets():
    method = MiniMaxH3WorldModelFineTuneMethod.__new__(MiniMaxH3WorldModelFineTuneMethod)
    method.temporal_delta_weight = 0.0
    method.temporal_delta_warmup_steps = 0
    method.camera_ranking_weight = 0.0
    method.camera_ranking_margin_ratio = 0.0
    method.camera_ranking_every_microbatches = 1
    method.camera_counterfactual_variants = ("static", "reverse")
    method.latent_motion_weight = 1.0
    method.static_latent_motion_weight = 1.0
    method.latent_motion_max_shift = 1
    method.latent_motion_temperature = 0.1
    method.precondition_outputs = False
    method._worldmodel_microbatch = 0
    method._attn_kind = "dense"
    prediction = torch.randn(1, 3, 1, 4, 4, requires_grad=True)
    method.student = _StaticPredictionStudent(prediction)
    camera = torch.eye(4).repeat(1, 3, 1, 1)
    camera[:, 1:, 0, 3] = torch.tensor([1.0, 2.0])
    batch = SimpleNamespace(
        latents=torch.randn_like(prediction),
        noisy_model_input=torch.zeros(1, 1, 3, 4, 4),
        noise=torch.zeros(1, 1, 3, 4, 4),
        sigmas=torch.ones(1, 1, 1, 1, 1),
        timesteps=torch.ones(1),
        input_kwargs={
            "action_mask": torch.ones(1, 3, dtype=torch.bool),
            "camera_to_world": camera,
        },
    )
    base_losses = {
        "total_loss": prediction.sum() * 0.0,
        "finetune_loss": prediction.sum() * 0.0,
    }

    first = method.augment_loss_map(dict(base_losses), prediction, batch, iteration=1)
    second = method.augment_loss_map(dict(base_losses), prediction, batch, iteration=2)

    torch.testing.assert_close(method.student.cameras[0], camera[:, :1].expand_as(camera))
    torch.testing.assert_close(method.student.cameras[1][:, 0], camera[:, 0])
    torch.testing.assert_close(method.student.cameras[1][:, 1:, 0, 3], -camera[:, 1:, 0, 3])
    assert first["camera_static_latent_motion_loss"] > 0
    assert first["camera_reverse_latent_motion_loss"] == 0
    assert second["camera_static_latent_motion_loss"] == 0
    assert second["camera_reverse_latent_motion_loss"] > 0
    torch.testing.assert_close(batch.input_kwargs["camera_to_world"], camera)


def test_worldmodel_static_counterfactual_reconstructs_repeated_first_frame():
    method = MiniMaxH3WorldModelFineTuneMethod.__new__(MiniMaxH3WorldModelFineTuneMethod)
    method.temporal_delta_weight = 0.0
    method.temporal_delta_warmup_steps = 0
    method.camera_ranking_weight = 0.0
    method.camera_ranking_margin_ratio = 0.0
    method.camera_ranking_every_microbatches = 1
    method.camera_counterfactual_variants = ("static",)
    method.camera_static_reconstruction_weight = 2.0
    method.latent_motion_weight = 0.0
    method.static_latent_motion_weight = 1.0
    method.precondition_outputs = False
    method._worldmodel_microbatch = 0
    method._attn_kind = "dense"
    clean = torch.tensor([1.0, 2.0, 3.0]).view(1, 3, 1, 1, 1)
    noise = torch.tensor([4.0, 5.0, 6.0]).view(1, 1, 3, 1, 1)
    static_target = noise.permute(0, 2, 1, 3, 4) - clean[:, :1].expand_as(clean)
    prediction = torch.zeros_like(clean, requires_grad=True)
    method.student = _StaticPredictionStudent(prediction)
    camera = torch.eye(4).repeat(1, 3, 1, 1)
    camera[:, 1:, 0, 3] = torch.tensor([1.0, 2.0])
    batch = SimpleNamespace(
        latents=clean,
        noisy_model_input=noise,
        noise=noise,
        sigmas=torch.ones(1, 1, 1, 1, 1),
        timesteps=torch.ones(1),
        input_kwargs={
            "action_mask": torch.ones(1, 3, dtype=torch.bool),
            "camera_to_world": camera,
        },
    )

    losses = method.augment_loss_map(
        {"total_loss": prediction.sum() * 0.0, "finetune_loss": prediction.sum() * 0.0},
        prediction,
        batch,
        iteration=1,
    )

    torch.testing.assert_close(losses["camera_static_reconstruction_loss"], static_target.square().mean())
    torch.testing.assert_close(losses["total_loss"], 2.0 * static_target.square().mean())


def test_worldmodel_static_reconstruction_supports_sampled_sigma():
    method = MiniMaxH3WorldModelFineTuneMethod.__new__(MiniMaxH3WorldModelFineTuneMethod)
    method.temporal_delta_weight = method.camera_ranking_weight = method.latent_motion_weight = 0.0
    method.temporal_delta_warmup_steps = 0
    method.camera_ranking_margin_ratio = 0.0
    method.camera_ranking_every_microbatches = 1
    method.camera_counterfactual_variants = ("static",)
    method.camera_static_reconstruction_weight = 1.0
    method.static_latent_motion_weight = 1.0
    method.precondition_outputs = False
    method._worldmodel_microbatch = 0
    method._attn_kind = "dense"
    prediction = torch.zeros(1, 2, 1, 1, 1)
    method.student = _StaticPredictionStudent(prediction)
    camera = torch.eye(4).repeat(1, 2, 1, 1)
    clean = torch.tensor([1.0, 2.0]).view(1, 2, 1, 1, 1)
    noise = torch.tensor([3.0, 4.0]).view(1, 1, 2, 1, 1)
    sigma = torch.full((1, 1, 1, 1, 1), 0.5)
    noisy = (1.0 - sigma) * clean.permute(0, 2, 1, 3, 4) + sigma * noise
    expected = (noise.permute(0, 2, 1, 3, 4) - clean[:, :1].expand_as(clean)).square().mean()
    batch = SimpleNamespace(
        latents=clean,
        noisy_model_input=noisy,
        noise=noise,
        sigmas=sigma,
        timesteps=torch.ones(1),
        input_kwargs={"action_mask": torch.ones(1, 2, dtype=torch.bool), "camera_to_world": camera},
    )
    losses = method.augment_loss_map(
        {"total_loss": prediction.sum(), "finetune_loss": prediction.sum()},
        prediction,
        batch,
        iteration=1,
    )
    torch.testing.assert_close(losses["camera_static_reconstruction_loss"], expected)


def test_same_input_counterfactual_penalizes_action_invariant_prediction():
    method = MiniMaxH3WorldModelFineTuneMethod.__new__(MiniMaxH3WorldModelFineTuneMethod)
    method.temporal_delta_weight = 0.0
    method.temporal_delta_warmup_steps = 0
    method.camera_ranking_weight = 1.0
    method.camera_ranking_margin_ratio = 0.1
    method.camera_ranking_every_microbatches = 1
    method.camera_counterfactual_variants = ("reverse",)
    method.camera_counterfactual_same_input = True
    method.camera_sensitivity_weight = 1.0
    method.camera_response_range_weight = 0.0
    method.camera_sensitivity_min_scale = 0.1
    method.camera_sensitivity_max_scale = 2.0
    method.camera_static_reconstruction_weight = 0.0
    method.latent_motion_weight = 0.0
    method.latent_motion_direction_weight = 0.0
    method.static_latent_motion_weight = 1.0
    method.precondition_outputs = False
    method._worldmodel_microbatch = 0
    method._attn_kind = "dense"

    prediction = torch.zeros(1, 3, 1, 1, 1, requires_grad=True)
    method.student = _StaticPredictionStudent(prediction)
    camera = torch.eye(4).repeat(1, 3, 1, 1)
    camera[:, 1:, 0, 3] = torch.tensor([1.0, 2.0])
    clean = torch.zeros_like(prediction)
    noisy = torch.ones(1, 1, 3, 1, 1)
    batch = SimpleNamespace(
        latents=clean,
        noisy_model_input=noisy,
        noise=noisy,
        sigmas=torch.ones(1, 1, 1, 1, 1),
        timesteps=torch.ones(1),
        input_kwargs={
            "action_mask": torch.ones(1, 3, dtype=torch.bool),
            "camera_to_world": camera,
        },
    )

    losses = method.augment_loss_map(
        {"total_loss": prediction.sum() * 0.0, "finetune_loss": torch.tensor(1.0)},
        prediction,
        batch,
        iteration=1,
    )

    torch.testing.assert_close(method.student.noisy_inputs[0], noisy.permute(0, 2, 1, 3, 4))
    assert losses["camera_ranking_loss"] > 0.0
    assert losses["camera_sensitivity_loss"] > 0.0
    torch.testing.assert_close(losses["camera_response_rms"], torch.tensor(0.0))
    assert losses["total_loss"] > 0.0
    torch.testing.assert_close(batch.input_kwargs["camera_to_world"], camera)
    losses["total_loss"].backward()
    assert prediction.grad is not None
    assert bool(torch.isfinite(prediction.grad).all())


@pytest.mark.parametrize("variant", ("reverse", "shuffled"))
def test_zero_action_delta_skips_counterfactual_losses_with_finite_backward(variant: str):
    method = MiniMaxH3WorldModelFineTuneMethod.__new__(MiniMaxH3WorldModelFineTuneMethod)
    method.temporal_delta_weight = 0.0
    method.temporal_delta_warmup_steps = 0
    method.camera_ranking_weight = 1.0
    method.camera_ranking_margin_ratio = 0.1
    method.camera_ranking_every_microbatches = 1
    method.camera_counterfactual_variants = (variant,)
    method.camera_counterfactual_same_input = True
    method.camera_static_paired_target = True
    method.camera_sensitivity_weight = 1.0
    method.camera_response_range_weight = 1.0
    method.camera_sensitivity_min_scale = 0.1
    method.camera_sensitivity_max_scale = 2.0
    method.camera_response_action_epsilon = 1.0e-6
    method.camera_static_reconstruction_weight = 0.0
    method.latent_motion_weight = 0.0
    method.latent_motion_direction_weight = 0.0
    method.static_latent_motion_weight = 1.0
    method.precondition_outputs = False
    method.fail_on_nonfinite_loss = True
    method._worldmodel_microbatch = 0
    method._attn_kind = "dense"

    prediction = torch.zeros(1, 3, 1, 1, 1, requires_grad=True)
    method.student = _StaticPredictionStudent(prediction)
    camera = torch.eye(4).repeat(1, 3, 1, 1)
    clean = torch.zeros_like(prediction)
    noisy = torch.ones(1, 1, 3, 1, 1)
    batch = SimpleNamespace(
        latents=clean,
        noisy_model_input=noisy,
        noise=noisy,
        sigmas=torch.ones(1, 1, 1, 1, 1),
        timesteps=torch.ones(1),
        input_kwargs={
            "action_mask": torch.ones(1, 3, dtype=torch.bool),
            "camera_to_world": camera,
        },
    )
    finetune_loss = (prediction - 1.0).square().mean()

    losses = method.augment_loss_map(
        {"total_loss": finetune_loss, "finetune_loss": finetune_loss},
        prediction,
        batch,
        iteration=1,
    )
    losses["total_loss"].backward()

    torch.testing.assert_close(losses["camera_response_rms"], torch.tensor(0.0))
    torch.testing.assert_close(losses["camera_ranking_loss"], torch.tensor(0.0))
    torch.testing.assert_close(losses["camera_sensitivity_loss"], torch.tensor(0.0))
    torch.testing.assert_close(losses["camera_response_range_loss"], torch.tensor(0.0))
    assert prediction.grad is not None
    assert bool(torch.isfinite(prediction.grad).all())


def test_worldmodel_rejects_nonfinite_total_loss_before_backward():
    method = MiniMaxH3WorldModelFineTuneMethod.__new__(MiniMaxH3WorldModelFineTuneMethod)
    method.temporal_delta_weight = 0.0
    method.temporal_delta_warmup_steps = 0
    method.camera_ranking_weight = 0.0
    method.camera_ranking_every_microbatches = 1
    method.camera_counterfactual_variants = ("reverse",)
    method.precondition_outputs = False
    method.fail_on_nonfinite_loss = True
    method._worldmodel_microbatch = 0
    prediction = torch.zeros(1, 2, 1, 1, 1)
    batch = SimpleNamespace(
        latents=torch.zeros_like(prediction),
        noisy_model_input=torch.zeros(1, 1, 2, 1, 1),
        sigmas=torch.ones(1, 1, 1, 1, 1),
        input_kwargs={
            "action_mask": torch.ones(1, 2, dtype=torch.bool),
        },
    )

    with pytest.raises(FloatingPointError, match="before backward.*total_loss"):
        method.augment_loss_map(
            {"total_loss": torch.tensor(float("nan")), "finetune_loss": torch.tensor(0.0)},
            prediction,
            batch,
            iteration=1,
        )


def test_mixed_counterfactual_policy_pairs_only_static_and_ranks_wrong_actions():
    method = MiniMaxH3WorldModelFineTuneMethod.__new__(MiniMaxH3WorldModelFineTuneMethod)
    method.temporal_delta_weight = 0.0
    method.temporal_delta_warmup_steps = 0
    method.camera_ranking_weight = 1.0
    method.camera_ranking_margin_ratio = 0.1
    method.camera_ranking_every_microbatches = 1
    method.camera_counterfactual_variants = ("static", "reverse", "shuffled")
    method.camera_counterfactual_same_input = True
    method.camera_static_paired_target = True
    method.camera_sensitivity_weight = 1.0
    method.camera_response_range_weight = 0.0
    method.camera_sensitivity_min_scale = 0.1
    method.camera_sensitivity_max_scale = 2.0
    method.camera_static_reconstruction_weight = 1.0
    method.latent_motion_weight = 0.0
    method.latent_motion_direction_weight = 0.0
    method.static_latent_motion_weight = 1.0
    method.precondition_outputs = False
    method._worldmodel_microbatch = 0
    method._attn_kind = "dense"

    prediction = torch.zeros(1, 4, 1, 1, 1, requires_grad=True)
    method.student = _StaticPredictionStudent(prediction)
    camera = torch.eye(4).repeat(1, 4, 1, 1)
    camera[:, 1:, 0, 3] = torch.tensor([1.0, 2.0, 4.0])
    clean = torch.tensor([1.0, 2.0, 3.0, 5.0]).view_as(prediction)
    noise = clean.permute(0, 2, 1, 3, 4)
    sigma = torch.full((1, 1, 1, 1, 1), 0.5)
    noisy = clean.permute(0, 2, 1, 3, 4)
    batch = SimpleNamespace(
        latents=clean,
        noisy_model_input=noisy,
        noise=noise,
        sigmas=sigma,
        timesteps=torch.ones(1),
        input_kwargs={
            "action_mask": torch.ones(1, 4, dtype=torch.bool),
            "camera_to_world": camera,
        },
    )
    base_losses = {"total_loss": prediction.sum() * 0.0, "finetune_loss": torch.tensor(1.0)}

    static = method.augment_loss_map(dict(base_losses), prediction, batch, iteration=1)
    reverse = method.augment_loss_map(dict(base_losses), prediction, batch, iteration=2)
    shuffled = method.augment_loss_map(dict(base_losses), prediction, batch, iteration=3)

    static_latents = clean[:, :1].expand_as(clean).permute(0, 2, 1, 3, 4)
    expected_static_noisy = (1.0 - sigma) * static_latents + sigma * noise
    torch.testing.assert_close(method.student.noisy_inputs[0], expected_static_noisy.permute(0, 2, 1, 3, 4))
    torch.testing.assert_close(method.student.noisy_inputs[1], noisy.permute(0, 2, 1, 3, 4))
    torch.testing.assert_close(method.student.noisy_inputs[2], noisy.permute(0, 2, 1, 3, 4))
    torch.testing.assert_close(static["camera_ranking_loss"], torch.tensor(0.0))
    torch.testing.assert_close(static["camera_sensitivity_loss"], torch.tensor(0.0))
    assert static["camera_static_reconstruction_loss"] > 0.0
    assert reverse["camera_ranking_loss"] > 0.0
    assert reverse["camera_sensitivity_loss"] > 0.0
    assert shuffled["camera_ranking_loss"] > 0.0
    assert shuffled["camera_sensitivity_loss"] > 0.0
    torch.testing.assert_close(batch.input_kwargs["camera_to_world"], camera)


def test_motion_balanced_indices_interleave_action_quartiles(tmp_path: Path):
    actions = []
    masks = []
    for magnitude in range(8):
        action = np.zeros((3, 6), dtype=np.float32)
        action[:, 0] = float(magnitude)
        mask = np.ones(3, dtype=np.float32)
        actions.append(action)
        masks.append(mask)
    table = pa.table(
        {
            "action_continuous_bytes": [action.tobytes() for action in actions],
            "action_continuous_shape": [list(action.shape) for action in actions],
            "action_valid_mask_bytes": [mask.tobytes() for mask in masks],
            "action_valid_mask_shape": [list(mask.shape) for mask in masks],
        }
    )
    pq.write_table(table, tmp_path / "data.parquet")

    indices = build_motion_balanced_indices(
        str(tmp_path),
        action_channel_scales=(1.0,) * 6,
        seed=7,
        num_buckets=4,
    )

    assert sorted(indices) == list(range(8))
    for offset in range(0, len(indices), 4):
        assert [index // 2 for index in indices[offset : offset + 4]] == [0, 1, 2, 3]
