# SPDX-License-Identifier: Apache-2.0
"""Unit contracts for H3 world-model loss validation."""

from types import SimpleNamespace

import pytest
import torch

from fastvideo.pipelines import TrainingBatch
from fastvideo.train.callbacks.worldmodel_validation import (
    WorldModelLossValidationCallback,
    _video_flow_loss,
)


def test_worldmodel_validation_matches_noise_minus_clean_target():
    clean = torch.ones((1, 2, 3, 1, 1))
    noise = torch.full((1, 3, 2, 1, 1), 3.0)
    batch = TrainingBatch(
        latents=clean,
        noise=noise,
        noisy_model_input=torch.zeros((1, 3, 2, 1, 1)),
        sigmas=torch.ones((1, 1, 1, 1, 1)),
    )

    assert _video_flow_loss(torch.full_like(clean, 2.0), batch, precondition_outputs=False) == 0.0


def test_worldmodel_validation_cadence(monkeypatch: pytest.MonkeyPatch):
    callback = WorldModelLossValidationCallback(data_path="/tmp/validation", every_steps=10, run_at_start=False)
    callback.dataloader = object()
    calls: list[int] = []
    monkeypatch.setattr(callback, "_run_validation", lambda _method, iteration: calls.append(iteration))
    method = SimpleNamespace()

    callback.on_validation_begin(method, iteration=0)
    callback.on_validation_begin(method, iteration=9)
    callback.on_validation_begin(method, iteration=10)

    assert calls == [10]


def test_worldmodel_validation_logs_paired_static_target(monkeypatch: pytest.MonkeyPatch):
    clean = torch.tensor([1.0, 2.0]).view(1, 2, 1, 1, 1)
    noise = clean.permute(0, 2, 1, 3, 4)
    prepared = TrainingBatch(
        latents=clean,
        noise=noise,
        noisy_model_input=noise.clone(),
        sigmas=torch.full((1, 1, 1, 1, 1), 0.5),
        timesteps=torch.ones(1),
        input_kwargs={"camera_to_world": torch.eye(4).repeat(1, 2, 1, 1)},
    )

    class Student:
        device = torch.device("cpu")
        transformer = torch.nn.Identity()

        def prepare_batch(self, *_args, **_kwargs):
            return prepared

        def predict_noise(self, noisy, *_args, **_kwargs):
            return torch.zeros_like(noisy)

    class Tracker:
        def __init__(self):
            self.records = []

        def log(self, metrics, iteration):
            self.records.append((metrics, iteration))

    class WorldGroup:
        rank = 0
        world_size = 1

        @staticmethod
        def all_reduce(metrics):
            return metrics

    callback = WorldModelLossValidationCallback(data_path="/tmp/validation", num_batches=1)
    callback.training_config = SimpleNamespace(model=SimpleNamespace(precondition_outputs=False))
    callback.camera_conditioning = "hybrid"
    callback.dataloader = [{}]
    tracker = Tracker()
    method = SimpleNamespace(
        student=Student(),
        tracker=tracker,
        camera_static_paired_target=True,
    )
    monkeypatch.setattr(
        "fastvideo.train.callbacks.worldmodel_validation.get_world_group",
        lambda: WorldGroup(),
    )

    callback._run_validation(method, iteration=7)

    metrics, iteration = tracker.records[0]
    assert iteration == 7
    torch.testing.assert_close(
        torch.tensor(metrics["validation_worldmodel/paired_static_hybrid_flow_loss"]),
        torch.tensor(0.5),
    )
