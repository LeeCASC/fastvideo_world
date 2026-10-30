# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for fixed-camera counterfactual rollout trajectories."""

import torch

from fastvideo.models.dits.minimax_h3_prope import make_camera_trajectory_variant


def _translation_path() -> torch.Tensor:
    camera = torch.eye(4).repeat(1, 4, 1, 1)
    camera[0, :, 0, 3] = torch.arange(4, dtype=torch.float32)
    return camera


def test_real_camera_static_and_reverse_preserve_first_pose():
    camera = _translation_path()

    static = make_camera_trajectory_variant(camera, "static", seed=3)
    reverse = make_camera_trajectory_variant(camera, "reverse", seed=3)

    torch.testing.assert_close(static, camera[:, :1].expand_as(camera))
    torch.testing.assert_close(reverse[:, 0], camera[:, 0])
    torch.testing.assert_close(reverse[0, :, 0, 3], -camera[0, :, 0, 3])


def test_real_camera_shuffle_is_deterministic_and_keeps_first_pose():
    camera = _translation_path()
    shuffled_a = make_camera_trajectory_variant(camera, "shuffled", seed=7)
    shuffled_b = make_camera_trajectory_variant(camera, "shuffled", seed=7)

    torch.testing.assert_close(shuffled_a, shuffled_b)
    torch.testing.assert_close(shuffled_a[:, 0], camera[:, 0])
    assert sorted(shuffled_a[0, 1:, 0, 3].tolist()) == [1.0, 2.0, 3.0]
