# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fastvideo.dataset.vwm_h3 import (
    H3VWMDataConfig,
    H3VWMRawDataset,
    canonicalize_source_intrinsics,
    build_local_camera_actions,
    build_normalized_local_camera_actions,
    h3_latent_source_frame_indices,
    normalize_camera_trajectory_translation,
    rebase_camera_to_first_frame,
    resample_source_frame_indices,
    source_balanced_manifest_indices,
    source_limited_manifest_indices,
)
from vwm_datasets import build_se3_c2w_from_pitch_yaw_roll


def _pose(*, x: float = 0.0, y: float = 0.0, z: float = 0.0, yaw: float = 0.0) -> np.ndarray:
    pose = build_se3_c2w_from_pitch_yaw_roll(0.0, yaw, 0.0)
    pose[:3, 3] = np.asarray([x, y, z], dtype=np.float32)
    return pose


def test_h3_latent_source_frame_indices_cover_124_frames() -> None:
    indices = h3_latent_source_frame_indices(124)

    assert indices.shape == (37,)
    assert indices[0] == 0
    assert indices[-1] == 123
    assert np.array_equal(indices[:6], np.asarray([0, 4, 8, 12, 16, 17]))


@pytest.mark.parametrize("source_name", [
    "h3_js_easy_fasth3_t2v_js_1000_124",
    "h3_js_easy_fasth3_t2v_js_1000_long",
    "h3_js_easy_h3_i2v_js_easy_124",
    "h3_js_easy_h3_i2v_js_easy_long",
])
def test_js_easy_pixel_intrinsics_are_normalized_from_vggt_resolution(source_name: str) -> None:
    intrinsic = np.array([[922.0, 0.0, 336.0], [0.0, 924.0, 192.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    normalized = canonicalize_source_intrinsics(
        intrinsic, source_name=source_name, image_width=1344, image_height=768)
    np.testing.assert_allclose(normalized[0], [922.0 / 672.0, 0.0, 0.5])
    np.testing.assert_allclose(normalized[1], [0.0, 924.0 / 384.0, 0.5])
    np.testing.assert_allclose(intrinsic[0], [922.0, 0.0, 336.0])


def test_js_easy_large_but_valid_normalized_focal_length_is_retained() -> None:
    intrinsic = np.array([[4883.0, 0.0, 336.0], [0.0, 5023.0, 192.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    normalized = canonicalize_source_intrinsics(
        intrinsic, source_name="h3_js_easy_h3_i2v_js_easy_124", image_width=1344, image_height=768)
    assert normalized[1, 1] > 10.0


def test_unknown_pixel_intrinsics_fail_closed() -> None:
    intrinsic = np.array([[900.0, 0.0, 336.0], [0.0, 900.0, 192.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="not normalized"):
        canonicalize_source_intrinsics(intrinsic, source_name="unknown", image_width=1344, image_height=768)


def test_resample_source_frame_indices_preserves_long_clip_time() -> None:
    indices = resample_source_frame_indices(
        source_num_frames=81,
        source_fps=15.0,
        target_num_frames=124,
        sample_key="sample",
        seed=7,
    )

    assert indices.shape == (124,)
    assert np.all(indices[1:] >= indices[:-1])
    assert indices.min() >= 0
    assert indices.max() < 81
    assert indices[-1] - indices[0] in {76, 77}


def test_resample_source_frame_indices_stretches_train_v2_context_clip() -> None:
    indices = resample_source_frame_indices(
        source_num_frames=77,
        source_fps=16.0,
        target_num_frames=124,
    )

    assert indices[0] == 0
    assert indices[-1] == 76
    assert len(np.unique(indices)) == 77


def test_resample_source_frame_indices_rejects_short_clip() -> None:
    with pytest.raises(ValueError, match="at least"):
        resample_source_frame_indices(
            source_num_frames=56,
            source_fps=16.0,
            target_num_frames=124,
        )


def test_resample_source_frame_indices_can_explicitly_stretch_short_clip() -> None:
    indices = resample_source_frame_indices(
        source_num_frames=15,
        source_fps=16.0,
        target_num_frames=124,
        min_source_duration_seconds=0.0,
    )

    assert indices.shape == (124,)
    assert indices[0] == 0 and indices[-1] == 14


def test_h3_data_config_rejects_negative_minimum_duration() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        H3VWMDataConfig(manifest_path="train.jsonl", min_source_duration_seconds=-1.0)


def test_build_local_camera_actions_uses_previous_camera_axes() -> None:
    poses = np.stack([
        _pose(),
        _pose(z=2.0),
        _pose(x=3.0, z=2.0),
        _pose(x=3.0, y=4.0, z=2.0),
        _pose(x=3.0, y=4.0, z=2.0, yaw=90.0),
    ])

    actions = build_local_camera_actions(poses)

    assert np.allclose(actions[0], 0.0)
    assert np.allclose(actions[1, :3], [2.0, 0.0, 0.0])
    assert np.allclose(actions[2, :3], [0.0, 3.0, 0.0])
    assert np.allclose(actions[3, :3], [0.0, 0.0, 4.0])
    assert actions[4, 4] == pytest.approx(90.0)


def test_camera_action_normalization_removes_source_units() -> None:
    poses = np.stack([_pose(), _pose(z=20.0), _pose(x=20.0, z=20.0, yaw=90.0)])

    normalized = normalize_camera_trajectory_translation(poses)
    actions = build_normalized_local_camera_actions(poses)

    assert np.linalg.norm(normalized[-1, :3, 3]) == pytest.approx(1.0)
    assert np.allclose(actions[1, :3], [1.0 / np.sqrt(2.0), 0.0, 0.0])
    assert actions[2, 4] == pytest.approx(0.5)


def test_rebase_camera_to_first_frame_preserves_relative_actions() -> None:
    poses = np.stack([
        _pose(x=2.0, y=3.0, z=4.0, yaw=20.0),
        _pose(x=3.0, y=3.5, z=5.0, yaw=25.0),
        _pose(x=4.0, y=4.0, z=6.0, yaw=30.0),
    ])

    rebased = rebase_camera_to_first_frame(poses)

    assert np.allclose(rebased[0], np.eye(4), atol=1e-6)
    assert np.allclose(rebased[1], np.linalg.inv(poses[0]) @ poses[1], atol=1e-6)
    assert np.allclose(build_local_camera_actions(rebased), build_local_camera_actions(poses), atol=1e-6)


def test_source_balanced_manifest_indices_mix_every_finite_prefix() -> None:
    records = [
        SimpleNamespace(source_name=source, sample_id=f"{source}-{index}")
        for source, count in (("large", 6), ("medium", 3), ("small", 1))
        for index in range(count)
    ]

    first = source_balanced_manifest_indices(records, list(range(len(records))), seed=17)
    second = source_balanced_manifest_indices(records, list(range(len(records))), seed=17)
    first_sources = [records[index].source_name for index in first]

    assert first == second
    assert sorted(first) == list(range(len(records)))
    assert first_sources[:3] == ["large", "medium", "small"]
    assert first_sources[3:5] == ["large", "medium"]


def test_source_limited_manifest_indices_caps_only_requested_sources() -> None:
    records = [
        SimpleNamespace(source_name=source, sample_id=f"{source}-{index}")
        for source, count in (("large", 6), ("small", 2))
        for index in range(count)
    ]

    first = source_limited_manifest_indices(records, {"large": 3}, seed=17)
    second = source_limited_manifest_indices(records, {"large": 3}, seed=17)
    selected_sources = [records[index].source_name for index in first]

    assert first == second
    assert selected_sources.count("large") == 3
    assert selected_sources.count("small") == 2


def test_source_limited_manifest_indices_rejects_unknown_or_empty_limits() -> None:
    records = [SimpleNamespace(source_name="known", sample_id="sample")]

    with pytest.raises(ValueError, match="positive"):
        source_limited_manifest_indices(records, {"known": 0}, seed=17)
    with pytest.raises(ValueError, match="unknown"):
        source_limited_manifest_indices(records, {"missing": 1}, seed=17)


@pytest.mark.skipif(
    "FASTVIDEO_VWM_TEST_MANIFEST" not in os.environ,
    reason="set FASTVIDEO_VWM_TEST_MANIFEST to run the mounted-data smoke test",
)
def test_h3_vwm_raw_dataset_mounted_sample() -> None:
    manifest = Path(os.environ["FASTVIDEO_VWM_TEST_MANIFEST"])
    dataset = H3VWMRawDataset(
        H3VWMDataConfig(
            manifest_path=manifest,
            split="all",
            source_names=("context",),
        )
    )

    sample = dataset[0]
    assert sample["video"].shape == (3, 124, 480, 832)
    assert sample["video"].dtype == torch.uint8
    assert sample["action_continuous"].shape == (37, 6)
    assert sample["latent_frame_indices"].shape == (37,)
    assert not any("audio" in key for key in sample)
