# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import numpy as np
import pytest
import torch

from fastvideo.dataset.dataloader.schema import pyarrow_schema_minimax_h3_worldmodel_prope
from fastvideo.pipelines.preprocess.minimax_h3_worldmodel import indexed_legacy_cache


def _row(tmp_path: Path) -> dict:
    camera = np.broadcast_to(np.eye(4, dtype=np.float32), (124, 4, 4)).copy()
    camera[:, 2, 3] = np.linspace(0, 1, 124)
    intrinsics = np.broadcast_to(np.eye(3, dtype=np.float32), (124, 3, 3)).copy()
    intrinsics[:, 0, 0] = 800
    intrinsics[:, 1, 1] = 800
    intrinsics[:, 0, 2] = 336
    intrinsics[:, 1, 2] = 192
    camera_path = tmp_path / "camera.npz"
    intrinsics_path = tmp_path / "intrinsics.npz"
    np.savez(camera_path, camera_to_world=camera)
    np.savez(intrinsics_path, intrinsics=intrinsics)
    return {
        "global_id": "h3_js_easy:test_W_fast", "source_name": "h3_js_easy",
        "prompt": "A camera observes a moving vehicle.",
        "cache_path": str(tmp_path / "cache.pth"), "camera_path": str(camera_path),
        "camera_convention": "canonical_y_up", "translation_scale_xyz": [1.0, 1.0, 2.0],
        "intrinsics_path": str(intrinsics_path), "intrinsics_key": "intrinsics",
        "intrinsics_alignment": "target", "intrinsics_units": "pixel",
        "intrinsics_source_hw": [384, 672], "source_frame_indices": list(range(124)),
        "source_fps": 24.0, "quality_approved": True,
    }


def test_indexed_h3_cache_calibrates_speed_and_converts_camera_intrinsics(tmp_path, monkeypatch):
    row = _row(tmp_path)
    monkeypatch.setattr(
        indexed_legacy_cache, "extract_legacy_h3_cache_payload",
        lambda _path: (
            torch.zeros(24, 37, 30, 52), torch.zeros(24, 1, 30, 52),
            torch.zeros(2, 5120), torch.tensor([0, 1]),
        ),
    )
    record = indexed_legacy_cache.convert_indexed_cache_row(row)
    assert set(record) == set(pyarrow_schema_minimax_h3_worldmodel_prope.names)
    camera = np.frombuffer(record["camera_to_world_bytes"], dtype=np.float32).reshape(37, 4, 4)
    intrinsics = np.frombuffer(record["camera_intrinsics_bytes"], dtype=np.float32).reshape(37, 3, 3)
    actions = np.frombuffer(record["action_continuous_bytes"], dtype=np.float32).reshape(37, 6)
    assert camera[-1, 2, 3] == pytest.approx(2.0)
    assert actions[:, 0].sum() == pytest.approx(2.0)
    assert intrinsics[0, 0, 2] == pytest.approx(0.5, abs=0.01)
    assert intrinsics[0, 1, 2] == pytest.approx(0.5, abs=0.01)


def test_indexed_h3_cache_refuses_unapproved_video(tmp_path):
    row = _row(tmp_path)
    row["quality_approved"] = False
    with pytest.raises(ValueError, match="quality-approved"):
        indexed_legacy_cache.convert_indexed_cache_row(row)
