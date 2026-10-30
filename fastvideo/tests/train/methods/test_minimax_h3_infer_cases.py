# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts.evaluation.infer_minimax_h3_worldmodel_checkpoint import _apply_camera_ablation, _read_case_manifest


def test_single_action_case_manifest_assigns_one_distinct_frame_per_group(tmp_path: Path) -> None:
    cases = [
        {"case_id": f"case_{index}", "data_path": str(tmp_path), "row_index": index,
         "trajectory": "J" if index % 2 == 0 else "L"}
        for index in range(4)
    ]
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(cases), encoding="utf-8")
    assert _read_case_manifest(path, 4) == cases

    cases[1]["row_index"] = 0
    path.write_text(json.dumps(cases), encoding="utf-8")
    with pytest.raises(ValueError, match="One first frame"):
        _read_case_manifest(path, 4)


def test_single_action_case_manifest_rejects_partial_eight_gpu_plan(tmp_path: Path) -> None:
    path = tmp_path / "cases.json"
    path.write_text(json.dumps([{"case_id": "only_one", "data_path": str(tmp_path),
                                "row_index": 0, "trajectory": "J"}]), encoding="utf-8")
    with pytest.raises(ValueError, match="4 entries"):
        _read_case_manifest(path, 4)


def test_camera_path_ablation_keeps_adapter_input_but_freezes_prope_geometry() -> None:
    camera = torch.eye(4).repeat(1, 3, 1, 1)
    camera[:, 1, 0, 3] = 0.5
    camera[:, 2, 0, 3] = 1.0
    intrinsics = torch.eye(3).repeat(1, 3, 1, 1)
    intrinsics[:, 1, 0, 0] = 1.5

    model = SimpleNamespace(
        transformer=SimpleNamespace(action_adapter=SimpleNamespace(weight=torch.nn.Parameter(torch.ones(2, 2)))),
        _packed_prope_matrices=lambda _layout, c, k: (c, k),
    )
    _apply_camera_ablation(model, "prope")
    projection, returned_intrinsics = model._packed_prope_matrices(None, camera, intrinsics)
    torch.testing.assert_close(projection, camera[:, :1].expand_as(camera))
    torch.testing.assert_close(returned_intrinsics, intrinsics[:, :1].expand_as(intrinsics))
    torch.testing.assert_close(model.transformer.action_adapter.weight, torch.ones(2, 2))

    _apply_camera_ablation(model, "adapter")
    torch.testing.assert_close(model.transformer.action_adapter.weight, torch.zeros(2, 2))
    _apply_camera_ablation(model, "both")
    projection, _ = model._packed_prope_matrices(None, camera, intrinsics)
    torch.testing.assert_close(projection, camera[:, :1].expand_as(camera))
    with pytest.raises(ValueError, match="Unknown camera ablation"):
        _apply_camera_ablation(model, "unknown")
