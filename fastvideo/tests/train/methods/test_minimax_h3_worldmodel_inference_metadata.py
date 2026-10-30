# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest
import torch


_INFERENCE = runpy.run_path(
    str(
        Path(__file__).resolve().parents[4]
        / "scripts/evaluation/infer_minimax_h3_worldmodel_checkpoint.py"
    )
)
write_merged_metadata = _INFERENCE["_write_merged_metadata"]
make_camera_trajectory = _INFERENCE["make_camera_trajectory"]
scale_prope_gates = _INFERENCE["_scale_prope_gates"]
apply_camera_ablation = _INFERENCE["_apply_camera_ablation"]


def _metadata(*, seed: int, trajectory: str, variant: str) -> dict:
    return {
        "checkpoint": "/checkpoint-300",
        "validation_data": "/validation",
        "row_index": 0,
        "sample": {"id": "sample"},
        "seed": seed,
        "num_inference_steps_requested": 50,
        "num_transformer_steps": 49,
        "trajectories": [trajectory],
        "real_trajectory_gate_scales": None,
        "real_trajectory_variants": [variant],
        "audio": False,
    }


def test_rollout_metadata_merges_trajectory_batches(tmp_path: Path) -> None:
    path = tmp_path / "metadata.json"
    write_merged_metadata(path, _metadata(seed=7, trajectory="real_correct", variant="correct"))
    write_merged_metadata(path, _metadata(seed=7, trajectory="real_static", variant="static"))

    merged = json.loads(path.read_text(encoding="utf-8"))
    assert merged["trajectories"] == ["real_correct", "real_static"]
    assert merged["real_trajectory_variants"] == ["correct", "static"]


def test_rollout_metadata_rejects_incompatible_batches(tmp_path: Path) -> None:
    path = tmp_path / "metadata.json"
    write_merged_metadata(path, _metadata(seed=7, trajectory="real_correct", variant="correct"))

    with pytest.raises(RuntimeError, match="seed"):
        write_merged_metadata(path, _metadata(seed=8, trajectory="real_static", variant="static"))


@pytest.mark.parametrize("action,axis,sign", [("W", 2, 1), ("S", 2, -1), ("A", 0, -1), ("D", 0, 1)])
def test_action_trajectory_translation_scale(action: str, axis: int, sign: int) -> None:
    slow = make_camera_trajectory(action, 37, scale=0.5)
    fast = make_camera_trajectory(action, 37, scale=2.0)
    assert slow[-1, axis, 3].item() == pytest.approx(sign * 0.5)
    assert fast[-1, axis, 3].item() == pytest.approx(sign * 2.0)


def test_action_trajectory_pitch_and_yaw_directions() -> None:
    trajectories = {action: make_camera_trajectory(action, 37) for action in "IJKL"}
    assert trajectories["I"][-1, 1, 2] > 0  # forward axis rises
    assert trajectories["K"][-1, 1, 2] < 0
    assert trajectories["J"][-1, 0, 2] < 0  # forward axis points left
    assert trajectories["L"][-1, 0, 2] > 0


def test_residual_prope_inference_scale_and_ablation() -> None:
    camera_output = torch.nn.Parameter(torch.ones(2, 2))
    transformer = SimpleNamespace(
        config=SimpleNamespace(arch_config=SimpleNamespace(prope_mode="residual")),
        named_parameters=lambda: iter((("transformer_blocks.1.attn.prope_o.weight", camera_output),)),
    )
    model = SimpleNamespace(transformer=transformer)
    assert scale_prope_gates(model, 2.0) == pytest.approx(2.0)
    torch.testing.assert_close(camera_output, torch.full((2, 2), 2.0))
    apply_camera_ablation(model, "prope")
    assert torch.count_nonzero(camera_output) == 0
