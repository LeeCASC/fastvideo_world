# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import runpy

import numpy as np

_ANALYSIS = runpy.run_path(
    str(Path(__file__).resolve().parents[4] / "scripts/evaluation/analyze_minimax_h3_camera_control.py")
)
acceptance_metrics = _ANALYSIS["acceptance_metrics"]
compare_trajectory = _ANALYSIS["compare_trajectory"]


def _trajectory(dx: float, dy: float, *, wobble: float = 0.0) -> dict:
    motion = np.tile(np.array([[dx, dy]], dtype=np.float64), (12, 1))
    motion[::2, 1] += wobble
    cumulative = np.concatenate((np.zeros((1, 2)), np.cumsum(motion, axis=0)), axis=0)
    return {
        "transition_motion": motion,
        "cumulative_motion": cumulative,
        "endpoint": cumulative[-1],
        "path_length": float(np.linalg.norm(motion, axis=1).sum()),
        "mean_inlier_ratio": 0.9,
        "mean_tracked_points": 100.0,
    }


def test_camera_control_acceptance_requires_signed_counterfactuals():
    ground_truth = _trajectory(2.0, 0.1)
    trajectories = {
        "ground_truth": ground_truth,
        "real_correct": _trajectory(1.5, 0.08),
        "real_static": _trajectory(0.02, 0.0),
        "real_reverse": _trajectory(-1.2, -0.06),
        "real_shuffled": _trajectory(0.1, 0.0, wobble=1.0),
    }
    metrics = {
        name: compare_trajectory(trajectory, ground_truth)
        for name, trajectory in trajectories.items()
    }

    result = acceptance_metrics(metrics)

    assert result["passed"] is True
    assert result["passed_count"] == result["total_count"]


def test_camera_control_acceptance_rejects_action_insensitive_motion():
    ground_truth = _trajectory(2.0, 0.0)
    same_motion = _trajectory(-1.0, 0.0)
    trajectories = {
        "ground_truth": ground_truth,
        "real_correct": same_motion,
        "real_static": same_motion,
        "real_reverse": same_motion,
        "real_shuffled": same_motion,
    }
    metrics = {
        name: compare_trajectory(trajectory, ground_truth)
        for name, trajectory in trajectories.items()
    }

    result = acceptance_metrics(metrics)

    assert result["passed"] is False
    assert result["checks"]["correct_direction"] is False
    assert result["checks"]["static_suppression"] is False
