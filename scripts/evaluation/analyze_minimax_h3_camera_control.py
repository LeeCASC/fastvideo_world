# SPDX-License-Identifier: Apache-2.0
"""Quantify signed camera control in MiniMax-H3 counterfactual rollouts.

The evaluator expects one directory containing ``ground_truth.mp4`` plus
``real_{correct,static,reverse,shuffled}.mp4``. It estimates robust global image
motion with tracked corners and an affine RANSAC model, compares every generated
trajectory against ground truth, writes machine-readable acceptance results,
and produces a compact first/middle/last-frame montage for visual review.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROLLOUT_NAMES = ("ground_truth", "real_correct", "real_static", "real_reverse", "real_shuffled")


def _read_video(path: Path) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        capture.release()
    if len(frames) < 2:
        raise ValueError(f"Camera-control analysis needs at least two frames: {path}")
    return frames


def _global_motion(previous: np.ndarray, current: np.ndarray) -> tuple[np.ndarray, float, int]:
    """Estimate previous-to-current translation and affine inlier ratio."""
    previous_gray = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    scale = min(1.0, 480.0 / float(max(previous_gray.shape)))
    if scale < 1.0:
        size = (round(previous_gray.shape[1] * scale), round(previous_gray.shape[0] * scale))
        previous_gray = cv2.resize(previous_gray, size, interpolation=cv2.INTER_AREA)
        current_gray = cv2.resize(current_gray, size, interpolation=cv2.INTER_AREA)
    corners = cv2.goodFeaturesToTrack(
        previous_gray,
        maxCorners=800,
        qualityLevel=0.01,
        minDistance=5,
        blockSize=5,
    )
    if corners is None or len(corners) < 8:
        return np.zeros(2, dtype=np.float64), 0.0, 0
    tracked, status, _ = cv2.calcOpticalFlowPyrLK(
        previous_gray,
        current_gray,
        corners,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if tracked is None or status is None:
        return np.zeros(2, dtype=np.float64), 0.0, 0
    valid = status.reshape(-1).astype(bool)
    source = corners.reshape(-1, 2)[valid]
    target = tracked.reshape(-1, 2)[valid]
    if len(source) < 8:
        return np.zeros(2, dtype=np.float64), 0.0, int(len(source))
    affine, inliers = cv2.estimateAffinePartial2D(
        source,
        target,
        method=cv2.RANSAC,
        ransacReprojThreshold=2.0,
        maxIters=2000,
        confidence=0.995,
        refineIters=10,
    )
    if affine is None or inliers is None:
        displacement = np.median(target - source, axis=0) / scale
        return displacement.astype(np.float64), 0.0, int(len(source))
    displacement = affine[:, 2] / scale
    return displacement.astype(np.float64), float(inliers.mean()), int(len(source))


def estimate_camera_trajectory(frames: list[np.ndarray]) -> dict[str, Any]:
    motions: list[np.ndarray] = []
    inlier_ratios: list[float] = []
    tracked_points: list[int] = []
    for previous, current in zip(frames[:-1], frames[1:], strict=True):
        motion, inlier_ratio, point_count = _global_motion(previous, current)
        motions.append(motion)
        inlier_ratios.append(inlier_ratio)
        tracked_points.append(point_count)
    transition_motion = np.stack(motions)
    cumulative = np.concatenate((np.zeros((1, 2)), np.cumsum(transition_motion, axis=0)), axis=0)
    return {
        "transition_motion": transition_motion,
        "cumulative_motion": cumulative,
        "endpoint": cumulative[-1],
        "path_length": float(np.linalg.norm(transition_motion, axis=1).sum()),
        "mean_inlier_ratio": float(np.mean(inlier_ratios)),
        "mean_tracked_points": float(np.mean(tracked_points)),
    }


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return 0.0 if denominator < 1.0e-8 else float(np.dot(left, right) / denominator)


def compare_trajectory(candidate: dict[str, Any], ground_truth: dict[str, Any]) -> dict[str, float]:
    candidate_motion = candidate["transition_motion"]
    target_motion = ground_truth["transition_motion"]
    count = min(len(candidate_motion), len(target_motion))
    candidate_motion = candidate_motion[:count]
    target_motion = target_motion[:count]
    target_endpoint = target_motion.sum(axis=0)
    candidate_endpoint = candidate_motion.sum(axis=0)
    target_energy = float(np.square(target_motion).sum())
    temporal_projection = (
        0.0 if target_energy < 1.0e-8 else float((candidate_motion * target_motion).sum() / target_energy)
    )
    target_norm = float(np.linalg.norm(target_endpoint))
    projection_ratio = (
        0.0
        if target_norm < 1.0e-8
        else float(np.dot(candidate_endpoint, target_endpoint) / (target_norm * target_norm))
    )
    return {
        "endpoint_x": float(candidate_endpoint[0]),
        "endpoint_y": float(candidate_endpoint[1]),
        "endpoint_norm": float(np.linalg.norm(candidate_endpoint)),
        "path_length": float(candidate["path_length"]),
        "endpoint_direction_cosine": _cosine(candidate_endpoint, target_endpoint),
        "negative_endpoint_direction_cosine": _cosine(candidate_endpoint, -target_endpoint),
        "endpoint_projection_ratio": projection_ratio,
        "temporal_projection_ratio": temporal_projection,
        "mean_inlier_ratio": float(candidate["mean_inlier_ratio"]),
        "mean_tracked_points": float(candidate["mean_tracked_points"]),
    }


def acceptance_metrics(metrics: dict[str, dict[str, float]]) -> dict[str, Any]:
    gt_norm = max(metrics["ground_truth"]["endpoint_norm"], 1.0e-6)
    correct = metrics["real_correct"]
    static = metrics["real_static"]
    reverse = metrics["real_reverse"]
    shuffled = metrics["real_shuffled"]
    checks = {
        "correct_direction": correct["endpoint_direction_cosine"] >= 0.5,
        "correct_magnitude": correct["endpoint_projection_ratio"] >= 0.2,
        "correct_temporal_order": correct["temporal_projection_ratio"] >= 0.1,
        "static_suppression": static["endpoint_norm"] / gt_norm <= 0.35,
        "reverse_direction": reverse["negative_endpoint_direction_cosine"] >= 0.5,
        "reverse_magnitude": reverse["endpoint_projection_ratio"] <= -0.15,
        "shuffle_degrades_order": (
            correct["temporal_projection_ratio"] - shuffled["temporal_projection_ratio"] >= 0.05
        ),
        "coherent_tracking": min(
            correct["mean_inlier_ratio"],
            static["mean_inlier_ratio"],
            reverse["mean_inlier_ratio"],
        ) >= 0.5,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "static_to_gt_endpoint_ratio": static["endpoint_norm"] / gt_norm,
        "passed_count": sum(checks.values()),
        "total_count": len(checks),
    }


def _write_montage(path: Path, videos: dict[str, list[np.ndarray]]) -> None:
    cells: list[np.ndarray] = []
    for name in ROLLOUT_NAMES:
        frames = videos[name]
        indices = (0, len(frames) // 2, len(frames) - 1)
        selected = []
        for index in indices:
            frame = cv2.resize(frames[index], (416, 240), interpolation=cv2.INTER_AREA)
            cv2.putText(
                frame,
                f"{name} frame={index}",
                (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
            selected.append(frame)
        cells.append(np.concatenate(selected, axis=0))
    cv2.imwrite(str(path), np.concatenate(cells, axis=1))


def analyze(input_dir: Path) -> dict[str, Any]:
    videos = {name: _read_video(input_dir / f"{name}.mp4") for name in ROLLOUT_NAMES}
    trajectories = {name: estimate_camera_trajectory(frames) for name, frames in videos.items()}
    ground_truth = trajectories["ground_truth"]
    metrics = {name: compare_trajectory(trajectory, ground_truth) for name, trajectory in trajectories.items()}
    result = {
        "input_dir": str(input_dir.resolve()),
        "metrics": metrics,
        "acceptance": acceptance_metrics(metrics),
    }
    (input_dir / "analysis_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    _write_montage(input_dir / "rollout_montage.jpg", videos)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    args = parser.parse_args()
    result = analyze(args.input_dir)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["acceptance"]["passed"] else 2)


if __name__ == "__main__":
    main()
