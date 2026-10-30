#!/usr/bin/env python3
"""FastVideoWM-compatible CPU triage for the merged 61-frame registry.

This is intentionally conservative and provisional. It does not run VLM, does
not mutate sources, and does not certify subject deformation/texture quality.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import math
from pathlib import Path

import cv2
import numpy as np

cv2.setNumThreads(1)


def sample_video(row: dict) -> dict:
    cap = cv2.VideoCapture(row["video"])
    if not cap.isOpened():
        return {"error": "video_open_failed"}
    indices = np.asarray(row["source_frame_indices"], dtype=np.int64)
    frames = []
    try:
        for index in np.unique(np.rint(np.linspace(0, len(indices) - 1, min(7, len(indices)))).astype(int)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(indices[index]))
            ok, frame = cap.read()
            if not ok:
                return {"error": "sampled_frame_decode_failed"}
            frames.append(cv2.cvtColor(cv2.resize(frame, (256, 144), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY))
    finally:
        cap.release()
    lap = [float(cv2.Laplacian(x, cv2.CV_32F).var()) for x in frames]
    contrast = [float(x.std()) for x in frames]
    brightness = [float(x.mean()) for x in frames]
    changes = [float(cv2.absdiff(a, b).mean()) for a, b in zip(frames, frames[1:])]
    return {
        "median_laplacian_variance": float(np.median(lap)),
        "median_contrast": float(np.median(contrast)),
        "median_brightness": float(np.median(brightness)),
        "median_sampled_frame_change": float(np.median(changes)) if changes else 0.0,
    }


def pose_metrics(row: dict) -> dict:
    with np.load(row["camera"], allow_pickle=False) as payload:
        key = "camera_to_world" if "camera_to_world" in payload else "c2w"
        c2w = np.asarray(payload[key], dtype=np.float64)
        k = np.asarray(payload["intrinsics"], dtype=np.float64) if "intrinsics" in payload else None
    if c2w.ndim != 3 or c2w.shape[1:] != (4, 4) or len(c2w) < 2 or not np.isfinite(c2w).all():
        raise ValueError("invalid_pose")
    r = c2w[:, :3, :3]
    rigidity = float(np.max(np.abs(r @ r.transpose(0, 2, 1) - np.eye(3))))
    det = float(np.max(np.abs(np.linalg.det(r) - 1.0)))
    if max(rigidity, det) > 0.05:
        raise ValueError("pose_not_rigid")
    t = c2w[:, :3, 3]
    displacement = float(np.linalg.norm(t - t[:1], axis=1).max())
    relative = r[:-1].transpose(0, 2, 1) @ r[1:]
    cosine = np.clip((np.trace(relative, axis1=1, axis2=2) - 1.0) * 0.5, -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine))
    step = np.linalg.norm(np.diff(t, axis=0), axis=1)
    active = step[step > max(float(step.max(initial=0.0)) * 0.05, 1e-10)]
    spike = float(active.max() / max(float(np.median(active)), 1e-10)) if len(active) else 0.0
    static = displacement <= 1e-5 and float(angle.max(initial=0.0)) <= 0.10
    near_static = displacement <= 0.01 and float(angle.max(initial=0.0)) <= 0.50
    return {
        "pose_frames": int(len(c2w)),
        "rigidity_error": rigidity,
        "det_error": det,
        "translation_displacement": displacement,
        "rotation_displacement_deg": float(angle.max(initial=0.0)),
        "rotation_step_max_deg": float(angle.max(initial=0.0)),
        "translation_step_spike_ratio": spike,
        "static_camera": static,
        "near_static_camera": near_static,
        "intrinsics_finite": bool(k is None or np.isfinite(k).all()),
    }


def audit(row: dict) -> dict:
    hard, review = [], []
    try:
        media = sample_video(row)
        if "error" in media:
            hard.append(media["error"])
        else:
            if media["median_brightness"] < 2.0 or media["median_brightness"] > 253.0:
                hard.append("extreme_brightness")
            if media["median_contrast"] < 2.0 and media["median_laplacian_variance"] < 1.0:
                hard.append("near_constant_image")
            if media["median_sampled_frame_change"] < 0.05:
                review.append("sampled_frames_nearly_frozen")
            if media["median_laplacian_variance"] < 8.0:
                review.append("possible_blur_or_low_texture")
    except Exception as exc:
        media = {"error": type(exc).__name__}
        hard.append("media_audit_error")
    try:
        motion = pose_metrics(row)
        if not motion["intrinsics_finite"]:
            hard.append("intrinsics_nonfinite")
        if motion["static_camera"]:
            hard.append("static_camera")
        elif motion["near_static_camera"]:
            review.append("near_static_camera")
        if motion["rotation_step_max_deg"] > 5.0:
            review.append("abrupt_rotation")
        if motion["translation_step_spike_ratio"] > 12.0:
            review.append("translation_speed_spike")
    except Exception as exc:
        motion = {"error": type(exc).__name__}
        hard.append("pose_audit_error")
    status = "hard_reject" if hard else "review" if review else "provisional_pass"
    return {
        "window_id": row["window_id"], "global_id": row["global_id"],
        "status": status, "hard_rejection_reasons": hard, "review_reasons": review,
        "media": media, "trajectory": motion,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()
    rows = [json.loads(x) for x in args.input.open(encoding="utf-8") if x.strip()]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = sum(1 for _ in args.output.open(encoding="utf-8")) if args.output.is_file() else 0
    counts = Counter()
    with args.output.open("a", encoding="utf-8") as handle:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for number, result in enumerate(pool.map(audit, rows[completed:], chunksize=8), completed + 1):
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                counts[result["status"]] += 1
                if number % 1000 == 0 or number == len(rows):
                    handle.flush()
                    print(f"cpu_quality={number}/{len(rows)} session={dict(counts)}", flush=True)
    print(f"complete={args.output}", flush=True)


if __name__ == "__main__":
    main()
