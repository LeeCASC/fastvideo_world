#!/usr/bin/env python3
"""CPU-only video/pose audit for the frozen eight-direction H3 diagnostic.

This is a quantitative screen, not a substitute for human video review. In
particular moving foreground objects can dominate optical flow; report their
effect instead of silently relabeling the measured camera trajectory.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics

import cv2
import numpy as np


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def video_metrics(path: Path) -> dict:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open source video: {path}")
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if count < 110:
        raise ValueError(f"Source video has only {count} frames: {path}")
    samples = []
    for index in (0, 5, 30, 35, 60, 65, 90, 95, 118, 123):
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok:
            raise ValueError(f"Cannot decode frame {index} from {path}")
        gray = cv2.cvtColor(cv2.resize(frame, (320, 180)), cv2.COLOR_BGR2GRAY)
        samples.append(gray)
    capture.release()
    flows = []
    for first, second in zip(samples[::2], samples[1::2]):
        flow = cv2.calcOpticalFlowFarneback(first, second, None, .5, 3, 19, 3, 5, 1.2, 0)
        # Median suppresses a small moving foreground; divergence is useful
        # for forward/back motion where the median horizontal flow is ~zero.
        x = np.arange(flow.shape[1], dtype=np.float32)[None, :] - (flow.shape[1] - 1) / 2
        y = np.arange(flow.shape[0], dtype=np.float32)[:, None] - (flow.shape[0] - 1) / 2
        flows.append((
            float(np.median(flow[..., 0])),
            float(np.median(flow[..., 1])),
            float(np.mean(flow[..., 0] * x / 160 + flow[..., 1] * y / 90)),
            float(np.median(np.linalg.norm(flow, axis=-1))),
        ))
    blur = [float(cv2.Laplacian(frame, cv2.CV_64F).var()) for frame in samples[::2]]
    result = np.median(np.asarray(flows), axis=0)
    return {
        "frames": count,
        "flow_median_x": float(result[0]),
        "flow_median_y": float(result[1]),
        "flow_expansion": float(result[2]),
        "flow_magnitude": float(result[3]),
        "blur_laplacian_min": min(blur),
        "blur_laplacian_median": float(np.median(blur)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ids", type=Path, required=True)
    parser.add_argument("--source-mapping", type=Path, required=True)
    parser.add_argument("--records-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    source = {row["global_id"]: row for row in read_jsonl(args.source_mapping)}
    identities = list(read_jsonl(args.ids))
    if args.limit:
        identities = identities[:args.limit]
    results = []
    failures = []
    for number, identity in enumerate(identities, 1):
        global_id = identity["global_id"]
        row = source[global_id]
        record = args.records_root / row["record_path"]
        try:
            measured = video_metrics(record / "video.mp4")
            with np.load(record / "camera_c2w.npz", allow_pickle=False) as camera_file:
                key = "camera_to_world" if "camera_to_world" in camera_file else "c2w"
                camera = np.asarray(camera_file[key], dtype=np.float64)
            if camera.ndim != 3 or camera.shape[1:] != (4, 4) or not np.isfinite(camera).all():
                raise ValueError(f"Invalid source camera path: {camera.shape}")
            if camera.shape[0] not in (37, 124):
                raise ValueError(f"Unexpected source camera length: {camera.shape[0]}")
            measured["camera_translation_x"] = float(camera[-1, 0, 3] - camera[0, 0, 3])
            measured["camera_translation_z"] = float(camera[-1, 2, 3] - camera[0, 2, 3])
            measured["camera_yaw_proxy"] = float(camera[-1, 0, 2] - camera[0, 0, 2])
            measured["camera_pitch_proxy"] = float(camera[-1, 2, 1] - camera[0, 2, 1])
            measured.update({
                "id": identity["id"], "global_id": global_id,
                "direction": identity["direction"], "scene": identity["scene"],
                "speed": identity["speed"], "gpu_qc_pass": row.get("gpu_qc_pass"),
            })
            results.append(measured)
        except Exception as error:
            failures.append({"id": identity["id"], "error": str(error)})
        if number % 20 == 0:
            print(f"Audited {number}/{len(identities)} videos; failures={len(failures)}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    (args.output.parent / "video_pose_metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in results)
    )
    by_direction = defaultdict(list)
    for row in results:
        by_direction[row["direction"]].append(row)
    expected_flow = {
        "W": ("flow_expansion", 1), "S": ("flow_expansion", -1),
        "A": ("flow_median_x", 1), "D": ("flow_median_x", -1),
        "I": ("flow_median_y", 1), "K": ("flow_median_y", -1),
        "J": ("flow_median_x", 1), "L": ("flow_median_x", -1),
    }
    review = []
    for row in results:
        metric, sign = expected_flow[row["direction"]]
        directional_flow = row[metric] * sign
        if directional_flow < 0.1:
            review.append({
                "id": row["id"], "global_id": row["global_id"],
                "direction": row["direction"], "signed_flow": directional_flow,
                "reason": "weak/opposite coarse optical flow; manual review required, not automatic rejection",
            })
    (args.output.parent / "needs_manual_review.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in review)
    )
    directions = {}
    for direction, rows in by_direction.items():
        directions[direction] = {
            "count": len(rows),
            "median_horizontal_flow": statistics.median(row["flow_median_x"] for row in rows),
            "median_vertical_flow": statistics.median(row["flow_median_y"] for row in rows),
            "median_expansion": statistics.median(row["flow_expansion"] for row in rows),
            "median_blur_score": statistics.median(row["blur_laplacian_median"] for row in rows),
        }
    report = {
        "requested": len(identities), "audited": len(results), "failed": len(failures),
        "source_video_frame_counts": dict(Counter(row["frames"] for row in results)),
        "directions": directions,
        "needs_manual_review": len(review),
        "failures": failures,
        "note": "Optical flow is a screen only; moving objects and depth can obscure camera direction. Review flagged pairs before expanding beyond diagnostic training.",
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
