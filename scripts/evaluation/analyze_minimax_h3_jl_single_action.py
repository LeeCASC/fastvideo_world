#!/usr/bin/env python3
"""Write a coarse, non-decisive horizontal-motion proxy for J/L rollouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np


def analyze_video(path: Path) -> dict[str, float]:
    horizontal: list[float] = []
    vertical: list[float] = []
    with imageio.get_reader(path, format="ffmpeg") as reader:
        previous = None
        for frame_index, frame in enumerate(reader):
            if frame_index % 4:
                continue
            gray = cv2.cvtColor(cv2.resize(frame, (208, 120)), cv2.COLOR_RGB2GRAY)
            if previous is not None:
                flow = cv2.calcOpticalFlowFarneback(previous, gray, None, 0.5, 3, 21, 3, 5, 1.2, 0)
                center = flow[12:-12, 20:-20]
                horizontal.append(float(np.median(center[..., 0])))
                vertical.append(float(np.median(center[..., 1])))
            previous = gray
    if not horizontal:
        raise ValueError(f"No video motion pairs in {path}")
    return {
        "median_horizontal_pixels_per_4_frames_at_208px_width": float(np.median(horizontal)),
        "median_vertical_pixels_per_4_frames_at_120px_height": float(np.median(vertical)),
        "median_absolute_horizontal_pixels": float(np.median(np.abs(horizontal))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    cases = json.loads(args.case_manifest.read_text(encoding="utf-8"))
    results = []
    for case in cases:
        video = args.output_root / case["case_id"] / f"{case['trajectory']}.mp4"
        results.append({
            "case_id": case["case_id"],
            "trajectory": case["trajectory"],
            "video": str(video),
            **analyze_video(video),
        })
    report = {
        "method": "Median central-region Farneback optical flow on RGB video, sampled every four frames",
        "interpretation": "Camera-left J usually shifts static background right (+x); L shifts left (-x). "
        "Foreground motion and generated scene changes can invalidate this proxy. Inspect videos before judging control.",
        "cases": results,
    }
    output = args.output_root.parent / "motion_proxy.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote non-decisive J/L motion proxy: {output}", flush=True)


if __name__ == "__main__":
    main()
