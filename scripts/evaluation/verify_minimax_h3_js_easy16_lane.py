# SPDX-License-Identifier: Apache-2.0
"""Fail closed unless every requested JS-easy rollout is complete and aligned."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


def verify_video(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(path)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Could not decode video: {path}")
    count = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame.shape != (480, 832, 3):
                raise ValueError(f"Unexpected video frame size: {path}: {frame.shape}")
            count += 1
    finally:
        capture.release()
    if count != 124:
        raise ValueError(f"Expected 124 decoded frames, got {count}: {path}")


def verify_case(case: dict, video_root: Path, *, require_flow_loss: bool) -> None:
    directory = video_root / case["case_id"]
    path = directory / f"{case['trajectory']}.mp4"
    verify_video(path)
    camera = np.load(directory / f"{case['trajectory']}_camera_to_world.npy")
    if camera.shape != (1, 37, 4, 4) or not np.isfinite(camera).all():
        raise ValueError(f"Invalid saved camera path: {directory}")
    if not np.allclose(camera[0, 0], np.eye(4), atol=1e-4):
        raise ValueError(f"Camera path is not first-frame-relative: {directory}")
    if require_flow_loss:
        data = json.loads((directory / "flow_loss.json").read_text(encoding="utf-8"))
        for key in ("flow_loss_real_camera", "flow_loss_static_camera", "camera_prediction_rms_response"):
            if not math.isfinite(float(data[key])):
                raise ValueError(f"Invalid GT flow metric {key}: {directory}")
    print(f"verified {case['case_id']} {case['trajectory']}: 124 frames", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--manifest", type=Path)
    group.add_argument("--suite-root", type=Path)
    parser.add_argument("--video-root", type=Path)
    parser.add_argument("--require-flow-loss", choices=("yes", "no"), default="no")
    parser.add_argument("--checkpoint-label", default="checkpoint-1200")
    args = parser.parse_args()
    if args.manifest:
        if args.video_root is None:
            parser.error("--video-root is required with --manifest")
        cases = json.loads(args.manifest.read_text(encoding="utf-8"))
        if len(cases) != 4:
            raise ValueError("Expected four cases per eight-GPU lane")
        for case in cases:
            verify_case(case, args.video_root, require_flow_loss=args.require_flow_loss == "yes")
    else:
        root = args.suite_root
        for mode in ("gt", "single"):
            for index in range(4):
                manifest = root / "manifests" / f"{mode}_group_{index}.json"
                cases = json.loads(manifest.read_text(encoding="utf-8"))
                if len(cases) != 4:
                    raise ValueError(f"Expected four cases: {manifest}")
                for case in cases:
                    verify_case(case, root / "videos" / mode / args.checkpoint_label,
                                require_flow_loss=mode == "gt")
        print("Verified all 32 requested videos and 16 GT flow-loss files", flush=True)


if __name__ == "__main__":
    main()
