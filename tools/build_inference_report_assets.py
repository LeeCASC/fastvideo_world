#!/usr/bin/env python3
"""Create compact frame sheets for the H3/FastVideoWM inference report."""

from __future__ import annotations

import re
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont


OUT = Path(
    "/home/lizihao/projects/military_world_model_agent/FastVideoWM/docs/training/"
    "assets/inference_review_zh_20261014"
)


GROUPS = {
    "h3_baseline_vs_ft2000_js00001_W": [
        ("H3 baseline step-10000", "/datacc05/lizihao/outputs/h3_world/js_easy17_wasdijkl_baseline_vs_ft2000_20260917/baseline_50steps/videos/js_00001_W.mp4"),
        ("H3 fine-tune step-2000", "/datacc05/lizihao/outputs/h3_world/js_easy17_wasdijkl_baseline_vs_ft2000_20260917/finetune2000_50steps/videos/js_00001_W.mp4"),
    ],
    "cross_model_latest_js00001_W": [
        ("H3 latest step-1000", "/datacc05/lizihao/outputs/h3_world/js_easy17_h3_latest_vs_fastvideowm_latest_20261009/h3_world/videos/js_00001_W.mp4"),
        ("FastVideoWM latest step-400", "/datacc05/lizihao/outputs/h3_world/js_easy17_h3_latest_vs_fastvideowm_latest_20261009/fastvideowm/videos/js_00001/wasd/checkpoint-400/W.mp4"),
    ],
    "fastvideowm_step750_vs_1200_vs_1500_js00001_W": [
        ("FastVideoWM step-750", "/datacc05/lizihao/worldmodel/fastvideowm_full_65389_20260927/inference_step750_single_action_20260930/row_00/checkpoint-750/W.mp4"),
        ("FastVideoWM step-1200", "/datacc05/lizihao/worldmodel/fastvideowm_js_easy16_step1200_20261012/videos/single/checkpoint-1200/js_00001/W.mp4"),
        ("FastVideoWM step-1500", "/datacc05/lizihao/worldmodel/fastvideowm_js_easy16_step1500_20261013/videos/single/checkpoint-1500/js_00001/W.mp4"),
    ],
    "fastvideowm_step200_row11_all8": [
        ("W", "/datacc05/lizihao/worldmodel/fastvideowm_simple_camera_20261002/evaluation/step200_heldout_50steps/validation_row_11/wasd/checkpoint-200/W.mp4"),
        ("A", "/datacc05/lizihao/worldmodel/fastvideowm_simple_camera_20261002/evaluation/step200_heldout_50steps/validation_row_11/wasd/checkpoint-200/A.mp4"),
        ("I", "/datacc05/lizihao/worldmodel/fastvideowm_simple_camera_20261002/evaluation/step200_heldout_50steps/validation_row_11/ijkl/checkpoint-200/I.mp4"),
        ("J", "/datacc05/lizihao/worldmodel/fastvideowm_simple_camera_20261002/evaluation/step200_heldout_50steps/validation_row_11/ijkl/checkpoint-200/J.mp4"),
    ],
    "fastvideowm_step300_row11_all8": [
        ("W", "/datacc05/lizihao/worldmodel/fastvideowm_multiaction_20261002/evaluation/step300_heldout_50steps/validation_row_11/wasd/checkpoint-300/W.mp4"),
        ("A", "/datacc05/lizihao/worldmodel/fastvideowm_multiaction_20261002/evaluation/step300_heldout_50steps/validation_row_11/wasd/checkpoint-300/A.mp4"),
        ("I", "/datacc05/lizihao/worldmodel/fastvideowm_multiaction_20261002/evaluation/step300_heldout_50steps/validation_row_11/ijkl/checkpoint-300/I.mp4"),
        ("J", "/datacc05/lizihao/worldmodel/fastvideowm_multiaction_20261002/evaluation/step300_heldout_50steps/validation_row_11/ijkl/checkpoint-300/J.mp4"),
    ],
}


def read_samples(path: Path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames in {path}")
    return [frames[i] for i in (0, len(frames) // 2, len(frames) - 1)], len(frames)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    records = {}
    for name, entries in GROUPS.items():
        cell_w, cell_h, label_h = 360, 240, 42
        canvas = Image.new("RGB", (cell_w * 3, (cell_h + label_h) * len(entries)), "white")
        draw = ImageDraw.Draw(canvas)
        group_records = []
        for row, (label, raw_path) in enumerate(entries):
            path = Path(raw_path)
            frames, count = read_samples(path)
            draw.text((8, row * (cell_h + label_h) + 5), label, fill="black", font=font)
            for col, frame in enumerate(frames):
                image = Image.fromarray(frame)
                image.thumbnail((cell_w - 8, cell_h - 8))
                x = col * cell_w + (cell_w - image.width) // 2
                y = row * (cell_h + label_h) + label_h + (cell_h - image.height) // 2
                canvas.paste(image, (x, y))
            group_records.append({"label": label, "path": raw_path, "decoded_frames": count})
        output = OUT / f"{re.sub(r'[^A-Za-z0-9_.-]+', '_', name)}.jpg"
        canvas.save(output, quality=92)
        records[name] = {"sheet": str(output), "videos": group_records}
    (OUT / "manifest.json").write_text(__import__("json").dumps(records, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {len(records)} sheets to {OUT}")


if __name__ == "__main__":
    main()
