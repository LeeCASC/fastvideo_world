# SPDX-License-Identifier: Apache-2.0
"""Check 5x8 H3 action sweep completeness and make visual review sheets.

Flow metrics are only a rough camera-motion diagnostic, not a substitute for
watching the generated clips. They are measured on the image border to reduce
foreground subject-motion contamination.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw


ACTIONS = "WASDIJKL"
COMPONENT = {
    "W": ("radial", 1), "S": ("radial", -1),
    "A": ("horizontal", 1), "D": ("horizontal", -1),
    "I": ("vertical", 1), "K": ("vertical", -1),
    "J": ("horizontal", 1), "L": ("horizontal", -1),
}


def measure_video(path: Path) -> tuple[dict, list[np.ndarray]]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {path}")
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if count != 124:
        raise RuntimeError(f"Expected 124 frames but got {count}: {path}")
    yy, xx = np.mgrid[:120, :208]
    edge = (xx < 40) | (xx >= 168) | (yy < 22) | (yy >= 98)
    radius_x = (xx - 103.5) / np.maximum(np.hypot(xx - 103.5, yy - 59.5), 1)
    radius_y = (yy - 59.5) / np.maximum(np.hypot(xx - 103.5, yy - 59.5), 1)
    metrics: list[tuple[float, float, float]] = []
    blur: list[float] = []
    sampled: list[np.ndarray] = []
    previous = None
    for index in range(count):
        ok, bgr = capture.read()
        if not ok:
            raise RuntimeError(f"Failed decoding frame {index}: {path}")
        if index in (0, 62, 123):
            sampled.append(cv2.cvtColor(cv2.resize(bgr, (416, 240)), cv2.COLOR_BGR2RGB))
        gray = cv2.cvtColor(cv2.resize(bgr, (208, 120)), cv2.COLOR_BGR2GRAY)
        if index % 3 == 0:
            blur.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
        if previous is not None and index % 3 == 0:
            flow = cv2.calcOpticalFlowFarneback(
                previous, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0,
            )
            dx, dy = flow[..., 0], flow[..., 1]
            metrics.append((
                float(np.median(dx[edge])),
                float(np.median(dy[edge])),
                float(np.median((dx * radius_x + dy * radius_y)[edge])),
            ))
        previous = gray
    capture.release()
    if len(sampled) != 3 or not metrics:
        raise RuntimeError(f"Incomplete measurement: {path}")
    median = np.median(np.asarray(metrics), axis=0)
    return {
        "frames": count,
        "horizontal": round(float(median[0]), 4),
        "vertical": round(float(median[1]), 4),
        "radial": round(float(median[2]), 4),
        "median_laplacian_variance": round(float(np.median(blur)), 2),
    }, sampled


def row_ids(data_path: Path) -> list[str]:
    ids: list[str] = []
    for file in sorted(data_path.rglob("*.parquet")):
        ids.extend(pq.read_table(file, columns=["id"]).column("id").to_pylist())
    return ids


def contact_sheet(root: Path, scene: int, frames_by_action: dict[str, list[np.ndarray]]) -> Path:
    cell_w, cell_h = 416, 264
    canvas = Image.new("RGB", (cell_w * 3, cell_h * 8), "white")
    draw = ImageDraw.Draw(canvas)
    for row, action in enumerate(ACTIONS):
        for col, frame in enumerate(frames_by_action[action]):
            canvas.paste(Image.fromarray(frame), (col * cell_w, row * cell_h + 24))
        draw.text((6, row * cell_h + 5), f"{action}   first / middle / last", fill="black")
    path = root / f"scene_{scene:02d}_contact_sheet.jpg"
    canvas.save(path, quality=88)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--rows", type=int, nargs=5, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    first_dir = root / f"scene_00_row_{args.rows[0]:02d}" / "checkpoint-100"
    meta = json.loads((first_dir / "metadata.json").read_text())
    ids = row_ids(Path(meta["validation_data"]))
    records = []
    lines = [
        "# Step-100 action-only pilot: 5 scenes × 8 directions",
        "",
        "All runs use the same checkpoint, first-frame conditioning, seed within each",
        "scene, 124 frames, 50 requested sampler steps, and one constant camera",
        "trajectory per video. Text action descriptions are not injected: this FastVideoWM",
        "pilot uses numeric camera trajectories via PRoPE / the action adapter.",
        "",
        "Optical-flow signs below are *diagnostics*, not proof of action adherence.",
        "Dynamic objects, occlusion, and model artifacts can affect them; review the",
        "contact sheets and videos before deciding whether training works.",
        "",
        "| Scene | Validation row | Sample ID | Contact sheet |",
        "|---:|---:|---|---|",
    ]
    for scene, row in enumerate(args.rows):
        directory = root / f"scene_{scene:02d}_row_{row:02d}" / "checkpoint-100"
        frames_by_action = {}
        scene_record = {"scene": scene, "validation_row": row, "sample_id": ids[row], "actions": {}}
        for action in ACTIONS:
            video = directory / f"{action}.mp4"
            metrics, sampled = measure_video(video)
            component, sign = COMPONENT[action]
            metrics["expected_component"] = component
            metrics["signed_response_proxy"] = round(sign * metrics[component], 4)
            metrics["video"] = str(video)
            scene_record["actions"][action] = metrics
            frames_by_action[action] = sampled
        sheet = contact_sheet(root, scene, frames_by_action)
        scene_record["contact_sheet"] = str(sheet)
        records.append(scene_record)
        lines.append(f"| {scene} | {row} | `{ids[row]}` | [{sheet.name}]({sheet.name}) |")
    lines += [
        "", "## Directional response proxy", "",
        "Positive numbers indicate flow in the expected image-space direction;",
        "negative numbers are opposite. Units are pixels per frame at 208×120.",
        "Near-zero values mean the proxy is inconclusive.", "",
        "| Scene | W | A | S | D | I | J | K | L |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in records:
        values = [f"{record['actions'][a]['signed_response_proxy']:+.3f}" for a in ACTIONS]
        lines.append(f"| {record['scene']} | " + " | ".join(values) + " |")
    lines += ["", "Generated videos are under each `scene_NN_row_NN/checkpoint-100/` directory.", ""]
    (root / "metrics.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    (root / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"verified {len(records) * 8} videos; summary={root / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
