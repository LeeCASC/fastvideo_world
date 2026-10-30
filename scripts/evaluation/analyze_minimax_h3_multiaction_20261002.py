#!/usr/bin/env python3
"""Summarize eight-direction rollouts with deliberately non-decisive motion proxies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch


KEYS = "WASDIJKL"
EXPECTED = {"W": ("radial", 1), "S": ("radial", -1),
            "A": ("horizontal", 1), "D": ("horizontal", -1),
            "I": ("vertical", 1), "K": ("vertical", -1),
            "J": ("horizontal", 1), "L": ("horizontal", -1)}


def motion_proxy(path: Path) -> dict[str, float]:
    samples = []
    previous = None
    yy, xx = np.mgrid[0:120, 0:208]
    radial_x = (xx - 103.5).astype(np.float32)
    radial_y = (yy - 59.5).astype(np.float32)
    radius = np.sqrt(radial_x**2 + radial_y**2).clip(min=1.0)
    with imageio.get_reader(path, format="ffmpeg") as reader:
        for index, frame in enumerate(reader):
            if index % 4:
                continue
            gray = cv2.cvtColor(cv2.resize(frame, (208, 120)), cv2.COLOR_RGB2GRAY)
            if previous is not None:
                flow = cv2.calcOpticalFlowFarneback(previous, gray, None, 0.5, 3, 21, 3, 5, 1.2, 0)
                center = flow[12:-12, 20:-20]
                outward = (flow[..., 0] * radial_x + flow[..., 1] * radial_y) / radius
                samples.append((np.median(center[..., 0]), np.median(center[..., 1]),
                                np.median(outward[12:-12, 20:-20])))
            previous = gray
    if not samples:
        raise ValueError(f"no optical-flow samples: {path}")
    medians = np.median(np.asarray(samples), axis=0)
    return {"horizontal": float(medians[0]), "vertical": float(medians[1]), "radial": float(medians[2])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--step", type=int, default=300)
    parser.add_argument("--rows", type=int, nargs="+", default=[11, 23])
    args = parser.parse_args()
    cases = []
    for row in args.rows:
        base = args.root / f"validation_row_{row}"
        measurements = {}
        latents = {}
        for key in KEYS:
            group = "wasd" if key in "WASD" else "ijkl"
            directory = base / group / f"checkpoint-{args.step}"
            measurements[key] = motion_proxy(directory / f"{key}.mp4")
            latents[key] = torch.load(directory / f"{key}.pt", map_location="cpu", weights_only=True).float()
        signed = {}
        for key, (axis, sign) in EXPECTED.items():
            value = measurements[key][axis]
            signed[key] = {"axis": axis, "expected_sign": sign, "measured": value,
                           "proxy_sign_matches": value * sign > 0.05}
        pair_difference = {}
        for first, second in (("W", "S"), ("A", "D"), ("I", "K"), ("J", "L")):
            rms = float((latents[first] - latents[second]).square().mean().sqrt())
            reference = float((latents[first].square().mean() + latents[second].square().mean()).sqrt() / 2**0.5)
            pair_difference[f"{first}/{second}"] = rms / max(reference, 1.0e-8)
        cases.append({"validation_row": row, "motion_proxy_pixels_per_four_frames": measurements,
                      "expected_sign_check": signed, "opposite_latent_rms_fraction": pair_difference,
                      "montage": str(base / "WASD_IJKL_montage.mp4")})
    report = {"status": "proxy_only_not_a_control_verdict", "cases": cases,
              "caveat": "Median optical flow can be dominated by moving objects, changes in content or depth. "
                        "Latent differences show response, not correct camera direction. Human video review is required."}
    (args.root / "motion_proxy.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [f"# Step-{args.step} eight-direction evaluation", "",
             "This report gives motion proxies, **not** a verified action-following conclusion.", ""]
    for case in cases:
        lines.append(f"## Held-out row {case['validation_row']}")
        lines.append("")
        lines.append(f"Montage: `{case['montage']}` (top W/A/S/D; bottom I/J/K/L).")
        lines.append("")
        for key in KEYS:
            item = case["expected_sign_check"][key]
            lines.append(f"- {key}: {item['axis']} {item['measured']:+.3f} px / 4 frames; "
                         f"proxy sign {'matches' if item['proxy_sign_matches'] else 'does not match'}")
        lines.append(f"- Opposite-command latent RMS fractions: {case['opposite_latent_rms_fraction']}")
        lines.append("")
    lines.append("Inspect both montage videos and the individual MP4s before judging control or visual quality.")
    (args.root / "analysis.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {args.root / 'analysis.md'}", flush=True)


if __name__ == "__main__":
    main()
