#!/usr/bin/env python3
"""Verify and summarize a held-out MiniMax-H3 single-action camera sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from summarize_minimax_h3_action_sweep import ACTIONS, COMPONENT, contact_sheet, measure_video, row_ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--data-path", required=True, type=Path)
    parser.add_argument("--rows", required=True, type=int, nargs="+")
    parser.add_argument("--checkpoint-step", required=True, type=int)
    args = parser.parse_args()

    root = args.root.resolve()
    ids = row_ids(args.data_path)
    results = []
    lines = [
        f"# Checkpoint-{args.checkpoint_step} held-out single-action camera sweep",
        "",
        "Each 124-frame video uses one constant numeric camera trajectory. All eight",
        "actions for a given first frame share the same seed, first-frame condition,",
        "fixed first-frame intrinsics, trajectory scale, and 50 requested sampler steps.",
        "No action text or audio is used. Optical flow is a rough image-motion",
        "proxy, not proof that the camera followed the command.",
        "",
        "| Scene | Validation row | Source ID | Contact sheet |",
        "|---:|---:|---|---|",
    ]
    for scene, row in enumerate(args.rows):
        directory = root / f"row_{row:02d}" / f"checkpoint-{args.checkpoint_step}"
        frames_by_action = {}
        record = {"scene": scene, "validation_row": row, "sample_id": ids[row], "actions": {}}
        for action in ACTIONS:
            video = directory / f"{action}.mp4"
            metrics, sampled = measure_video(video)
            component, sign = COMPONENT[action]
            metrics["expected_component"] = component
            metrics["signed_response_proxy"] = round(sign * metrics[component], 4)
            metrics["video"] = str(video)
            record["actions"][action] = metrics
            frames_by_action[action] = sampled
        sheet = contact_sheet(root, scene, frames_by_action)
        record["contact_sheet"] = str(sheet)
        results.append(record)
        lines.append(f"| {scene} | {row} | `{ids[row]}` | [{sheet.name}]({sheet.name}) |")

    lines.extend([
        "", "## Signed image-motion proxy (pixels/frame at 208×120)", "",
        "Positive agrees with the expected image-motion component; negative is opposite.",
        "Dynamic subjects, occlusions, and visual artifacts can invalidate this sign test.",
        "", "| Scene | W | A | S | D | I | J | K | L |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for record in results:
        values = [f"{record['actions'][action]['signed_response_proxy']:+.3f}" for action in ACTIONS]
        lines.append(f"| {record['scene']} | " + " | ".join(values) + " |")
    lines.append("")

    (root / "metrics.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    (root / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Verified {len(results) * len(ACTIONS)} videos; summary={root / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
