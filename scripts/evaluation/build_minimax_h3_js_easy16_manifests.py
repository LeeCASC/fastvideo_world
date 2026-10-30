# SPDX-License-Identifier: Apache-2.0
"""Freeze four-lane GT and single-action manifests for JS-easy 00001–00016."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

from scripts.evaluation.prepare_minimax_h3_js_easy16_gt import CASE_IDS


ACTIONS = "WWAASSDDIIJJKKLL"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--first-frame-root", type=Path, required=True)
    parser.add_argument("--gt-root", type=Path, required=True)
    parser.add_argument("--train-ids", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=811)
    args = parser.parse_args()

    train_rows = [json.loads(line) for line in args.train_ids.read_text(encoding="utf-8").splitlines() if line]
    train_ids = {str(row["global_id"]) for row in train_rows}
    train_scenes = {str(row["scene"]) for row in train_rows}
    manifests = args.output_root / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    overlap_rows = []
    cases = []
    for case_id, action in zip(CASE_IDS, ACTIONS, strict=True):
        benchmark_dir = args.benchmark_root / case_id
        metadata = json.loads((benchmark_dir / "sample_001.json").read_text(encoding="utf-8"))
        source_id = str(metadata["meta"]["source_sample_id"])
        image_token = str(metadata["meta"]["source_input_image"]).split("/")[-1].split("_")[0:2]
        image_token = "_".join(image_token)
        exact = sorted(value for value in train_ids if source_id in value)
        potential_scenes = sorted(scene for scene in train_scenes if image_token in scene)
        overlap_rows.append({
            "case_id": case_id,
            "source_sample_id": source_id,
            "base_image_token": image_token,
            "exact_train_id_matches": exact,
            "potential_scene_matches": potential_scenes,
            "held_out_claim_allowed": not exact and not potential_scenes,
            "caveat": "Token-level scene check is not a perceptual image-duplicate search.",
        })
        first_dir = args.first_frame_root / case_id
        gt_dir = args.gt_root / case_id
        for data_dir in (first_dir, gt_dir):
            if not (data_dir / "data_000000.parquet").is_file():
                raise FileNotFoundError(f"Conditioned case missing: {data_dir}")
        comparable_columns = ["first_frame_latent_bytes", "text_embedding_bytes", "camera_intrinsics_bytes"]
        first = pq.read_table(first_dir / "data_000000.parquet", columns=comparable_columns).to_pylist()[0]
        ground_truth = pq.read_table(gt_dir / "data_000000.parquet", columns=comparable_columns).to_pylist()[0]
        if any(first[name] != ground_truth[name] for name in comparable_columns):
            raise ValueError(f"GT and simple conditions disagree on first frame or calibrated K: {case_id}")
        cases.append((case_id, action, first_dir, gt_dir))

    for group_index in range(4):
        group = cases[group_index * 4:(group_index + 1) * 4]
        gt = [
            {"case_id": case_id, "data_path": str(gt_dir), "row_index": 0,
             "trajectory": "real", "seed": args.seed}
            for case_id, _, _, gt_dir in group
        ]
        single = [
            {"case_id": case_id, "data_path": str(first_dir), "row_index": 0,
             "trajectory": action, "seed": args.seed}
            for case_id, action, first_dir, _ in group
        ]
        for name, value in (("gt", gt), ("single", single)):
            (manifests / f"{name}_group_{group_index}.json").write_text(
                json.dumps(value, indent=2) + "\n", encoding="utf-8"
            )

    run_manifest = {
        "checkpoint": str(args.checkpoint.resolve()),
        "cases": list(CASE_IDS),
        "single_action_assignments": dict(zip(CASE_IDS, ACTIONS, strict=True)),
        "seed": args.seed,
        "num_inference_steps": 50,
        "output_frames": 124,
        "output_fps": 24,
        "output_hw": [480, 832],
        "trajectory_scale": 1.0,
        "gt_temporal_alignment": "61-to-124 nearest frame at equal normalized progress",
        "gt_intrinsics": "per-frame pixel K normalized at pose processed_hw then center-cropped",
        "synthetic_intrinsics": "frozen at first latent frame",
        "camera_control": "numeric C2W and normalized K; no action text",
        "potential_train_overlap": overlap_rows,
    }
    manifest_path = args.output_root / "run_manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("checkpoint") != run_manifest["checkpoint"]:
            raise ValueError(f"Refusing to mix checkpoints in one evaluation directory: {manifest_path}")
    manifest_path.write_text(
        json.dumps(run_manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote eight four-case manifests and overlap audit: {manifests}", flush=True)


if __name__ == "__main__":
    main()
