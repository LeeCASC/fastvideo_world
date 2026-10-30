#!/usr/bin/env python3
"""Materialize comparable PRoPE-only and hybrid overnight training configs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    args = parser.parse_args()
    root = args.data_root
    report = json.loads((root / "report.json").read_text())
    scales = report["action_channel_scales"]
    if report["unique_samples"] < 80 or report["action_clipping_rate"] > 0.02:
        raise SystemExit("Paired data or action range failed preflight")
    configs = root / "configs"
    configs.mkdir(exist_ok=True)
    for branch in ("prope", "hybrid"):
        for phase in ("high_noise", "mixed_noise"):
            cfg = yaml.safe_load(args.base_config.read_text())
            student = cfg["models"]["student"]
            student["camera_conditioning"] = branch
            student["action_representation"] = "relative_pose" if branch == "hybrid" else "incremental"
            student["trainable_modules"] = (["action_adapter", "proj_in", "norm_out", "proj_out"]
                                            if branch == "hybrid" else ["proj_in", "norm_out", "proj_out"])
            student["action_channel_scales"] = [float(value) for value in scales]
            student["action_normalized_clip"] = 5.0
            student["sampling_index_path"] = str(root / "train_indices.json")
            student["minimum_video_noise_sigma"] = 0.6 if phase == "mixed_noise" else 0.0
            if phase == "high_noise":
                student["fixed_video_noise_sigma"] = 1.0
            else:
                student.pop("fixed_video_noise_sigma", None)
                student["init_from_training_checkpoint"] = str(root / "runs" / branch / "high_noise" /
                                                               "checkpoints" / "checkpoint-200")
            method = cfg["method"]
            method["temporal_delta_weight"] = 0.2
            method["camera_ranking_weight"] = 2.0
            method["camera_ranking_margin_ratio"] = 0.05
            method["camera_ranking_every_microbatches"] = 2
            method["camera_counterfactual_variants"] = ["reverse", "static"]
            method["camera_counterfactual_same_input"] = True
            method["camera_static_paired_target"] = False
            method["camera_sensitivity_weight"] = 0.2
            method["camera_response_range_weight"] = 0.02
            method["camera_sensitivity_min_scale"] = 0.03
            method["camera_sensitivity_max_scale"] = 0.5
            dist = cfg["training"]["distributed"]
            dist["num_gpus"] = 4
            dist["sp_size"] = 2
            dist["hsdp_replicate_dim"] = 2
            dist["hsdp_shard_dim"] = 2
            cfg["training"]["data"]["data_path"] = str(root / "data")
            loop = cfg["training"]["loop"]
            loop["max_train_steps"] = 200 if phase == "high_noise" else 500
            loop["gradient_accumulation_steps"] = 2  # 2 DP x 1 local x 2 GA = total BS 4 per job.
            checkpoint = cfg["training"]["checkpoint"]
            checkpoint["output_dir"] = str(root / "runs" / branch / phase / "checkpoints")
            checkpoint["training_state_checkpointing_steps"] = 25
            checkpoint["checkpoints_total_limit"] = 12
            cfg["training"]["tracker"]["run_name"] = f"h3_camera_night_{branch}_{phase}"
            cfg["callbacks"]["worldmodel_validation"]["every_steps"] = 50
            cfg["callbacks"]["worldmodel_validation"]["num_batches"] = 4
            cfg["callbacks"]["worldmodel_validation"]["run_at_start"] = phase == "high_noise"
            cfg["pipeline"]["dit_config"]["action_dim"] = 6 if branch == "hybrid" else 0
            path = configs / f"{branch}_{phase}.yaml"
            if path.exists():
                raise SystemExit(f"Refusing to overwrite config: {path}")
            path.write_text(yaml.safe_dump(cfg, sort_keys=False))
            print(path)


if __name__ == "__main__":
    main()
