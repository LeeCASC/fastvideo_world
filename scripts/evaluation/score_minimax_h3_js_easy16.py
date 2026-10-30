# SPDX-License-Identifier: Apache-2.0
"""Score the fixed 16-case JS-easy GT and single-action FastVideoWM rollouts."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from fastvideo.eval.metrics.common.psnr.metric import PSNRMetric
from fastvideo.eval.metrics.common.ssim.metric import SSIMMetric
from fastvideo.eval.metrics.vbench.temporal_flickering.metric import TemporalFlickeringMetric
from scripts.evaluation.prepare_minimax_h3_js_easy16_gt import (
    CASE_IDS,
    TARGET_FRAMES,
    TARGET_HW,
    resize_center_crop_rgb,
    source_indices,
)


ACTIONS = "WWAASSDDIIJJKKLL"
OPTIONAL_NO_REFERENCE_VBENCH = {
    "vbench.subject_consistency": ("subject_consistency", "SubjectConsistencyMetric"),
    "vbench.background_consistency": ("background_consistency", "BackgroundConsistencyMetric"),
    "vbench.dynamic_degree": ("dynamic_degree", "DynamicDegreeMetric"),
    "vbench.motion_smoothness": ("motion_smoothness", "MotionSmoothnessMetric"),
    "vbench.imaging_quality": ("imaging_quality", "ImagingQualityMetric"),
    "vbench.aesthetic_quality": ("aesthetic_quality", "AestheticQualityMetric"),
}


def read_video(path: Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {path}")
    frames = []
    try:
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if not frames:
        raise ValueError(f"No decoded frames: {path}")
    return np.stack(frames)


def tensor_video(frames: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2).float().div_(255.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--checkpoint-label", default="checkpoint-1200")
    parser.add_argument("--require-lpips", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    psnr = PSNRMetric(chunk_size=4).to(device)
    ssim = SSIMMetric(chunk_size=2).to(device)
    flicker = TemporalFlickeringMetric().to("cpu")
    lpips = None
    if importlib.util.find_spec("lpips"):
        from fastvideo.eval.metrics.common.lpips.metric import LPIPSMetric

        lpips = LPIPSMetric(net="alex", chunk_size=4).to(device)
        lpips.setup()
    elif args.require_lpips:
        raise RuntimeError("LPIPS package is unavailable in the evaluation environment")

    results = []
    for case_id, action in zip(CASE_IDS, ACTIONS, strict=True):
        gt_path = args.output_root / "videos/gt" / args.checkpoint_label / case_id / "real.mp4"
        single_path = args.output_root / "videos/single" / args.checkpoint_label / case_id / f"{action}.mp4"
        reference_path = args.benchmark_root / case_id / "sample_001_gt.mp4"
        generated = read_video(gt_path)
        simple = read_video(single_path)
        original = read_video(reference_path)
        if original.shape[0] != 61:
            raise ValueError(f"Expected 61 original GT frames: {reference_path}")
        if generated.shape != (TARGET_FRAMES, *TARGET_HW, 3) or simple.shape != generated.shape:
            raise ValueError(f"Expected 124 generated frames at 480x832: {case_id}")
        aligned_gt = np.stack([resize_center_crop_rgb(frame) for frame in original])[source_indices()]
        paired = {"video": tensor_video(generated[1:]), "reference": tensor_video(aligned_gt[1:])}
        simple_sample = {"video": tensor_video(simple)}
        flow_path = args.output_root / "videos/gt" / args.checkpoint_label / case_id / "flow_loss.json"
        if not flow_path.is_file():
            raise FileNotFoundError(f"GT flow loss missing: {flow_path}")
        row = {
            "case_id": case_id,
            "single_action": action,
            "gt_temporal_alignment": "61-to-124 nearest frame at equal normalized progress",
            "conditioning_frame_excluded_from_paired_metrics": True,
            "psnr": psnr.compute(paired).score,
            "ssim": ssim.compute(paired).score,
            "lpips": lpips.compute(paired).score if lpips is not None else None,
            "lpips_missing_reason": None if lpips is not None else "lpips package not installed",
            "vbench_temporal_flickering": flicker.compute(simple_sample).score,
            "flow_validation": json.loads(flow_path.read_text(encoding="utf-8")),
        }
        results.append(row)
        print(f"scored {case_id}: PSNR={row['psnr']:.3f} SSIM={row['ssim']:.4f}", flush=True)

    metric_status = {
        "psnr": "complete",
        "ssim": "complete",
        "lpips": "complete" if lpips is not None else "missing_dependency",
        "vbench.temporal_flickering": "complete",
    }
    # Evaluate one optional backbone at a time so their GPU weights do not
    # accumulate. Missing packages/checkpoints are reported, not fabricated.
    for name, (module_name, class_name) in OPTIONAL_NO_REFERENCE_VBENCH.items():
        metric = None
        try:
            module = importlib.import_module(f"fastvideo.eval.metrics.vbench.{module_name}.metric")
            metric = getattr(module, class_name)()
            missing = [dependency for dependency in metric.dependencies if importlib.util.find_spec(dependency) is None]
            if missing:
                raise ModuleNotFoundError(f"Missing optional dependencies: {', '.join(missing)}")
            metric.to(device if metric.needs_gpu else "cpu")
            metric.setup()
            for row in results:
                case_id = row["case_id"]
                action = row["single_action"]
                video_path = args.output_root / "videos/single" / args.checkpoint_label / case_id / f"{action}.mp4"
                row[name] = metric.compute({"video": tensor_video(read_video(video_path))}).score
            metric_status[name] = "complete"
            print(f"scored optional {name}", flush=True)
        except Exception as error:
            metric_status[name] = f"unavailable: {type(error).__name__}: {error}"
            for row in results:
                row[name] = None
            print(f"skipped optional {name}: {error}", flush=True)
        finally:
            del metric
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    numeric_keys = ("psnr", "ssim", "lpips", "vbench_temporal_flickering", *OPTIONAL_NO_REFERENCE_VBENCH)
    aggregate = {
        key: (float(np.mean([row[key] for row in results])) if all(row[key] is not None for row in results) else None)
        for key in numeric_keys
    }
    aggregate["flow_loss_real_camera"] = float(np.mean([
        row["flow_validation"]["flow_loss_real_camera"] for row in results
    ]))
    aggregate["flow_loss_static_camera"] = float(np.mean([
        row["flow_validation"]["flow_loss_static_camera"] for row in results
    ]))
    summary = {
        "checkpoint_label": args.checkpoint_label,
        "num_gt_cases": len(results),
        "num_single_action_cases": len(results),
        "metric_status": metric_status,
        "camera_direction_adherence": "requires separate visual/geometry review; VBench quality is not control accuracy",
        "mean": aggregate,
        "cases": results,
    }
    path = args.output_root / "metrics.json"
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {path}", flush=True)


if __name__ == "__main__":
    main()
