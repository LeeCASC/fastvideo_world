"""Score native H3 outputs, preserving unsupported/unavailable metrics as null."""
from __future__ import annotations

import argparse
import gc
import importlib
import json
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import torch

from scripts.evaluation.portable_js_easy_h3 import INDEX, atomic_json

DIMENSIONS = {
    "subject_consistency": "SubjectConsistencyMetric",
    "background_consistency": "BackgroundConsistencyMetric",
    "temporal_flickering": "TemporalFlickeringMetric",
    "motion_smoothness": "MotionSmoothnessMetric",
    "dynamic_degree": "DynamicDegreeMetric",
    "aesthetic_quality": "AestheticQualityMetric",
    "imaging_quality": "ImagingQualityMetric",
}


def read_video(path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, image = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    cap.release()
    frames = np.stack(frames)
    if frames.shape != (124, 480, 832, 3):
        raise ValueError(f"Unexpected decoded video {path}: {frames.shape}")
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    cases = json.loads((args.output / "inference_manifest.json").read_text())
    for case in cases:
        if not (args.output / "videos" / case["output_key"] / "VIDEO_COMPLETE").exists():
            raise ValueError(f"Missing complete video: {case['output_key']}")
    atomic_json(args.output / "status.json", {"phase": "metrics", "videos_complete": len(cases)})
    import lpips

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    lp = lpips.LPIPS(net="alex").to(device).eval()
    gt_scores = []
    with torch.inference_mode():
        for case in cases:
            if case["group"] != "trajectory_gt":
                continue
            folder = args.output / "videos" / case["output_key"]
            generated = read_video(folder / "raw.mp4")
            gt = np.load(args.output / "inputs" / case["sample_id"] / "gt_pixels_61.npy")[INDEX]
            frame_scores = []
            for frame_index, (left, right) in enumerate(zip(gt, generated, strict=True)):
                frame_scores.append({
                    "frame": frame_index,
                    "psnr": float(peak_signal_noise_ratio(left, right, data_range=255)),
                    "ssim": float(structural_similarity(left, right, channel_axis=2, data_range=255)),
                })
            for start in range(0, 124, 8):
                a = torch.from_numpy(gt[start:start+8].copy()).permute(0, 3, 1, 2).to(device).float()/127.5-1
                b = torch.from_numpy(generated[start:start+8].copy()).permute(0, 3, 1, 2).to(device).float()/127.5-1
                values = lp(a, b).reshape(-1).cpu().tolist()
                for j, value in enumerate(values):
                    frame_scores[start+j]["lpips"] = value
            flow = json.loads((folder / "flow_loss.json").read_text())
            score = {"id": case["id"], "group": case["group"], "alpha": case["alpha"],
                "val_loss": flow["val_loss"], "static_camera_loss": flow["static_camera_loss"],
                **{key: float(np.mean([r[key] for r in frame_scores])) for key in ("psnr", "ssim", "lpips")},
                "first_frame_included": True, "temporal_alignment": "61 source -> 124 H3, equal progress",
                "frame_metrics": frame_scores}
            atomic_json(folder / "metrics.json", score)
            gt_scores.append(score)
            print(f"GT_METRICS {case['output_key']} PSNR={score['psnr']:.4f}", flush=True)
    del lp
    gc.collect()
    torch.cuda.empty_cache()
    no_ref = [row for row in cases if row["group"] != "trajectory_gt"]
    rows = {row["output_key"]: {"id": row["id"], "group": row["group"], "alpha": row["alpha"]} for row in no_ref}
    status = {}
    # Use existing metric implementations without modifying algorithms or
    # silently renormalizing scores. Missing checkpoints/packages are explicit.
    for name, cls in DIMENSIONS.items():
        metric = None
        try:
            module = importlib.import_module(f"fastvideo.eval.metrics.vbench.{name}.metric")
            metric = getattr(module, cls)()
            metric.to(device if metric.needs_gpu else "cpu")
            metric.setup()
            for case in no_ref:
                pixels = read_video(args.output / "videos" / case["output_key"] / "raw.mp4")
                video = torch.from_numpy(pixels.copy()).permute(0, 3, 1, 2).float().div_(255)
                rows[case["output_key"]][name] = float(metric.compute({"video": video}).score)
            status[name] = "complete using FastVideo VBench metric adapter; not official aggregate"
        except Exception as error:
            status[name] = f"unavailable: {type(error).__name__}: {error}"
            for row in rows.values():
                row[name] = None
        finally:
            del metric
            gc.collect()
            torch.cuda.empty_cache()
        print(f"NOREF_METRIC {name}: {status[name]}", flush=True)
    summaries = []
    for alpha in (1.0, 0.5):
        gt = [row for row in gt_scores if row["alpha"] == alpha]
        summaries.append({"group": "trajectory_gt", "alpha": alpha, "samples": len(gt),
            **{key: float(np.mean([row[key] for row in gt])) for key in ("val_loss", "psnr", "ssim", "lpips")}})
        for group in ("single_action", "single_action_text"):
            selected = [row for row in rows.values() if row["alpha"] == alpha and row["group"] == group]
            summaries.append({"group": group, "alpha": alpha, "samples": len(selected),
                **{key: float(np.mean([row[key] for row in selected])) if all(row[key] is not None for row in selected) else None
                   for key in DIMENSIONS}})
    uses_text = any(case.get("text_condition_used", False) for case in cases)
    atomic_json(args.output / "metrics.json", {"summary": summaries, "gt_cases": gt_scores,
        "no_reference_cases": list(rows.values()), "no_reference_metric_status": status,
        "official_vbench_seven_complete": False,
        "subject_text_control": ("Actual text was injected; direction adherence needs independent video review"
            if uses_text else "Vision-only inference; C group is a negative control"),
        "camera_control_accuracy": "requires independent geometric/video review; quality scores do not establish adherence"})
    atomic_json(args.output / "comparison.json", summaries)
    import csv
    fields = sorted({key for row in summaries for key in row})
    with (args.output / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)
    (args.output / "INFERENCE_COMPLETE").touch()
    (args.output / "CORE_METRICS_COMPLETE").touch()
    atomic_json(args.output / "status.json", {"phase": "complete_with_metric_limitations",
        "videos_complete": len(cases), "official_vbench_complete": False, "metric_status": status})


if __name__ == "__main__":
    main()
