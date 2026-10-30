#!/usr/bin/env python3
"""Build a merged, no-VAE 61-frame registry from old approved and new VWM data.

The script is intentionally metadata/pose-only. It does not modify source
videos, encode VAE latents, or claim visual quality approval. A later CPU/GPU
quality stage consumes the emitted ``quality_status=pending_cpu_quality`` rows.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

import numpy as np


NEW_ROOT = Path("/datacc05/xunzizheng/cosmos-predict2-worldmodel/vwm_unified")
OLD_ROOT = Path("/datacc05/lizihao/worldmodel/fastvideowm_approved_uncached_v1_20260925")
OUT_ROOT = Path("/datacc05/lizihao/worldmodel/vwm_unified_61f_20261015")


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def canonical_source(name: str) -> str:
    if name.startswith("train_v2_"):
        return name[len("train_v2_"):]
    return name


def source_key(row: dict[str, Any]) -> tuple[str, str]:
    return canonical_source(str(row["source_name"])), str(row["sample_id"])


def load_rows() -> list[dict[str, Any]]:
    train_ids = {
        (str(row["source_name"]), str(row["sample_id"]))
        for row in read_jsonl(NEW_ROOT / "manifests/train.jsonl")
    }
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    # Old approved data is the fallback. The new unified record wins on overlap.
    for row in read_jsonl(OLD_ROOT / "manifests/approved_uncached.jsonl"):
        item = dict(row)
        key = source_key(item)
        item.update({
            "registry_source": "previous_fastvideowm_approved",
            "source_root": str(OLD_ROOT),
            "new_train_manifest": False,
        })
        rows.setdefault(key, item)
    for row in read_jsonl(NEW_ROOT / "manifests/all.jsonl"):
        item = dict(row)
        key = source_key(item)
        item.update({
            "registry_source": "cosmos_vwm_unified",
            "source_root": str(NEW_ROOT),
            "new_train_manifest": (str(row["source_name"]), str(row["sample_id"])) in train_ids,
        })
        rows[key] = item
    return list(rows.values())


def make_window(row: dict[str, Any], window_frames: int) -> dict[str, Any]:
    fps = float(row.get("fps") or 0.0)
    nframes = int(row.get("num_frames") or 0)
    if fps <= 0 or nframes <= 0:
        raise ValueError("missing fps/num_frames")
    source_indices = np.rint(np.arange(window_frames, dtype=np.float64) * fps / 24.0).astype(int)
    if int(source_indices[-1]) >= nframes:
        raise ValueError(f"source clip is too short for {window_frames} frames at 24 fps")
    root = Path(str(row["source_root"]))
    record = root / str(row["record_path"])
    video = record / "video.mp4"
    camera = record / "camera_c2w.npz"
    prompt = record / "prompt.txt"
    if not video.is_file() or not camera.is_file() or not prompt.is_file():
        raise FileNotFoundError(f"missing video/camera/prompt under {record}")
    # Structural pose check only; visual quality remains a separate stage.
    with np.load(camera, allow_pickle=False) as payload:
        key = "camera_to_world" if "camera_to_world" in payload else "c2w"
        c2w = np.asarray(payload[key])
        intrinsics = np.asarray(payload["intrinsics"]) if "intrinsics" in payload else None
    if c2w.ndim != 3 or c2w.shape[1:] != (4, 4) or not np.isfinite(c2w).all():
        raise ValueError(f"invalid camera shape={c2w.shape}")
    if c2w.shape[0] not in (nframes, 37, 61, 121, 124, 243, 241, 362, 345):
        raise ValueError(f"unexpected camera length={c2w.shape[0]}")
    if intrinsics is not None and (not np.isfinite(intrinsics).all()):
        raise ValueError("non-finite intrinsics")
    canonical = canonical_source(str(row["source_name"]))
    window_id = f"{canonical}:{row['sample_id']}:{window_frames}f_s0"
    return {
        "window_id": window_id,
        "source_name": str(row["source_name"]),
        "canonical_source": canonical,
        "sample_id": str(row["sample_id"]),
        "global_id": f"{canonical}:{row['sample_id']}",
        "registry_source": row["registry_source"],
        "source_root": str(root),
        "record_path": str(row["record_path"]),
        "video": str(video),
        "camera": str(camera),
        "prompt_source": str(prompt),
        "source_frames": nframes,
        "source_fps": fps,
        "target_fps": 24.0,
        "target_num_frames": window_frames,
        "window_start": 0,
        "window_policy": "prefix_window_v1",
        "source_frame_indices": source_indices.tolist(),
        "width": int(row.get("width") or 0),
        "height": int(row.get("height") or 0),
        "new_train_manifest": bool(row.get("new_train_manifest", False)),
        "quality_status": "pending_cpu_quality",
        "prompt_status": "pending_vlm",
        "vae_status": "not_encoded",
        "pose_status": "structurally_valid",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT_ROOT)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--frames", type=int, default=61)
    args = parser.parse_args()
    if args.frames < 1:
        raise ValueError("--frames must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "registry").mkdir(exist_ok=True)
    (args.output / "prompts").mkdir(exist_ok=True)
    rows = load_rows()
    if args.limit:
        rows = rows[:args.limit]
    result = []
    failures = []
    for number, row in enumerate(rows, 1):
        try:
            result.append(make_window(row, args.frames))
        except Exception as exc:
            failures.append({
                "source_name": row.get("source_name"),
                "sample_id": row.get("sample_id"),
                "record_path": row.get("record_path"),
                "error": str(exc),
            })
        if number % 10000 == 0:
            print(f"processed={number}/{len(rows)} accepted={len(result)} failed={len(failures)}", flush=True)
    with (args.output / "registry" / f"windows_{args.frames}_candidates.jsonl").open("w", encoding="utf-8") as handle:
        for item in result:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    with (args.output / "registry" / f"windows_{args.frames}_structural_failures.jsonl").open("w", encoding="utf-8") as handle:
        for item in failures:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    prompt_queue = args.output / "prompts" / "vlm_queue.jsonl"
    with prompt_queue.open("w", encoding="utf-8") as handle:
        for item in result:
            handle.write(json.dumps({
                "window_id": item["window_id"],
                "video": item["video"],
                "source_frame_indices": item["source_frame_indices"],
                "prompt_status": "pending_vlm",
            }, ensure_ascii=False) + "\n")
    summary = {
        "version": f"vwm_unified_{args.frames}f_20261016_v1",
        "target_frames": args.frames,
        "target_fps": 24.0,
        "input_old_manifest": str(OLD_ROOT / "manifests/approved_uncached.jsonl"),
        "input_new_manifest": str(NEW_ROOT / "manifests/all.jsonl"),
        "unique_input_records": len(rows),
        "candidate_windows": len(result),
        "structural_failures": len(failures),
        "quality_status": "pending_cpu_quality",
        "prompt_status": "pending_vlm",
        "vae_status": "not_encoded",
        "source_counts": dict(Counter(x["canonical_source"] for x in result)),
    }
    (args.output / "reports").mkdir(exist_ok=True)
    (args.output / "reports" / "registry_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
