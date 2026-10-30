#!/usr/bin/env python3
"""Generate resumable scene/subject captions for materialized 61f windows.

Camera motion is intentionally excluded from the caption. Camera control is
represented only by the numeric pose condition in the registry/pose archive.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

H3_VWM = Path("/home/lizihao/projects/worldmodel/H3-World/code/vwm")
if str(H3_VWM) not in sys.path:
    sys.path.insert(0, str(H3_VWM))
from caption_scenes_qwen3vl import (  # noqa: E402
    has_camera_control_language,
    infer_batch,
    load_frames,
    parse_json,
)

PROMPT_POLICY = "observed_scene_subject_motion_no_camera_v2"
PROMPT = """Analyze six chronological frames from one video window. Return exactly one JSON object:
{"scene_description":"...","confidence":0.0,"quality_flags":[]}
Write one to three factual English sentences, at most 80 words. Describe the
setting, lighting, visible subjects/objects, appearance, and visible subject or
object motion across frames. Describe only evidence in the frames. Never
describe camera motion, camera controls, the viewpoint, shot, framing,
perspective, panning, tilting, zooming, tracking, following, drifting, or a
stationary camera. Do not infer intent, location, audio, or events outside the
frames. Output JSON only."""


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def completed_ids(root: Path) -> set[str]:
    done: set[str] = set()
    for path in root.glob("captions_shard_*.jsonl"):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("error") in (None, "") and row.get("scene_description"):
                    done.add(str(row.get("global_id")))
    return done


def sampled_indices(row: dict, samples: int) -> list[int]:
    source = row.get("source_frame_indices")
    if not isinstance(source, list) or not source:
        raise ValueError("missing source_frame_indices")
    positions = np.rint(np.linspace(0, len(source) - 1, samples)).astype(np.int64)
    return [int(source[position]) for position in positions]


def clean_camera_caption(text: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    kept = [sentence.strip() for sentence in sentences if sentence.strip() and not has_camera_control_language(sentence)]
    return " ".join(kept).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model", type=Path, default=Path("/datacc05/lizihao/models/worldarena/qwenvl3"))
    parser.add_argument("--num-frames", type=int, default=6)
    parser.add_argument("--max-frame-edge", type=int, default=384)
    args = parser.parse_args()
    if not 0 <= args.shard < args.num_shards:
        raise ValueError("invalid shard")
    rows = sorted(read_jsonl(args.registry), key=lambda row: str(row["global_id"]))
    rows = rows[args.shard :: args.num_shards]
    args.output_root.mkdir(parents=True, exist_ok=True)
    output = args.output_root / f"captions_shard_{args.shard:02d}_of_{args.num_shards:02d}.jsonl"
    done = completed_ids(args.output_root)
    rows = [row for row in rows if str(row["global_id"]) not in done]
    print(f"shard={args.shard}/{args.num_shards} remaining={len(rows)} output={output}", flush=True)
    if not rows:
        return

    device = torch.device(args.device)
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    processor.tokenizer.padding_side = "left"
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True
    ).to(device).eval()
    started = time.perf_counter()
    with output.open("a", encoding="utf-8") as handle:
        for position, row in enumerate(rows, start=1):
            result = {
                "window_id": row["window_id"],
                "global_id": row["global_id"],
                "video": row["video"],
                "model": str(args.model),
                "prompt_policy": PROMPT_POLICY,
            }
            try:
                indices = sampled_indices(row, args.num_frames)
                frames = load_frames(Path(row["video"]), indices, args.max_frame_edge)
                raw = infer_batch(model, processor, [frames], device, prompt=PROMPT)[0].strip()
                parsed = parse_json(raw)
                original = parsed["scene_description"]
                cleaned = clean_camera_caption(original)
                if not cleaned:
                    raise ValueError("caption became empty after camera-language filter")
                parsed["scene_description"] = cleaned
                if cleaned != original:
                    parsed["quality_flags"].append("removed_camera_control_sentence")
                result.update(parsed)
                result["sampled_source_frame_indices"] = indices
                result["error"] = None
            except Exception as error:
                result["error"] = f"{type(error).__name__}: {error}"
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            if position % 10 == 0 or position == len(rows):
                os.fsync(handle.fileno())
                elapsed = time.perf_counter() - started
                print(f"[{args.shard}/{args.num_shards}] {position}/{len(rows)} rate={position / max(elapsed, 1e-6):.3f}/s", flush=True)


if __name__ == "__main__":
    main()
