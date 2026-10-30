"""Text-on diagnostic for 16 matched forward/backward H3 subject-motion pairs."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path

from PIL import Image
import torch
import torch.distributed as dist
import yaml

from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_conditioning import MiniMaxH3ConditioningStage
from fastvideo.pipelines.preprocess.minimax_h3_worldmodel.encoding import encode_h3_worldmodel_condition
from fastvideo.pipelines.preprocess.minimax_h3_worldmodel.preprocess import (
    _fastvideo_args, _initialize_worker, _load_component, _load_model_index,
)
from scripts.evaluation.portable_js_easy_h3 import atomic_json, read_row, sha256, write_row
from scripts.evaluation.prepare_minimax_h3_js_easy_benchmark_20261008 import _set_float32_tensor


def manifest(output: Path) -> list[dict]:
    return json.loads((output / "inference_manifest.json").read_text())


def setup(args) -> None:
    old = json.loads((args.previous / "checkpoints.json").read_text())
    checkpoint = Path(old["frozen_checkpoint"])
    if not (checkpoint / "dcp/.metadata").is_file():
        raise FileNotFoundError(checkpoint)
    args.output.mkdir(parents=True, exist_ok=True)
    linked = args.output / "checkpoint" / checkpoint.name
    linked.parent.mkdir(parents=True, exist_ok=True)
    if not linked.exists():
        linked.symlink_to(checkpoint, target_is_directory=True)
    if linked.resolve() != checkpoint.resolve():
        raise ValueError("Existing checkpoint view points to another model")
    source_cfg = yaml.safe_load((args.previous / "inference_config.yaml").read_text())
    cfg = json.loads(json.dumps(source_cfg))
    cfg["models"]["student"]["vision_only_conditioning"] = False
    config_path = args.output / "inference_config.yaml"
    if config_path.exists() and yaml.safe_load(config_path.read_text()) != cfg:
        raise ValueError("A different configuration already exists in the output directory")
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    rows = json.loads((args.bundle / "manifests/subject_text.json").read_text())
    if len(rows) != 32 or len({r["id"] for r in rows}) != 32:
        raise ValueError("Expected 32 unique subject-text cases")
    for i in range(1, 17):
        sample = f"js_{i:05d}"
        pair = [r for r in rows if r["sample_id"] == sample]
        if len(pair) != 2 or {r["subject_direction"] for r in pair} != {"forward", "backward"}:
            raise ValueError(f"Incomplete subject pair: {sample}")
        for key in ("seed", "camera_pose", "first_frame_description", "action", "fps"):
            if pair[0][key] != pair[1][key]:
                raise ValueError(f"Pair mismatch {sample}/{key}")
        if pair[0]["prompt"] == pair[1]["prompt"]:
            raise ValueError(f"Identical prompts for opposite directions: {sample}")
        inp = args.output / "inputs" / sample
        inp.parent.mkdir(parents=True, exist_ok=True)
        source = args.previous / "inputs" / sample
        if not (source / "COMPLETE").is_file():
            raise ValueError(f"Native H3 image/pose cache not verified: {sample}")
        if not inp.exists():
            inp.symlink_to(source, target_is_directory=True)
        if inp.resolve() != source.resolve():
            raise ValueError(f"Input source changed: {sample}")
    cases = []
    for source in rows:
        row = dict(source, alpha=1.0, frames=124, source_frames=61,
            source_fps=source["fps"], fps=float(source["fps"]) * 124 / 61,
            text_condition_used=True,
            output_key=f"alpha100/single_action_text/{source['id']}",
            condition_parquet=str(args.output / "conditions" / source["id"] / "data.parquet"))
        cases.append(row)
    cases.sort(key=lambda row: row["id"])
    path = args.output / "inference_manifest.json"
    if path.exists() and json.loads(path.read_text()) != cases:
        raise ValueError("Output contains a different frozen manifest")
    atomic_json(path, cases)
    atomic_json(args.output / "checkpoints.json", {
        "source_checkpoint": old["source_checkpoint"], "frozen_checkpoint": str(linked),
        "checkpoint_sha256": old["checkpoint_sha256"], "checkpoint_hash_reference": str(args.previous / "checkpoints.json"),
        "config_sha256": sha256(config_path), "step": 800, "requested_cases": 32,
        "text_condition_used": True, "training_vision_only_conditioning": True,
        "inference_vision_only_conditioning": False,
        "experiment_type": "inference-only text-on diagnostic of an action-only-finetuned checkpoint",
        "config_change_from_previous_evaluation": {"models.student.vision_only_conditioning": [True, False]},
        "base_model": str(args.model_path), "alpha": 1.0,
        "fps_policy": "124-frame H3 native output at 124/61 times source FPS; duration preserved",
        "sampling": "H3 native shift12, requested50 steps, guidance1; no unsupported text CFG",
        "text_encoder": "native MiniMax-H3 Qwen3-VL, full prompt plus exact first frame, no UMT5 reuse",
        "audio": False,
    })
    atomic_json(args.output / "status.json", {"phase": "setup", "expected_videos": 32, "training_paused": True})
    print(f"SETUP_COMPLETE text-on 32 cases: {args.output}", flush=True)


@torch.no_grad()
def condition(args) -> None:
    torch.set_num_threads(2)
    all_cases = manifest(args.output)
    cases = all_cases[args.worker::args.workers]
    pending = [c for c in cases if not (args.output / "conditions" / c["id"] / "COMPLETE").exists()]
    if args.worker == 0:
        atomic_json(args.output / "status.json", {"phase": "encoding_full_qwen_text_and_image",
            "expected_cases": len(all_cases), "workers": args.workers, "text_condition_used": True})
    if not pending:
        return
    _initialize_worker(0, 30230 + args.worker)
    fast_args = _fastvideo_args(args.model_path)
    model_index = _load_model_index(args.model_path)
    tokenizer = _load_component("tokenizer", args.model_path, model_index, fast_args)
    processor = _load_component("processor", args.model_path, model_index, fast_args)
    encoder = _load_component("text_encoder", args.model_path, model_index, fast_args)
    stage = MiniMaxH3ConditioningStage(conditioner=encoder, tokenizer=tokenizer, processor=processor)
    for case in pending:
        inp = args.output / "inputs" / case["sample_id"]
        image = Image.open(inp / "first_frame.png").convert("RGB")
        embedding, tags = encode_h3_worldmodel_condition(stage, fast_args, caption=case["prompt"], first_frame=image)
        if embedding.shape[0] > 1024:
            raise ValueError(f"Prompt would be truncated: {case['id']} has {embedding.shape[0]} tokens")
        if not torch.isfinite(embedding).all() or not bool((tags == 1).any()) or not bool((tags == 0).any()):
            raise ValueError(f"Missing real text/vision tokens: {case['id']}")
        row = read_row(inp / "simple.parquet")
        row["caption"] = case["prompt"]
        row["id"] = f"portable_text_on/{case['id']}"
        _set_float32_tensor(row, "text_embedding", embedding.numpy())
        _set_float32_tensor(row, "text_token_tags", tags.numpy())
        target = Path(case["condition_parquet"])
        write_row(target, row)
        atomic_json(target.parent / "encoding.json", {
            "case_id": case["id"], "sample_id": case["sample_id"], "prompt": case["prompt"],
            "prompt_sha256": hashlib.sha256(case["prompt"].encode()).hexdigest(),
            "embedding_sha256": hashlib.sha256(row["text_embedding_bytes"]).hexdigest(),
            "first_frame_png_sha256": sha256(inp / "first_frame.png"),
            "embedding_rows": int(embedding.shape[0]), "vision_tokens": int((tags == 0).sum()),
            "text_tokens": int((tags == 1).sum()), "encoder": str(args.model_path / "text_encoder"),
            "prompt_truncated": False, "text_condition_used": True,
        })
        (target.parent / "COMPLETE").touch()
        print(f"TEXT_ENCODED {case['id']} text_tokens={int((tags==1).sum())} total_tokens={embedding.shape[0]}", flush=True)
    del stage, encoder
    gc.collect()
    torch.cuda.empty_cache()
    dist.destroy_process_group()


def audit(args) -> None:
    cases = manifest(args.output)
    results = []
    for i in range(1, 17):
        sample = f"js_{i:05d}"
        pair = {c["subject_direction"]: c for c in cases if c["sample_id"] == sample}
        forward = read_row(Path(pair["forward"]["condition_parquet"]))
        backward = read_row(Path(pair["backward"]["condition_parquet"]))
        for key in ("first_frame_latent_bytes", "camera_intrinsics_bytes", "vae_latent_bytes"):
            if forward[key] != backward[key]:
                raise ValueError(f"Non-text pair condition changed: {sample}/{key}")
        if forward["text_embedding_bytes"] == backward["text_embedding_bytes"]:
            raise ValueError(f"Opposite prompts produced identical embeddings: {sample}")
        for row, direction in ((forward, "forward"), (backward, "backward")):
            if row["caption"] != pair[direction]["prompt"]:
                raise ValueError(f"Wrong prompt encoded: {sample}/{direction}")
        enc_f = json.loads((Path(pair["forward"]["condition_parquet"]).parent / "encoding.json").read_text())
        enc_b = json.loads((Path(pair["backward"]["condition_parquet"]).parent / "encoding.json").read_text())
        if enc_f["text_tokens"] <= 0 or enc_b["text_tokens"] <= 0:
            raise ValueError("Text tokens missing")
        results.append({"sample_id": sample, "matched_image_pose_seed": True,
            "forward_text_tokens": enc_f["text_tokens"], "backward_text_tokens": enc_b["text_tokens"],
            "embeddings_differ": True, "prompt_truncated": False})
    atomic_json(args.output / "text_condition_audit.json", {"pairs": results, "passed": True, "cases": 32})
    (args.output / "TEXT_CONDITIONS_COMPLETE").touch()
    print("TEXT_CONDITION_AUDIT_PASSED 16 pairs, 32 actual text embeddings", flush=True)


def summarize(args) -> None:
    cases = manifest(args.output)
    results = []
    for i in range(1, 17):
        sample = f"js_{i:05d}"
        pair = {c["subject_direction"]: c for c in cases if c["sample_id"] == sample}
        tensors = {}
        for direction, case in pair.items():
            folder = args.output / "videos" / case["output_key"]
            if not (folder / "VIDEO_COMPLETE").exists():
                raise ValueError(f"Missing decoded video: {case['id']}")
            meta = json.loads((folder / "metadata.json").read_text())
            if not meta["scene_or_subject_text_used"] or meta["conditioning_audit"]["text_token_count_entering_dit"] <= 0:
                raise ValueError(f"Text did not enter DiT: {case['id']}")
            tensors[direction] = torch.load(folder / "latents.pt", map_location="cpu", weights_only=True).float()
        delta = float((tensors["forward"] - tensors["backward"]).square().mean().sqrt())
        results.append({"sample_id": sample, "forward_backward_latent_rms_difference": delta,
            "meaning": "nonzero establishes text sensitivity, not direction correctness"})
    atomic_json(args.output / "text_response_summary.json", {"pairs": results, "videos_complete": 32,
        "checkpoint": "union/checkpoint-800", "text_tokens_entered_dit": True,
        "training_was_vision_only": True, "inference_text_enabled_explicitly_by_user": True})
    (args.output / "ALL_VIDEOS_COMPLETE").touch()
    (args.output / "INFERENCE_COMPLETE").touch()
    atomic_json(args.output / "status.json", {"phase": "complete", "videos_complete": 32,
        "text_condition_used": True, "training_paused": True,
        "direction_adherence": "requires video/manual or geometric review"})
    print("COMPLETE 32 text-on videos", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["setup", "condition", "audit", "summarize"])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--previous", type=Path, required=True)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--model-path", type=Path, default=Path("/datacc05/lizihao_wm/worldmodel/ckpts/MiniMax-H3"))
    p.add_argument("--worker", type=int, default=0)
    p.add_argument("--workers", type=int, default=1)
    args = p.parse_args()
    globals()[args.command](args)


if __name__ == "__main__":
    main()
