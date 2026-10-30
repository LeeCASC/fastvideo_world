"""Adapt the portable JS-easy protocol to native MiniMax-H3 conditioning.

Run prepare on independent GPU workers, infer once on eight ranks (SP=2),
then decode independently. Source WAN latents/UMT5 are deliberately not read.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import gc
import hashlib
import json
import os
from pathlib import Path
import time

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import yaml

from fastvideo.dataset.dataloader.schema import pyarrow_schema_minimax_h3_worldmodel_prope as SCHEMA
from fastvideo.dataset.utils import collate_rows_from_parquet_schema
from fastvideo.dataset.vwm_h3 import h3_latent_source_frame_indices, rebase_camera_to_first_frame
from fastvideo.distributed import (
    get_local_torch_device, get_sp_group, get_world_rank,
    maybe_init_distributed_environment_and_model_parallel,
)
from fastvideo.models.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler
from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_conditioning import MiniMaxH3ConditioningStage
from fastvideo.pipelines.preprocess.minimax_h3_worldmodel.encoding import (
    encode_h3_worldmodel_condition, encode_h3_worldmodel_video,
)
from fastvideo.pipelines.preprocess.minimax_h3_worldmodel.preprocess import (
    _fastvideo_args, _initialize_worker, _load_component, _load_model_index,
)
from fastvideo.train.utils.config import load_run_config
from fastvideo.train.utils.instantiate import instantiate
from fastvideo.training.checkpointing_utils import ModelWrapper
from scripts.evaluation.infer_minimax_h3_worldmodel_checkpoint import _audit_checkpoint_state, _decode_video
from scripts.evaluation.prepare_minimax_h3_js_easy16_gt import read_geometry
from scripts.evaluation.prepare_minimax_h3_js_easy_benchmark_20261008 import _encode_keyframe, _set_float32_tensor

INDEX = np.rint(np.linspace(0, 60, 124)).astype(np.int64)
ANCHORS = h3_latent_source_frame_indices(124)
NEUTRAL_CAPTION = "A video scene observed from a moving camera."


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temp, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def crop(image: Image.Image) -> Image.Image:
    width, height = image.size
    scale = max(832 / width, 480 / height)
    image = image.resize((max(832, round(width * scale)), max(480, round(height * scale))), Image.Resampling.LANCZOS)
    left, top = (image.width - 832) // 2, (image.height - 480) // 2
    return image.crop((left, top, left + 832, top + 480))


def write_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".parquet.tmp")
    pq.write_table(pa.Table.from_pylist([row], schema=SCHEMA), temporary, compression="zstd")
    os.replace(temporary, path)


def read_row(path: Path) -> dict:
    return pq.read_table(path).to_pylist()[0]


def inference_config(source: dict, *, preserve_init_checkpoint: bool = False,
                     num_gpus: int | None = None) -> dict:
    """Keep an explicitly requested model bootstrap, never trainer/optimizer state."""
    cfg = deepcopy(source)
    if not preserve_init_checkpoint:
        cfg["models"]["student"]["init_from_training_checkpoint"] = ""
    cfg["training"]["checkpoint"]["resume_from_checkpoint"] = ""
    if num_gpus is not None:
        if num_gpus not in (4, 8):
            raise ValueError("Portable H3 evaluation supports four or eight ranks with SP2")
        distributed = cfg["training"]["distributed"]
        if distributed["sp_size"] != 2 or distributed.get("tp_size", 1) != 1:
            raise ValueError("Portable H3 evaluation requires SP2 and TP1")
        distributed.update(num_gpus=num_gpus, hsdp_replicate_dim=num_gpus // 4, hsdp_shard_dim=4)
    return cfg


def freeze(args) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / "checkpoint" / args.checkpoint.name
    required = [args.checkpoint / "dcp/.metadata", args.checkpoint / "metadata.json"]
    if not all(p.is_file() and p.stat().st_size for p in required):
        raise ValueError("Checkpoint metadata is incomplete")
    shards = list((args.checkpoint / "dcp").glob("*.distcp"))
    if len(shards) != 8 or any(p.stat().st_size == 0 for p in shards):
        raise ValueError("Expected eight complete DCP shards")
    for source in args.checkpoint.rglob("*"):
        if source.is_file():
            dest = target / source.relative_to(args.checkpoint)
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                os.link(source, dest)
    hashes = {str(p.relative_to(target)): sha256(p) for p in sorted(target.rglob("*")) if p.is_file()}
    cfg = inference_config(yaml.safe_load(args.config.read_text()),
        preserve_init_checkpoint=getattr(args, "preserve_init_checkpoint", False),
        num_gpus=getattr(args, "inference_num_gpus", None))
    bootstrap = cfg["models"]["student"].get("init_from_training_checkpoint", "")
    if bootstrap and not (Path(bootstrap) / "dcp/.metadata").is_file():
        raise ValueError(f"Missing requested model bootstrap checkpoint: {bootstrap}")
    config_dest = args.output / "inference_config.yaml"
    if config_dest.exists() and yaml.safe_load(config_dest.read_text()) != cfg:
        raise ValueError("Refusing to overwrite a different inference configuration")
    config_dest.write_text(yaml.safe_dump(cfg, sort_keys=False))
    rows = json.loads((args.bundle / "manifests/full.json").read_text())
    assert Counter(row["group"] for row in rows) == {"trajectory_gt": 16, "single_action": 16, "single_action_text": 32}
    cases = []
    for alpha in (1.0, 0.5):
        for source in rows:
            row = dict(source, alpha=alpha, frames=124)
            row["source_frames"] = 61
            row["source_fps"] = source["fps"]
            # Preserve original duration while documenting the H3 frame grid.
            row["fps"] = float(source["fps"]) * 124 / 61
            row["text_condition_used"] = not cfg["models"]["student"].get("vision_only_conditioning", False)
            row["output_key"] = f"alpha{round(alpha * 100):03d}/{row['group']}/{row['id']}"
            cases.append(row)
    # HSDP collectives span multiple SP groups. All lanes in a wave must
    # perform the same number of forwards (GT adds two loss forwards).
    cases.sort(key=lambda row: (-row["alpha"], row["group"], row["id"]))
    existing = args.output / "inference_manifest.json"
    if existing.exists() and json.loads(existing.read_text()) != cases:
        raise ValueError("Output already contains a different manifest")
    atomic_json(existing, cases)
    atomic_json(args.output / "checkpoints.json", {
        "source_checkpoint": str(args.checkpoint), "frozen_checkpoint": str(target),
        "checkpoint_sha256": hashes, "config_sha256": sha256(config_dest),
        "training_config": str(args.config), "step": int(args.checkpoint.name.split("-")[-1]),
        "base_model": str(args.model_path), "architecture": "MiniMax-H3 residual PRoPE, SP=2",
        "gpu_count": cfg["training"]["distributed"]["num_gpus"],
        "parallel_video_lanes": cfg["training"]["distributed"]["num_gpus"] // 2,
        "model_bootstrap_checkpoint": bootstrap or None,
        "model_load_order": [str(args.model_path)] + ([bootstrap] if bootstrap else []) + [str(target)],
        "text_condition_used": cases[0]["text_condition_used"], "audio": False,
        "requested_cases": 128, "bundle": str(args.bundle),
        "protocol_differences": {
            "frames": "H3 native 124; 61 source frames sampled at equal progress, duration-preserving FPS",
            "vae": "Native MiniMax-H3; WAN latents are not reused",
            "encoder": "Native H3 Qwen3-VL image/text conditions; UMT5 is not reused",
            "sampling": "Native H3 flow scheduler shift=12, 50 requested steps, conditional guidance=1; no text CFG",
            "text": ("Actual manifest prompt and exact first frame must be jointly encoded and audited"
                if cases[0]["text_condition_used"] else
                "Source prompts retained for audit; vision-only inference ignores scene/subject text"),
            "C_group": ("Matched subject forward/backward text intervention; sensitivity is not semantic accuracy"
                if cases[0]["text_condition_used"] else
                "Negative control of unsupported text input, not evidence of subject-text ability"),
            "metrics": "All 124 decoded frames included; not directly comparable to native WAN 61-frame metrics",
        },
    })
    atomic_json(args.output / "status.json", {"phase": "frozen", "expected_videos": 128, "training_paused": True})
    print(f"FROZEN {target}", flush=True)


@torch.no_grad()
def prepare(args) -> None:
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    device = _initialize_worker(0, 30030 + args.worker)
    model_index = _load_model_index(args.model_path)
    fast_args = _fastvideo_args(args.model_path)
    names = [f"js_{i:05d}" for i in range(1, 17)][args.worker::args.workers]
    pending = [name for name in names if not (args.output / "inputs" / name / "COMPLETE").exists()]
    if args.worker == 0:
        atomic_json(args.output / "status.json", {"phase": "preparing_native_h3_inputs", "workers": args.workers,
            "scenes": 16, "expected_videos": 128, "training_paused": True})
    if not pending:
        dist.destroy_process_group()
        return
    vae = _load_component("vae", args.model_path, model_index, fast_args)
    vae.eval()
    for name in pending:
        folder = args.bundle / "data" / name
        out = args.output / "inputs" / name
        out.mkdir(parents=True, exist_ok=True)
        image = crop(Image.open(folder / "sample_001_first_frame.png").convert("RGB"))
        image.save(out / "first_frame.png")
        capture = cv2.VideoCapture(str(folder / "sample_001_gt.mp4"))
        frames = []
        source_hw = None
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            source_hw = bgr.shape[:2]
            frames.append(np.asarray(crop(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))))
        source_fps = capture.get(cv2.CAP_PROP_FPS)
        capture.release()
        if len(frames) != 61:
            raise ValueError(f"Expected 61 decoded source frames: {name}")
        pixels = np.stack(frames)
        np.save(out / "gt_pixels_61.npy", pixels)
        video = torch.from_numpy(pixels[INDEX].copy()).permute(3, 0, 1, 2)
        target, _ = encode_h3_worldmodel_video(vae, video, device=device)
        first = _encode_keyframe(vae, image, device)
        camera, intrinsics = read_geometry(folder, source_hw)
        # Use the portable package's actual camera matrices, including its
        # canonical signs, rather than recreating its keyboard trajectories.
        gt_poses = np.asarray(json.loads((args.bundle / "inputs" / f"{name}_trajectory_gt_pose.json").read_text())["c2w"], np.float32)
        single_poses = np.asarray(json.loads((args.bundle / "inputs" / f"{name}_single_action_pose.json").read_text())["c2w"], np.float32)
        for poses in (gt_poses, single_poses):
            if poses.shape != (61, 4, 4) or not np.isfinite(poses).all():
                raise ValueError(f"Invalid portable camera path: {name}")
            r = poses[:, :3, :3]
            if not np.allclose(r.transpose(0, 2, 1) @ r, np.eye(3), atol=0.02) or not (np.linalg.det(r) > 0.95).all():
                raise ValueError(f"Non-rigid portable camera path: {name}")
        # Native read_geometry performs calibration normalization/crop checks.
        # Its camera return is retained only to audit conventions against
        # the authoritative portable GT trajectory.
        gt_camera = rebase_camera_to_first_frame(gt_poses[INDEX][ANCHORS])
        np.save(out / "trajectory_gt.npy", gt_camera)
        np.save(out / "single_action.npy", rebase_camera_to_first_frame(single_poses[INDEX][ANCHORS]))
        row = {
            "id": f"portable/{name}", "source_name": "js_easy_benchmark_portable_20261021",
            "caption": NEUTRAL_CAPTION, "media_type": "video_worldmodel",
            "width": 832, "height": 480, "num_frames": 124, "fps": float(source_fps) * 124 / 61,
            "duration_sec": 61 / source_fps, "source_fps": float(source_fps),
            "source_frame_indices": INDEX.tolist(),
        }
        for key, value in {
            "vae_latent": target.numpy(), "first_frame_latent": first,
            "camera_to_world": gt_camera, "camera_intrinsics": intrinsics,
            "action_continuous": np.zeros((37, 6), np.float32),
            "action_valid_mask": np.ones(37, np.float32),
        }.items():
            _set_float32_tensor(row, key, value)
        # Intermediate has no Qwen fields yet; save only after conditioner.
        torch.save(row, out / "vae_row.pt")
        atomic_json(out / "alignment.json", {
            "source_sha256": {p.name: sha256(p) for p in sorted(folder.iterdir()) if p.is_file()},
            "source_indices": INDEX.tolist(), "h3_anchors": ANCHORS.tolist(),
            "pixel_preprocessing": "PIL LANCZOS aspect-preserving center crop 832x480",
            "native_geometry_max_difference_from_portable": float(np.abs(camera - gt_camera).max()),
            "future_gt_only_for_validation": True,
        })
        del video, target, pixels, frames
        gc.collect()
        torch.cuda.empty_cache()
        print(f"VAE_READY {name}", flush=True)
    del vae
    gc.collect()
    torch.cuda.empty_cache()
    tokenizer = _load_component("tokenizer", args.model_path, model_index, fast_args)
    processor = _load_component("processor", args.model_path, model_index, fast_args)
    encoder = _load_component("text_encoder", args.model_path, model_index, fast_args)
    stage = MiniMaxH3ConditioningStage(conditioner=encoder, tokenizer=tokenizer, processor=processor)
    for name in pending:
        out = args.output / "inputs" / name
        image = Image.open(out / "first_frame.png").convert("RGB")
        embedding, tags = encode_h3_worldmodel_condition(stage, fast_args, caption=NEUTRAL_CAPTION, first_frame=image)
        row = torch.load(out / "vae_row.pt", weights_only=False, map_location="cpu")
        _set_float32_tensor(row, "text_embedding", embedding.numpy())
        _set_float32_tensor(row, "text_token_tags", tags.numpy())
        write_row(out / "gt.parquet", row)
        simple = dict(row)
        _set_float32_tensor(simple, "vae_latent", np.zeros((24, 37, 30, 52), np.float32))
        write_row(out / "simple.parquet", simple)
        (out / "COMPLETE").touch()
        print(f"CONDITION_READY {name}", flush=True)
    dist.destroy_process_group()


def scaled_camera(camera: np.ndarray, alpha: float) -> torch.Tensor:
    camera = np.asarray(camera, np.float32).copy()
    rotation = Rotation.from_matrix(camera[:, :3, :3])
    camera[:, :3, :3] = Rotation.from_rotvec(rotation.as_rotvec() * alpha).as_matrix()
    camera[:, :3, 3] *= alpha
    if not np.allclose(camera[0], np.eye(4), atol=1e-4):
        raise ValueError("Camera must be relative to its first frame")
    return torch.from_numpy(camera)[None]


@torch.no_grad()
def flow_loss(model, batch, case) -> dict:
    noisy = batch.noisy_model_input.permute(0, 2, 1, 3, 4)
    target = batch.noise.permute(0, 2, 1, 3, 4).float() - batch.latents.float()
    correct = model.predict_noise(noisy, batch.timesteps, batch, conditional=True)
    original = batch.input_kwargs["camera_to_world"]
    batch.input_kwargs["camera_to_world"] = original[:, :1].expand_as(original).clone()
    try:
        static = model.predict_noise(noisy, batch.timesteps, batch, conditional=True)
    finally:
        batch.input_kwargs["camera_to_world"] = original
    return {
        "val_loss": float((correct.float() - target).square().mean()),
        "static_camera_loss": float((static.float() - target).square().mean()),
        "camera_prediction_rms_response": float((correct.float() - static.float()).square().mean().sqrt()),
        "definition": "native H3 fixed-seed shifted-video-flow MSE; not comparable to Cosmos/WAN loss",
        "seed": case["seed"], "timestep": batch.timesteps.detach().float().cpu().tolist(),
        "alpha": case["alpha"],
    }


@torch.no_grad()
def infer(args) -> None:
    cfg = load_run_config(args.output / "inference_config.yaml")
    tc = cfg.training
    maybe_init_distributed_environment_and_model_parallel(tc.distributed.tp_size, tc.distributed.sp_size)
    rank, group = get_world_rank(), get_sp_group()
    device = get_local_torch_device()
    torch.cuda.set_device(device)
    torch.set_num_threads(2)
    model = instantiate(cfg.models["student"], training_config=tc)
    state = {"roles.student.transformer": ModelWrapper(model.transformer)}
    checkpoint = args.output / "checkpoint" / args.checkpoint.name
    dcp.load(state, checkpoint_id=str(checkpoint / "dcp"))
    _audit_checkpoint_state(model, device)
    model.transformer.eval()
    model.sp_group = group
    cases = json.loads((args.output / "inference_manifest.json").read_text())
    if rank == 0:
        atomic_json(args.output / "status.json", {"phase": "inference",
            "checkpoint_step": int(args.checkpoint.name.split("-")[-1]),
            "gpu_count": dist.get_world_size(), "parallel_videos": dist.get_world_size() // 2,
            "expected_videos": len(cases), "training_paused": True})
    text_len = int(tc.pipeline_config.text_encoder_configs[0].arch_config.text_len)
    lane, lanes = rank // 2, dist.get_world_size() // 2
    if len(cases) % lanes:
        raise ValueError("Cases must divide evenly across SP groups")
    for wave in range(len(cases) // lanes):
        case = cases[wave * lanes + lane]
        folder = args.output / "videos" / case["output_key"]
        completed = (folder / "LATENT_COMPLETE").exists()
        pending = torch.tensor(int(not completed), device=device)
        dist.all_reduce(pending, op=dist.ReduceOp.MAX)
        if not pending.item():
            continue
        start = time.time()
        inp = args.output / "inputs" / case["sample_id"]
        is_gt = case["group"] == "trajectory_gt"
        row_path = Path(case["condition_parquet"]) if case.get("condition_parquet") else inp / (
            "gt.parquet" if is_gt else "simple.parquet")
        row = read_row(row_path)
        batch = collate_rows_from_parquet_schema([row], SCHEMA, text_len, cfg_rate=0)
        path = np.load(inp / ("trajectory_gt.npy" if is_gt else "single_action.npy"))
        camera = scaled_camera(path, case["alpha"])
        batch["camera_to_world"] = camera
        if not is_gt:
            batch["camera_intrinsics"] = batch["camera_intrinsics"][:, :1].expand(-1, 37, -1, -1).clone()
        generator = torch.Generator(device=device).manual_seed(case["seed"])
        prepared = model.prepare_batch(batch, generator=generator, latents_source="data")
        condition_audit = {
            "vision_only_conditioning": model.vision_only_conditioning,
            "embedding_rows_on_disk": row["text_embedding_shape"][0],
            "embedding_rows_entering_dit": int(prepared.encoder_hidden_states.shape[1]),
            "text_token_count_entering_dit": int((prepared.minimax_h3_layout.token_tags[
                prepared.minimax_h3_layout.text_indices] == 1).sum()),
            "vision_condition_token_count_entering_dit": int((prepared.minimax_h3_layout.token_tags[
                prepared.minimax_h3_layout.text_indices] == 0).sum()),
            "condition_parquet": str(row_path),
        }
        if case.get("text_condition_used"):
            if model.vision_only_conditioning or condition_audit["text_token_count_entering_dit"] <= 0:
                raise ValueError("Text-enabled evaluation did not retain actual text tokens")
            if row["caption"] != case["prompt"] or row["text_embedding_shape"][0] > text_len:
                raise ValueError("Condition prompt mismatch or text would be truncated")
        if group.is_first_rank:
            print(f"CONDITION_AUDIT {case['output_key']} {condition_audit}", flush=True)
        metrics = flow_loss(model, prepared, case) if is_gt else None
        # The generation state is independent of target video/GT flow-loss
        # generator; future video latents are never fed into the rollout.
        rng = torch.Generator(device=device).manual_seed(case["seed"] + 1)
        latents = torch.randn(prepared.latents.shape, generator=rng, device=device, dtype=torch.bfloat16)
        scheduler = MiniMaxH3Scheduler(shift=12.0)
        scheduler.set_timesteps(50, device=device)
        for step, timestep in enumerate(scheduler.timesteps):
            prepared.timesteps = timestep.reshape(1)
            prediction = model.predict_noise(latents, prepared.timesteps, prepared, conditional=True)
            latents = scheduler.step(-prediction, timestep, latents, return_dict=False)[0]
            if group.is_first_rank and (step % 10 == 0 or step + 1 == len(scheduler.timesteps)):
                print(f"INFER wave={wave} lane={lane} {case['output_key']} step={step+1}/{len(scheduler.timesteps)} seconds={time.time()-start:.1f}", flush=True)
        if group.is_first_rank and not completed:
            folder.mkdir(parents=True, exist_ok=True)
            torch.save(latents.cpu().to(torch.float16), folder / "latents.pt")
            np.save(folder / "camera_to_world.npy", camera[0].numpy())
            if metrics is not None:
                atomic_json(folder / "flow_loss.json", metrics)
            atomic_json(folder / "metadata.json", dict(case, checkpoint=str(checkpoint),
                generation_seconds=time.time()-start, generation_future_gt_input=False,
                scene_or_subject_text_used=not model.vision_only_conditioning,
                conditioning_audit=condition_audit,
                reference_frame_source=str(inp / "first_frame.png"),
                scheduler="MiniMaxH3Scheduler", scheduler_shift=12.0,
                requested_inference_steps=50, actual_transformer_calls=len(scheduler.timesteps),
                camera_intrinsics_used=model.prope_use_camera_intrinsics,
                audio_tokens=0))
            (folder / "LATENT_COMPLETE").touch()
            print(f"LATENT_COMPLETE {case['output_key']} seconds={time.time()-start:.1f}", flush=True)
        del prepared, latents, batch, row
        gc.collect()
        torch.cuda.empty_cache()
        dist.barrier()
    if rank == 0:
        (args.output / "ALL_LATENTS_COMPLETE").touch()
    dist.destroy_process_group()


@torch.no_grad()
def decode(args) -> None:
    from tools.visualize_h3_controls import annotate_case

    torch.set_num_threads(2)
    device = _initialize_worker(0, 30130 + args.worker)
    index = _load_model_index(args.model_path)
    vae = _load_component("vae", args.model_path, index, _fastvideo_args(args.model_path))
    vae.eval()
    all_cases = json.loads((args.output / "inference_manifest.json").read_text())
    cases = all_cases[args.worker::args.workers]
    if args.worker == 0:
        atomic_json(args.output / "status.json", {"phase": "decoding", "workers": args.workers,
            "expected_videos": len(all_cases), "training_paused": True})
    for case in cases:
        folder = args.output / "videos" / case["output_key"]
        path = folder / "raw.mp4"
        if (folder / "VIDEO_COMPLETE").exists():
            annotate_case(folder / "metadata.json", args.output)
            continue
        latents = torch.load(folder / "latents.pt", map_location="cpu", weights_only=True)
        frames = _decode_video(vae, latents, device)
        if frames.shape != (124, 480, 832, 3):
            raise ValueError(f"Unexpected decoded shape {frames.shape}")
        temporary = folder / "raw.tmp.mp4"
        imageio.mimsave(temporary, frames, fps=case["fps"], codec="libx264", quality=8, macro_block_size=1)
        os.replace(temporary, path)
        cap = cv2.VideoCapture(str(path))
        count = 0
        while True:
            ok, image = cap.read()
            if not ok:
                break
            if image.shape != (480, 832, 3):
                raise ValueError("MP4 pixel geometry mismatch")
            count += 1
        cap.release()
        if count != 124:
            raise ValueError(f"MP4 incomplete: {count} frames")
        (folder / "VIDEO_COMPLETE").touch()
        # Flat raw views for the official VBench CLI.
        flat = args.output / "raw" / f"alpha{round(case['alpha']*100):03d}" / case["group"]
        flat.mkdir(parents=True, exist_ok=True)
        dest = flat / f"{case['id']}.mp4"
        if not dest.exists():
            os.link(path, dest)
        print(f"VIDEO_COMPLETE {case['output_key']}", flush=True)
        annotate_case(folder / "metadata.json", args.output)
        del frames, latents
        gc.collect()
        torch.cuda.empty_cache()
    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["freeze", "prepare", "infer", "decode"])
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=Path("/datacc05/lizihao_wm/worldmodel/ckpts/MiniMax-H3"))
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--preserve-init-checkpoint", action="store_true",
        help="Retain the training model's parent-weight bootstrap before loading the evaluated checkpoint")
    parser.add_argument("--inference-num-gpus", type=int, choices=[4, 8], default=None,
        help="Override only the inference HSDP mesh; do not modify the source training configuration")
    args = parser.parse_args()
    globals()[args.command](args)


if __name__ == "__main__":
    main()
