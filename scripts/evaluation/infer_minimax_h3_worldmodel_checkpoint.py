# SPDX-License-Identifier: Apache-2.0
"""Sample a MiniMax-H3 Stage-0 PRoPE DCP checkpoint without audio.

The training checkpoint contains LoRA wrappers, selected video boundary layers,
and PRoPE gates, so loading it through the ordinary released H3 pipeline would
silently omit trained state. This evaluator reconstructs the training model,
loads DCP directly, performs the H3 rectified-flow rollout, and decodes video
latents with the released H3 VAE.

Each trajectory uses one sequence-parallel group. The two ranks in a group use
identical inputs, while independent groups can evaluate different trajectories
or PRoPE gate scales in parallel.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import re
from typing import Any

import imageio.v2 as imageio
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp

from fastvideo.dataset.dataloader.schema import pyarrow_schema_minimax_h3_worldmodel_prope
from fastvideo.dataset.utils import collate_rows_from_parquet_schema
from fastvideo.distributed import (
    get_local_torch_device,
    get_sp_group,
    get_sp_world_size,
    get_world_rank,
    maybe_init_distributed_environment_and_model_parallel,
)
from fastvideo.models.dits.minimax_h3_prope import make_camera_trajectory_variant
from fastvideo.models.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler
from fastvideo.train.utils.config import load_run_config
from fastvideo.train.utils.instantiate import instantiate
from fastvideo.train.utils.moduleloader import load_module_from_path
from fastvideo.training.checkpointing_utils import ModelWrapper


TRAJECTORY_NAMES = ("static", "translate_pos_x", "translate_neg_x", "yaw_pos_y")
ACTION_TRAJECTORIES = tuple("WASDIJKL")
REAL_TRAJECTORY_VARIANTS = ("correct", "static", "reverse", "shuffled")
CASE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


def _read_case_manifest(path: Path, expected_groups: int) -> list[dict[str, Any]]:
    cases = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or len(cases) != expected_groups:
        raise ValueError(f"Case manifest needs {expected_groups} entries, one per sequence-parallel group")
    seen_ids: set[str] = set()
    seen_frames: set[tuple[str, int]] = set()
    for case in cases:
        if not isinstance(case, dict) or not {"case_id", "data_path", "row_index", "trajectory"} <= case.keys():
            raise ValueError("Each case needs case_id, data_path, row_index, and trajectory")
        case_id = case["case_id"]
        data_path = str(case["data_path"])
        row_index = case["row_index"]
        trajectory = case["trajectory"]
        if not isinstance(case_id, str) or CASE_ID_PATTERN.fullmatch(case_id) is None or case_id in seen_ids:
            raise ValueError(f"Unsafe or duplicate case_id: {case_id!r}")
        if not isinstance(row_index, int) or isinstance(row_index, bool) or row_index < 0:
            raise ValueError(f"Invalid row_index for {case_id}: {row_index!r}")
        if trajectory not in (*ACTION_TRAJECTORIES, "real"):
            raise ValueError(f"Invalid single-action trajectory for {case_id}: {trajectory!r}")
        frame_key = (str(Path(data_path).resolve()), row_index)
        if frame_key in seen_frames:
            raise ValueError(f"One first frame must receive only one instruction: {frame_key}")
        seen_ids.add(case_id)
        seen_frames.add(frame_key)
    return cases


def _read_parquet_row(data_path: Path, row_index: int) -> dict[str, Any]:
    files = sorted(data_path.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {data_path}")
    remaining = int(row_index)
    for file_path in files:
        parquet_file = pq.ParquetFile(file_path)
        if remaining >= parquet_file.metadata.num_rows:
            remaining -= parquet_file.metadata.num_rows
            continue
        table = parquet_file.read()
        return {name: values[remaining] for name, values in table.to_pydict().items()}
    raise IndexError(f"row_index={row_index} is outside {data_path}")


def make_camera_trajectory(name: str, num_frames: int, *, scale: float = 1.0) -> torch.Tensor:
    """Return camera-to-world matrices in the training coordinate convention."""
    if name not in (*TRAJECTORY_NAMES, *ACTION_TRAJECTORIES):
        raise ValueError(f"Unknown trajectory {name!r}")
    if not 0.0 < scale <= 4.0:
        raise ValueError("trajectory scale must be in (0, 4]")
    poses = torch.eye(4, dtype=torch.float32).repeat(num_frames, 1, 1)
    progress = torch.linspace(0.0, 1.0, num_frames)
    if name in {"translate_pos_x", "D"}:
        poses[:, 0, 3] = progress * scale
    elif name in {"translate_neg_x", "A"}:
        poses[:, 0, 3] = -progress * scale
    elif name == "W":
        poses[:, 2, 3] = progress * scale
    elif name == "S":
        poses[:, 2, 3] = -progress * scale
    elif name in {"yaw_pos_y", "J", "L"}:
        direction = -1.0 if name == "J" else 1.0
        angle = progress * math.radians(20.0) * scale * direction
        cosine, sine = angle.cos(), angle.sin()
        poses[:, 0, 0] = cosine
        poses[:, 0, 2] = sine
        poses[:, 2, 0] = -sine
        poses[:, 2, 2] = cosine
    elif name in {"I", "K"}:
        # With +Y up, negative X rotation pitches the forward axis upward.
        direction = -1.0 if name == "I" else 1.0
        angle = progress * math.radians(20.0) * scale * direction
        cosine, sine = angle.cos(), angle.sin()
        poses[:, 1, 1] = cosine
        poses[:, 1, 2] = -sine
        poses[:, 2, 1] = sine
        poses[:, 2, 2] = cosine
    return poses


def _load_validation_batch(data_path: Path, row_index: int, text_len: int) -> dict[str, Any]:
    row = _read_parquet_row(data_path, row_index)
    return collate_rows_from_parquet_schema(
        [row],
        pyarrow_schema_minimax_h3_worldmodel_prope,
        text_len,
        cfg_rate=0.0,
    )


def _decode_video(vae: Any, latents_btchw: torch.Tensor, device: torch.device) -> np.ndarray:
    latents = latents_btchw.permute(0, 2, 1, 3, 4).float().to(device)
    latents = vae.denormalize_latents(latents)
    output = torch.empty(vae.decoded_pixel_shape(latents.shape), dtype=torch.float32, device="cpu")
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        vae.decode_to_pixels(latents, output)
    return (output[0].permute(1, 2, 3, 0).mul_(255).clamp_(0, 255).to(torch.uint8).numpy())


def _write_video(path: Path, frames: np.ndarray, fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, list(frames), fps=fps, format="mp4", macro_block_size=None)


def _write_merged_metadata(path: Path, metadata: dict[str, Any]) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        identity_keys = (
            "checkpoint",
            "validation_data",
            "row_index",
            "seed",
            "num_inference_steps_requested",
            "num_transformer_steps",
            "trajectory_scale",
            "freeze_intrinsics_first_frame",
            "audio",
            "reference_video_omitted",
        )
        mismatches = [key for key in identity_keys if existing.get(key) != metadata.get(key)]
        if mismatches:
            raise RuntimeError(
                f"Refusing to mix incompatible rollout metadata in {path}: {mismatches}"
            )
        for key in (
            "trajectories",
            "real_trajectory_gate_scales",
            "real_trajectory_variants",
        ):
            previous = existing.get(key) or []
            current = metadata.get(key) or []
            merged = list(dict.fromkeys([*previous, *current]))
            metadata[key] = merged or None
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _audit_checkpoint_state(model: Any, device: torch.device) -> None:
    gate_max = torch.zeros((), device=device)
    residual_max = torch.zeros((), device=device)
    lora_b_max = torch.zeros((), device=device)
    action_adapter_max = torch.zeros((), device=device)
    for name, parameter in model.transformer.named_parameters():
        local = parameter.to_local() if hasattr(parameter, "to_local") else parameter
        if name.endswith(".prope_gate"):
            gate_max = torch.maximum(gate_max, local.detach().abs().max().to(device))
        elif name.endswith(".prope_o.weight"):
            residual_max = torch.maximum(residual_max, local.detach().abs().max().to(device))
        elif name.endswith(".lora_B"):
            lora_b_max = torch.maximum(lora_b_max, local.detach().abs().max().to(device))
        elif name == "action_adapter.weight":
            action_adapter_max = torch.maximum(action_adapter_max, local.detach().abs().max().to(device))
    dist.all_reduce(gate_max, op=dist.ReduceOp.MAX)
    dist.all_reduce(residual_max, op=dist.ReduceOp.MAX)
    dist.all_reduce(lora_b_max, op=dist.ReduceOp.MAX)
    dist.all_reduce(action_adapter_max, op=dist.ReduceOp.MAX)
    lora_missing = bool(model.train_lora) and lora_b_max.item() == 0.0
    prope_mode = model.transformer.config.arch_config.prope_mode
    gate_missing = prope_mode == "gated" and gate_max.item() == 0.0
    residual_missing = prope_mode == "residual" and residual_max.item() == 0.0
    if gate_missing or residual_missing or lora_missing:
        raise RuntimeError(
            "DCP checkpoint did not restore the required PRoPE/LoRA weights: "
            f"gate_max={gate_max.item():.6g}, residual_max={residual_max.item():.6g}, "
            f"lora_b_max={lora_b_max.item():.6g}"
        )
    if model.camera_conditioning == "hybrid" and action_adapter_max.item() == 0.0:
        raise RuntimeError("DCP checkpoint did not restore a trained hybrid action adapter")
    if get_world_rank() == 0:
        print(
            f"checkpoint audit: gate_max={gate_max.item():.6g}, "
            f"residual_max={residual_max.item():.6g}, "
            f"lora_b_max={lora_b_max.item():.6g}, "
            f"action_adapter_max={action_adapter_max.item():.6g}",
            flush=True,
        )


@torch.no_grad()
def _scale_prope_gates(model: Any, scale: float) -> float | None:
    if scale <= 0.0:
        raise ValueError(f"PRoPE gate scale must be positive, got {scale}")
    prope_mode = model.transformer.config.arch_config.prope_mode
    if prope_mode == "exact":
        if scale != 1.0:
            raise ValueError("Exact PRoPE has no learnable gate and only supports gate scale 1")
        return None
    if prope_mode == "residual":
        scaled_max = 0.0
        for name, parameter in model.transformer.named_parameters():
            if not name.endswith(".prope_o.weight"):
                continue
            local = parameter.to_local() if hasattr(parameter, "to_local") else parameter
            local.mul_(float(scale))
            scaled_max = max(scaled_max, float(local.detach().abs().max()))
        if scaled_max == 0.0:
            raise RuntimeError("No non-zero residual PRoPE outputs were available to scale")
        return scaled_max
    scaled_max = 0.0
    for name, parameter in model.transformer.named_parameters():
        if not name.endswith(".prope_gate"):
            continue
        local = parameter.to_local() if hasattr(parameter, "to_local") else parameter
        local.mul_(float(scale))
        scaled_max = max(scaled_max, float(local.detach().abs().max()))
    if scaled_max == 0.0:
        raise RuntimeError("No non-zero PRoPE gates were available to scale")
    return scaled_max


def _apply_camera_ablation(model: Any, ablation: str) -> None:
    """Isolate the fixed PRoPE and learned adapter camera pathways.

    A constant PRoPE matrix is an exact global basis change, so its Q/K/V
    transform cancels with the output transform.  The adapter still receives
    the real trajectory.  Conversely, zeroing its linear weight leaves the
    checkpoint's PRoPE path untouched.
    """
    if ablation == "none":
        return
    if ablation == "both":
        _apply_camera_ablation(model, "adapter")
        _apply_camera_ablation(model, "prope")
        return
    if ablation == "adapter":
        adapter = model.transformer.action_adapter
        if adapter is None:
            raise ValueError("This model has no action adapter to ablate")
        with torch.no_grad():
            adapter.weight.zero_()
        return
    if ablation == "prope":
        if model.transformer.config.arch_config.prope_mode == "residual":
            with torch.no_grad():
                for name, parameter in model.transformer.named_parameters():
                    if name.endswith(".prope_o.weight"):
                        local = parameter.to_local() if hasattr(parameter, "to_local") else parameter
                        local.zero_()
            return
        original = model._packed_prope_matrices

        def reference_camera_only(layout: Any, camera_to_world: torch.Tensor,
                                  camera_intrinsics: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return original(
                layout,
                camera_to_world[:, :1].expand_as(camera_to_world),
                camera_intrinsics[:, :1].expand_as(camera_intrinsics),
            )

        model._packed_prope_matrices = reference_camera_only
        return
    raise ValueError(f"Unknown camera ablation {ablation!r}")


@torch.no_grad()
def run(args: argparse.Namespace) -> None:
    cfg = load_run_config(args.config)
    tc = cfg.training
    maybe_init_distributed_environment_and_model_parallel(
        tc.distributed.tp_size,
        tc.distributed.sp_size,
    )
    if get_sp_world_size() != 2:
        raise ValueError("This evaluator expects the training sp_size=2")

    world_size = dist.get_world_size()
    # A training DCP may have been written with more HSDP shards than this
    # inference lane has GPUs. DCP reshares model tensors at load time; build
    # the destination model using the available world size instead of
    # rejecting an otherwise loadable checkpoint (e.g. 8-GPU train -> 4-GPU
    # inference).
    shard_dim = min(tc.distributed.hsdp_shard_dim, world_size)
    if world_size % shard_dim != 0:
        raise ValueError(f"world_size={world_size} must be divisible by HSDP shard_dim={shard_dim}")
    tc.distributed.num_gpus = world_size
    tc.distributed.hsdp_shard_dim = shard_dim
    tc.distributed.hsdp_replicate_dim = world_size // shard_dim

    rank = get_world_rank()
    sp_group = get_sp_group()
    trajectory_index = rank // get_sp_world_size()
    gate_scales = args.real_trajectory_gate_scales
    real_variants = args.real_trajectory_variants
    synthetic_trajectories = args.trajectories or TRAJECTORY_NAMES
    if gate_scales is not None and real_variants is not None:
        raise ValueError("Choose either real trajectory gate scales or real trajectory variants")
    if args.trajectories is not None and (gate_scales is not None or real_variants is not None):
        raise ValueError("Synthetic trajectories cannot be combined with real-trajectory variants")
    num_sp_groups = world_size // get_sp_world_size()
    case_manifest = _read_case_manifest(Path(args.case_manifest), num_sp_groups) if args.case_manifest else None
    if case_manifest is not None and (args.trajectories is not None or gate_scales is not None or real_variants is not None):
        raise ValueError("Case manifest cannot be combined with trajectory or real-trajectory sweeps")
    if case_manifest is None and gate_scales is None and real_variants is None and num_sp_groups != len(synthetic_trajectories):
        raise ValueError(f"Expected {num_sp_groups} synthetic trajectories")
    if gate_scales is not None and len(gate_scales) != num_sp_groups:
        raise ValueError(
            "Real-trajectory gate evaluation needs one scale per SP group: "
            f"got {len(gate_scales)} scales for {num_sp_groups} groups"
        )
    if real_variants is not None and len(real_variants) != num_sp_groups:
        raise ValueError(
            "Real-trajectory counterfactual evaluation needs one variant per SP group: "
            f"got {len(real_variants)} variants for {num_sp_groups} groups"
        )
    gate_scale = float(gate_scales[trajectory_index]) if gate_scales is not None else 1.0
    case = case_manifest[trajectory_index] if case_manifest is not None else None
    if case is not None:
        trajectory_name = case["trajectory"]
    elif gate_scales is not None:
        trajectory_name = f"real_gate_{gate_scale:g}x"
    elif real_variants is not None:
        trajectory_name = f"real_{real_variants[trajectory_index]}"
    else:
        trajectory_name = synthetic_trajectories[trajectory_index]
    device = get_local_torch_device()
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = False

    model = instantiate(cfg.models["student"], training_config=tc)
    state = {
        "roles.student.transformer": ModelWrapper(model.transformer)
    }
    dcp.load(state, checkpoint_id=str(Path(args.checkpoint) / "dcp"))
    _audit_checkpoint_state(model, device)
    _apply_camera_ablation(model, args.camera_ablation)
    scaled_gate_max = _scale_prope_gates(model, gate_scale)
    if sp_group.is_first_rank:
        prope_status = (
            "exact_prope"
            if scaled_gate_max is None
            else f"local_scaled_gate_max={scaled_gate_max:.6g}"
        )
        print(
            f"rank={rank} trajectory={trajectory_name} gate_scale={gate_scale:g} "
            f"{prope_status}",
            flush=True,
        )
    model.transformer.eval()
    model.sp_group = sp_group

    text_len = int(tc.pipeline_config.text_encoder_configs[0].arch_config.text_len)
    data_path = Path(case["data_path"] if case is not None else args.data_path)
    row_index = int(case["row_index"] if case is not None else args.row_index)
    seed = int(case.get("seed", args.seed)) if case is not None else args.seed
    raw_batch = _load_validation_batch(data_path, row_index, text_len)
    num_latent_frames = int(raw_batch["vae_latent"].shape[2])
    if args.freeze_intrinsics_first_frame:
        first_intrinsics = raw_batch["camera_intrinsics"][:, :1]
        raw_batch["camera_intrinsics"] = first_intrinsics.expand(-1, num_latent_frames, -1, -1).clone()
    if gate_scales is None and real_variants is None and trajectory_name != "real":
        camera_to_world = make_camera_trajectory(
            trajectory_name, num_latent_frames, scale=args.trajectory_scale
        )[None]
        raw_batch["camera_to_world"] = camera_to_world
    elif real_variants is not None:
        camera_to_world = make_camera_trajectory_variant(
            raw_batch["camera_to_world"],
            real_variants[trajectory_index],
            seed=seed,
        )
        raw_batch["camera_to_world"] = camera_to_world
    else:
        camera_to_world = raw_batch["camera_to_world"].clone()

    condition_generator = torch.Generator(device=device).manual_seed(seed)
    prepared = model.prepare_batch(raw_batch, generator=condition_generator, latents_source="data")
    if prepared.latents is None or prepared.timesteps is None:
        raise RuntimeError("Prepared world-model batch is incomplete")

    output_dir = Path(args.output_dir) / Path(args.checkpoint).name
    if case is not None:
        output_dir /= case["case_id"]
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.evaluate_flow_loss:
        if trajectory_name != "real" or prepared.noisy_model_input is None or prepared.noise is None:
            raise ValueError("Flow validation requires a real-GT case with an encoded target latent")
        noisy = prepared.noisy_model_input.permute(0, 2, 1, 3, 4)
        correct_prediction = model.predict_noise(noisy, prepared.timesteps, prepared, conditional=True)
        if not isinstance(correct_prediction, torch.Tensor):
            raise TypeError("Flow validation expects one video prediction tensor")
        if tc.model.precondition_outputs:
            if prepared.sigmas is None:
                raise RuntimeError("Preconditioned validation requires sigmas")
            correct_target = prepared.latents.float()
            correct_prediction_for_loss = noisy - correct_prediction * prepared.sigmas
        else:
            correct_target = prepared.noise.permute(0, 2, 1, 3, 4).float() - prepared.latents.float()
            correct_prediction_for_loss = correct_prediction
        correct_loss = torch.nn.functional.mse_loss(correct_prediction_for_loss.float(), correct_target)
        if prepared.input_kwargs is None or "camera_to_world" not in prepared.input_kwargs:
            raise RuntimeError("Flow validation has no camera_to_world condition")
        original_camera = prepared.input_kwargs["camera_to_world"]
        prepared.input_kwargs["camera_to_world"] = original_camera[:, :1].expand_as(original_camera).clone()
        try:
            static_prediction = model.predict_noise(noisy, prepared.timesteps, prepared, conditional=True)
        finally:
            prepared.input_kwargs["camera_to_world"] = original_camera
        if not isinstance(static_prediction, torch.Tensor):
            raise TypeError("Flow validation expects one static-camera prediction tensor")
        if tc.model.precondition_outputs:
            static_prediction_for_loss = noisy - static_prediction * prepared.sigmas
        else:
            static_prediction_for_loss = static_prediction
        static_loss = torch.nn.functional.mse_loss(static_prediction_for_loss.float(), correct_target)
        if sp_group.is_first_rank:
            flow_metrics = {
                "case_id": case["case_id"],
                "seed": seed,
                "checkpoint": str(Path(args.checkpoint).resolve()),
                "evaluation_type": "single_fixed_noise_gt_flow_mse_not_dataset_validation_callback",
                "flow_loss_real_camera": float(correct_loss),
                "flow_loss_static_camera": float(static_loss),
                "real_minus_static_loss": float(correct_loss - static_loss),
                "camera_prediction_rms_response": float(
                    (correct_prediction.float() - static_prediction.float()).square().mean().sqrt()
                ),
                "precondition_outputs": bool(tc.model.precondition_outputs),
            }
            (output_dir / "flow_loss.json").write_text(json.dumps(flow_metrics, indent=2) + "\n", encoding="utf-8")
            print(f"flow validation: {case['case_id']} {flow_metrics}", flush=True)

    noise_generator = torch.Generator(device=device).manual_seed(seed + 1)
    latents = torch.randn(
        prepared.latents.shape,
        generator=noise_generator,
        device=device,
        dtype=torch.bfloat16,
    )
    scheduler = MiniMaxH3Scheduler(shift=12.0)
    scheduler.set_timesteps(args.num_inference_steps, device=device)
    if scheduler.timesteps is None:
        raise RuntimeError("MiniMax-H3 scheduler did not create timesteps")

    for step_index, timestep in enumerate(scheduler.timesteps):
        prepared.timesteps = timestep.reshape(1)
        prediction = model.predict_noise(latents, prepared.timesteps, prepared, conditional=True)
        if not isinstance(prediction, torch.Tensor):
            raise TypeError("World-model inference expects one video prediction tensor")
        # Training exposes noise-minus-clean, while the H3 scheduler consumes
        # the transformer's clean-minus-noise velocity.
        latents = scheduler.step(-prediction, timestep, latents, return_dict=False)[0]
        if sp_group.is_first_rank:
            print(
                f"rank={rank} trajectory={trajectory_name} "
                f"step={step_index + 1}/{len(scheduler.timesteps)}",
                flush=True,
            )

    latent_path = output_dir / f"{trajectory_name}.pt"
    if sp_group.is_first_rank:
        torch.save(latents.detach().cpu().to(torch.float16), latent_path)
        np.save(output_dir / f"{trajectory_name}_camera_to_world.npy", camera_to_world.numpy())

    # Release the 33B transformer before loading the VAE on one rank per SP
    # group. All four trajectories then decode concurrently on four GPUs.
    dist.barrier()
    del state, model, prepared, scheduler
    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier()

    if args.skip_decode:
        if case is not None or rank == 0:
            print(f"Saved latent-only smoke outputs under {output_dir}", flush=True)
    elif sp_group.is_first_rank:
        vae = load_module_from_path(
            model_path=str(tc.model_path),
            module_type="vae",
            training_config=tc,
        )
        vae.eval()
        frames = _decode_video(vae, torch.load(latent_path, map_location="cpu", weights_only=True), device)
        _write_video(output_dir / f"{trajectory_name}.mp4", frames, args.fps)
        if case is not None or rank == 0:
            first = raw_batch["first_frame_latent"].permute(0, 2, 1, 3, 4)
            first_frame = _decode_video(vae, first, device)[0]
            imageio.imwrite(output_dir / "condition_first_frame.png", first_frame)
            if not args.omit_reference_video:
                ground_truth = raw_batch["vae_latent"].permute(0, 2, 1, 3, 4)
                _write_video(output_dir / "ground_truth.mp4", _decode_video(vae, ground_truth, device), args.fps)
            metadata = {
                "checkpoint": str(Path(args.checkpoint).resolve()),
                "validation_data": str(data_path.resolve()),
                "row_index": row_index,
                "case_id": case["case_id"] if case is not None else None,
                "sample": raw_batch["info_list"][0],
                "seed": seed,
                "num_inference_steps_requested": args.num_inference_steps,
                "num_transformer_steps": args.num_inference_steps - 1,
                "trajectory_scale": args.trajectory_scale,
                "freeze_intrinsics_first_frame": args.freeze_intrinsics_first_frame,
                "trajectories": (
                    [f"real_gate_{float(scale):g}x" for scale in gate_scales]
                    if gate_scales is not None
                    else (
                        [f"real_{variant}" for variant in real_variants]
                        if real_variants is not None
                        else [trajectory_name] if case is not None else list(synthetic_trajectories)
                    )
                ),
                "real_trajectory_gate_scales": gate_scales,
                "real_trajectory_variants": real_variants,
                "camera_ablation": args.camera_ablation,
                "audio": False,
                "reference_video_omitted": args.omit_reference_video,
            }
            _write_merged_metadata(output_dir / "metadata.json", metadata)
        del vae
    dist.barrier()
    dist.destroy_process_group()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-path", help="Parquet directory for a single-frame trajectory sweep")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--case-manifest", help="JSON list with one distinct first-frame/action case per SP group")
    parser.add_argument("--row-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--trajectories", choices=(*TRAJECTORY_NAMES, *ACTION_TRAJECTORIES), nargs="+")
    parser.add_argument("--trajectory-scale", type=float, default=1.0,
                        help="Multiply the generated camera displacement/angle (0, 4].")
    parser.add_argument("--freeze-intrinsics-first-frame", action="store_true",
                        help="Use the first-frame camera K throughout a synthetic rollout.")
    parser.add_argument(
        "--real-trajectory-gate-scales",
        type=float,
        nargs="+",
        default=None,
        metavar="SCALE",
        help="Use the sample's real c2w on all four SP groups and apply one PRoPE gate scale per group.",
    )
    parser.add_argument(
        "--real-trajectory-variants",
        choices=REAL_TRAJECTORY_VARIANTS,
        nargs="+",
        default=None,
        metavar="VARIANT",
        help="Evaluate correct/static/reverse/shuffled variants of the sample's real c2w path.",
    )
    parser.add_argument("--skip-decode", action="store_true")
    parser.add_argument("--omit-reference-video", action="store_true",
                        help="Do not decode the reference latent when it only carries inference shape.")
    parser.add_argument("--evaluate-flow-loss", action="store_true",
                        help="For a real-GT case, score fixed-noise flow MSE with real and static camera paths.")
    parser.add_argument("--camera-ablation", choices=("none", "prope", "adapter", "both"), default="none",
                        help="Diagnostic only: neutralize PRoPE or the learned action adapter after checkpoint load.")
    args = parser.parse_args()
    if (args.case_manifest is None) == (args.data_path is None):
        parser.error("Provide exactly one of --data-path or --case-manifest")
    return args


if __name__ == "__main__":
    run(parse_args())
