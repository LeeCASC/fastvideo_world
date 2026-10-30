# SPDX-License-Identifier: Apache-2.0
"""Resumable two-phase preprocessing for MiniMax-H3 VWM Stage-0 training."""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Sequence
from io import BytesIO
import json
import os
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from fastvideo.configs.pipelines.minimax_h3 import MiniMaxH3PipelineConfig
from fastvideo.dataset.dataloader.parquet_io import records_to_table
from fastvideo.dataset.dataloader.record_schema import (
    minimax_h3_worldmodel_conditioned_record_creator,
    minimax_h3_worldmodel_prope_conditioned_record_creator,
    minimax_h3_worldmodel_prope_record_from_additive,
    minimax_h3_worldmodel_prope_vae_record_creator,
    minimax_h3_worldmodel_vae_record_creator,
)
from fastvideo.dataset.dataloader.schema import (
    pyarrow_schema_minimax_h3_worldmodel,
    pyarrow_schema_minimax_h3_worldmodel_prope,
    pyarrow_schema_minimax_h3_worldmodel_prope_vae,
    pyarrow_schema_minimax_h3_worldmodel_vae,
)
from fastvideo.dataset.vwm_h3 import (
    H3VWMDataConfig,
    H3VWMRawDataset,
    canonicalize_source_intrinsics,
    h3_latent_source_frame_indices,
    rebase_camera_to_first_frame,
)
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.models.loader.component_loader import PipelineComponentLoader
from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_conditioning import MiniMaxH3ConditioningStage
from fastvideo.pipelines.preprocess.minimax_h3_worldmodel.legacy_cache import LegacyH3WorldModelCache
from fastvideo.pipelines.preprocess.minimax_h3_worldmodel.encoding import (
    encode_h3_worldmodel_condition,
    encode_h3_worldmodel_video,
)
from vwm_datasets.transforms import transform_intrinsic_for_resize_center_crop


def iter_worker_chunks(
    total_samples: int,
    *,
    worker_index: int,
    num_workers: int,
    samples_per_shard: int,
    max_samples: int | None = None,
) -> Iterator[tuple[int, list[int]]]:
    """Yield deterministic rank-local shard indices for resume-safe outputs."""
    if total_samples < 0:
        raise ValueError("total_samples must be non-negative")
    if num_workers <= 0 or not 0 <= worker_index < num_workers:
        raise ValueError("worker_index must be in [0, num_workers)")
    if samples_per_shard <= 0:
        raise ValueError("samples_per_shard must be positive")
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be positive when provided")
    selected_count = min(total_samples, max_samples) if max_samples is not None else total_samples
    local_indices = list(range(worker_index, selected_count, num_workers))
    for shard_index, start in enumerate(range(0, len(local_indices), samples_per_shard)):
        yield shard_index, local_indices[start:start + samples_per_shard]


def _atomic_write_records(records: list[dict[str, Any]], schema: pa.Schema, output_path: Path) -> None:
    """Write one deterministic parquet shard without exposing partial files."""
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite an existing preprocessing shard: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.unlink(missing_ok=True)
    table = records_to_table(records, schema)
    try:
        pq.write_table(table, temporary_path, compression="zstd")
        os.replace(temporary_path, output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _read_ids(path: Path) -> list[str]:
    return [str(value) for value in pq.read_table(path, columns=["id"])["id"].to_pylist()]


def _reuse_completed_shard(path: Path, expected_ids: Sequence[str], schema: pa.Schema) -> bool:
    """Return true only when an existing shard exactly matches its fixed plan."""
    if not path.is_file():
        return False
    actual_schema = pq.read_schema(path)
    if not actual_schema.equals(schema, check_metadata=False):
        raise ValueError(f"Existing shard has an incompatible schema: {path}")
    actual_ids = _read_ids(path)
    if actual_ids != list(expected_ids):
        raise ValueError(f"Existing shard has unexpected sample IDs: {path}")
    return True


def _record_identity(dataset: H3VWMRawDataset, index: int) -> str:
    source_index = dataset.selected_indices[index]
    record = dataset.vwm_dataset.records[source_index]
    return f"{record.source_name}/{record.sample_id}"


def _encode_png(first_frame_chw: torch.Tensor) -> bytes:
    array = first_frame_chw.permute(1, 2, 0).contiguous().cpu().numpy()
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("first frame must be uint8 RGB [3,H,W]")
    buffer = BytesIO()
    Image.fromarray(array).save(buffer, format="PNG", compress_level=1)
    return buffer.getvalue()


def _load_model_index(model_path: Path) -> dict[str, Any]:
    index_path = model_path / "model_index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"MiniMax-H3 model index is missing: {index_path}")
    model_index = json.loads(index_path.read_text(encoding="utf-8"))
    if not isinstance(model_index, dict):
        raise ValueError(f"MiniMax-H3 model index must be a mapping: {index_path}")
    return model_index


def _load_component(
    name: str,
    model_path: Path,
    model_index: dict[str, Any],
    fastvideo_args: FastVideoArgs,
) -> Any:
    entry = model_index.get(name)
    if not isinstance(entry, list) or len(entry) < 2:
        raise ValueError(f"MiniMax-H3 model index has no valid {name!r} component")
    component_path = model_path / name
    if not component_path.is_dir():
        raise FileNotFoundError(f"MiniMax-H3 component is missing: {component_path}")
    return PipelineComponentLoader.load_module(
        module_name=name,
        component_model_path=str(component_path),
        transformers_or_diffusers=str(entry[0]),
        fastvideo_args=fastvideo_args,
    )


def _initialize_worker(device_index: int, master_port: int) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("MiniMax-H3 preprocessing requires CUDA")
    if not 0 <= device_index < torch.cuda.device_count():
        raise ValueError(f"device_index {device_index} is outside the visible CUDA device range")
    # Each preprocessing worker owns one independent one-rank process group.
    # Explicit assignment prevents inherited torchrun variables from joining
    # otherwise unrelated workers.
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["LOCAL_RANK"] = str(device_index)
    from fastvideo.distributed import maybe_init_distributed_environment_and_model_parallel

    maybe_init_distributed_environment_and_model_parallel(1, 1)
    return torch.device("cuda", device_index)


def _fastvideo_args(model_path: Path) -> FastVideoArgs:
    return FastVideoArgs(
        model_path=str(model_path),
        pipeline_config=MiniMaxH3PipelineConfig(),
        num_gpus=1,
        tp_size=1,
        sp_size=1,
        hsdp_shard_dim=1,
        use_fsdp_inference=False,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=False,
    )


def _worldmodel_schemas(conditioning: Literal["additive", "prope"]) -> tuple[pa.Schema, pa.Schema]:
    if conditioning == "additive":
        return pyarrow_schema_minimax_h3_worldmodel_vae, pyarrow_schema_minimax_h3_worldmodel
    if conditioning == "prope":
        return pyarrow_schema_minimax_h3_worldmodel_prope_vae, pyarrow_schema_minimax_h3_worldmodel_prope
    raise ValueError(f"Unsupported H3 camera conditioning: {conditioning}")


def rebase_prope_vae_record_camera(record: dict[str, Any]) -> dict[str, Any]:
    """Repair older VAE caches whose camera origin precedes the sampled clip."""
    if record["camera_to_world_dtype"] != "float32":
        raise ValueError(f"camera_to_world must be float32 for {record['id']}")
    camera = np.frombuffer(record["camera_to_world_bytes"], dtype=np.float32).reshape(record["camera_to_world_shape"])
    if np.allclose(camera[0], np.eye(4), atol=1.0e-5):
        return record
    repaired = dict(record)
    repaired["camera_to_world_bytes"] = rebase_camera_to_first_frame(camera).tobytes()
    return repaired


def repair_camera_parquet_shard(path: Path) -> tuple[int, int]:
    """Atomically repair legacy camera origins while preserving all other fields."""
    camera_fields = ["id", "camera_to_world_bytes", "camera_to_world_shape", "camera_to_world_dtype"]
    summary = pq.read_table(path, columns=camera_fields).to_pylist()
    needs_repair = []
    for row in summary:
        if row["camera_to_world_dtype"] != "float32":
            raise ValueError(f"camera_to_world must be float32 for {row['id']}")
        camera = np.frombuffer(row["camera_to_world_bytes"], dtype=np.float32).reshape(row["camera_to_world_shape"])
        needs_repair.append(not np.allclose(camera[0], np.eye(4), atol=1.0e-5))
    if not any(needs_repair):
        return len(summary), 0

    table = pq.read_table(path)
    records = table.to_pylist()
    if [row["id"] for row in records] != [row["id"] for row in summary]:
        raise ValueError(f"sample IDs changed during camera repair: {path}")
    for index, required in enumerate(needs_repair):
        if required:
            records[index] = rebase_prope_vae_record_camera(records[index])
    temporary = path.with_suffix(f".parquet.tmp.{os.getpid()}")
    try:
        pq.write_table(pa.Table.from_pylist(records, schema=table.schema), temporary, compression="zstd")
        if not pq.read_schema(temporary).equals(table.schema, check_metadata=False):
            raise ValueError(f"camera repair changed schema: {path}")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return len(summary), sum(needs_repair)


def preprocess_vae(
    *,
    manifest_path: Path,
    model_path: Path,
    output_dir: Path,
    split: Literal["train", "validation", "all"],
    worker_index: int,
    num_workers: int,
    device_index: int,
    samples_per_shard: int,
    max_samples: int | None,
    validation_fraction: float,
    min_source_duration_seconds: float,
    split_seed: int,
    master_port: int,
    conditioning: Literal["additive", "prope"] = "additive",
) -> None:
    """Encode video/keyframe latents and actions without loading Qwen3-VL."""
    device = _initialize_worker(device_index, master_port)
    model_path = model_path.resolve()
    model_index = _load_model_index(model_path)
    fastvideo_args = _fastvideo_args(model_path)
    vae = _load_component("vae", model_path, model_index, fastvideo_args)
    dataset = H3VWMRawDataset(
        H3VWMDataConfig(
            manifest_path=manifest_path,
            split=split,
            validation_fraction=validation_fraction,
            min_source_duration_seconds=min_source_duration_seconds,
            split_seed=split_seed,
        ))
    vae_schema, _ = _worldmodel_schemas(conditioning)
    worker_dir = output_dir / f"worker_{worker_index:03d}"
    for shard_index, sample_indices in iter_worker_chunks(
            len(dataset),
            worker_index=worker_index,
            num_workers=num_workers,
            samples_per_shard=samples_per_shard,
            max_samples=max_samples,
    ):
        output_path = worker_dir / f"data_{shard_index:06d}.parquet"
        expected_ids = [_record_identity(dataset, index) for index in sample_indices]
        if _reuse_completed_shard(output_path, expected_ids, vae_schema):
            print(f"worker {worker_index}: reuse {output_path}", flush=True)
            continue
        records: list[dict[str, Any]] = []
        for index in sample_indices:
            sample = dataset[index]
            video = sample["video"]
            assert isinstance(video, torch.Tensor)
            vae_latent, first_frame_latent = encode_h3_worldmodel_video(vae, video, device=device)
            creator = (minimax_h3_worldmodel_prope_vae_record_creator
                       if conditioning == "prope" else minimax_h3_worldmodel_vae_record_creator)
            pose_kwargs = ({
                "camera_to_world": sample["latent_camera_to_world"].numpy(),
                "camera_intrinsics": sample["latent_intrinsics"].numpy(),
            } if conditioning == "prope" else {})
            records.append(
                creator(
                    sample_id=str(sample["id"]),
                    source_name=str(sample["source_name"]),
                    caption=str(sample["caption"]),
                    vae_latent=vae_latent.numpy(),
                    first_frame_latent=first_frame_latent.numpy(),
                    first_frame_png=_encode_png(video[:, 0]),
                    action_continuous=sample["action_continuous"].numpy(),
                    action_valid_mask=sample["action_valid_mask"].numpy(),
                    source_frame_indices=sample["source_frame_indices"].numpy(),
                    width=int(sample["width"]),
                    height=int(sample["height"]),
                    fps=float(sample["fps"]),
                    source_fps=float(sample["source_fps"]),
                    **pose_kwargs,
                ))
        _atomic_write_records(records, vae_schema, output_path)
        print(f"worker {worker_index}: wrote {len(records)} rows to {output_path}", flush=True)


def preprocess_condition(
    *,
    input_dir: Path,
    model_path: Path,
    output_dir: Path,
    worker_index: int,
    num_workers: int,
    device_index: int,
    master_port: int,
    conditioning: Literal["additive", "prope"] = "additive",
) -> None:
    """Add native first-frame Qwen3-VL conditioning to VAE shards."""
    _initialize_worker(device_index, master_port)
    model_path = model_path.resolve()
    model_index = _load_model_index(model_path)
    fastvideo_args = _fastvideo_args(model_path)
    tokenizer = _load_component("tokenizer", model_path, model_index, fastvideo_args)
    processor = _load_component("processor", model_path, model_index, fastvideo_args)
    conditioner = _load_component("text_encoder", model_path, model_index, fastvideo_args)
    stage = MiniMaxH3ConditioningStage(
        conditioner=conditioner,
        tokenizer=tokenizer,
        processor=processor,
    )
    vae_schema, final_schema = _worldmodel_schemas(conditioning)
    input_paths = sorted(input_dir.rglob("*.parquet"))
    if not input_paths:
        raise FileNotFoundError(f"No VAE parquet shards found under {input_dir}")
    for input_path in input_paths[worker_index::num_workers]:
        relative_path = input_path.relative_to(input_dir)
        output_path = output_dir / relative_path
        if not pq.read_schema(input_path).equals(
                vae_schema,
                check_metadata=False,
        ):
            raise ValueError(f"Unexpected VAE schema in {input_path}")
        expected_ids = _read_ids(input_path)
        if _reuse_completed_shard(output_path, expected_ids, final_schema):
            if conditioning == "prope":
                _, repaired = repair_camera_parquet_shard(output_path)
                if repaired:
                    print(f"worker {worker_index}: repaired {repaired} cameras in {output_path}", flush=True)
            print(f"worker {worker_index}: reuse {output_path}", flush=True)
            continue
        table = pq.read_table(input_path, schema=vae_schema)
        records: list[dict[str, Any]] = []
        for vae_record in table.to_pylist():
            if conditioning == "prope":
                vae_record = rebase_prope_vae_record_camera(vae_record)
            image = Image.open(BytesIO(vae_record["first_frame_png"])).convert("RGB")
            embedding, tags = encode_h3_worldmodel_condition(
                stage,
                fastvideo_args,
                caption=str(vae_record["caption"]),
                first_frame=image,
            )
            creator = (minimax_h3_worldmodel_prope_conditioned_record_creator
                       if conditioning == "prope" else minimax_h3_worldmodel_conditioned_record_creator)
            records.append(creator(
                vae_record,
                text_embedding=embedding.numpy(),
                text_token_tags=tags.numpy(),
            ))
        _atomic_write_records(records, final_schema, output_path)
        print(f"worker {worker_index}: wrote {len(records)} rows to {output_path}", flush=True)


def augment_prope_geometry(
    *,
    input_dir: Path,
    manifest_path: Path,
    output_dir: Path,
    worker_index: int,
    num_workers: int,
) -> None:
    """Reuse additive H3 encodings and append latent-aligned VWM pose/K."""
    dataset = H3VWMRawDataset(H3VWMDataConfig(manifest_path=manifest_path, split="all"))
    source_indices_by_id = {
        f"{record.source_name}/{record.sample_id}": index
        for index, record in enumerate(dataset.vwm_dataset.records)
    }
    input_paths = sorted(input_dir.rglob("*.parquet"))
    if not input_paths:
        raise FileNotFoundError(f"No additive parquet shards found under {input_dir}")
    latent_frame_indices = h3_latent_source_frame_indices(dataset.config.num_frames)
    for input_path in input_paths[worker_index::num_workers]:
        relative_path = input_path.relative_to(input_dir)
        output_path = output_dir / relative_path
        if not pq.read_schema(input_path).equals(pyarrow_schema_minimax_h3_worldmodel, check_metadata=False):
            raise ValueError(f"Unexpected additive schema in {input_path}")
        expected_ids = _read_ids(input_path)
        if _reuse_completed_shard(output_path, expected_ids, pyarrow_schema_minimax_h3_worldmodel_prope):
            print(f"worker {worker_index}: reuse {output_path}", flush=True)
            continue
        table = pq.read_table(input_path, schema=pyarrow_schema_minimax_h3_worldmodel)
        records: list[dict[str, Any]] = []
        for additive_record in table.to_pylist():
            sample_id = str(additive_record["id"])
            if sample_id not in source_indices_by_id:
                raise KeyError(f"Additive sample is absent from the VWM manifest: {sample_id}")
            source_index = source_indices_by_id[sample_id]
            clip = dataset.vwm_dataset[source_index]
            selected_source_indices = np.asarray(additive_record["source_frame_indices"], dtype=np.int64)
            if selected_source_indices.shape != (dataset.config.num_frames, ):
                raise ValueError(f"Unexpected source-frame plan for {sample_id}: {selected_source_indices.shape}")
            camera_to_world = clip.camera_to_world[selected_source_indices][latent_frame_indices]
            if clip.intrinsics.shape == (3, 3):
                source_intrinsics = np.broadcast_to(
                    clip.intrinsics,
                    (len(latent_frame_indices), 3, 3),
                )
            else:
                source_intrinsics = clip.intrinsics[selected_source_indices][latent_frame_indices]
            source_intrinsics = canonicalize_source_intrinsics(
                source_intrinsics,
                source_name=clip.source_name,
                image_width=clip.width,
                image_height=clip.height,
            )
            camera_intrinsics = np.stack([
                transform_intrinsic_for_resize_center_crop(
                    value,
                    src_size=(clip.height, clip.width),
                    dst_size=(int(additive_record["height"]), int(additive_record["width"])),
                ) for value in source_intrinsics
            ])
            records.append(
                minimax_h3_worldmodel_prope_record_from_additive(
                    additive_record,
                    camera_to_world=camera_to_world,
                    camera_intrinsics=camera_intrinsics,
                ))
        _atomic_write_records(records, pyarrow_schema_minimax_h3_worldmodel_prope, output_path)
        print(f"worker {worker_index}: augmented {len(records)} rows to {output_path}", flush=True)


def _replace_float32_tensor(record: dict[str, Any], name: str, value: np.ndarray) -> None:
    array = np.ascontiguousarray(value, dtype=np.float32)
    record[f"{name}_bytes"] = array.tobytes()
    record[f"{name}_shape"] = list(array.shape)
    record[f"{name}_dtype"] = "float32"


def static_control_record(
    motion_record: dict[str, Any],
    *,
    static_vae_latent: np.ndarray,
    static_first_frame_latent: np.ndarray,
) -> dict[str, Any]:
    """Pair a zero camera trajectory with a genuinely static video target."""
    if set(motion_record) != set(pyarrow_schema_minimax_h3_worldmodel_prope.names):
        raise ValueError("motion_record does not match the H3 PRoPE world-model schema")
    latent_shape = tuple(int(value) for value in motion_record["vae_latent_shape"])
    first_shape = tuple(int(value) for value in motion_record["first_frame_latent_shape"])
    static_vae_latent = np.asarray(static_vae_latent, dtype=np.float32)
    static_first_frame_latent = np.asarray(static_first_frame_latent, dtype=np.float32)
    if static_vae_latent.shape != latent_shape or static_first_frame_latent.shape != first_shape:
        raise ValueError("static H3 latents must match the moving record; "
                         f"expected {latent_shape} and {first_shape}, got "
                         f"{static_vae_latent.shape} and {static_first_frame_latent.shape}")
    frames = latent_shape[1]
    static = dict(motion_record)
    static["id"] = f"{motion_record['id']}#static"
    _replace_float32_tensor(static, "vae_latent", static_vae_latent)
    _replace_float32_tensor(static, "first_frame_latent", static_first_frame_latent)
    _replace_float32_tensor(static, "action_continuous", np.zeros((frames, 6), dtype=np.float32))

    c2w_shape = tuple(int(value) for value in motion_record["camera_to_world_shape"])
    intrinsics_shape = tuple(int(value) for value in motion_record["camera_intrinsics_shape"])
    camera_to_world = np.frombuffer(motion_record["camera_to_world_bytes"], dtype=np.float32).reshape(c2w_shape)
    camera_intrinsics = np.frombuffer(motion_record["camera_intrinsics_bytes"],
                                      dtype=np.float32).reshape(intrinsics_shape)
    _replace_float32_tensor(static, "camera_to_world", np.repeat(camera_to_world[:1], frames, axis=0))
    _replace_float32_tensor(static, "camera_intrinsics", np.repeat(camera_intrinsics[:1], frames, axis=0))
    return static


def build_static_control_pairs(
    *,
    input_dir: Path,
    manifest_path: Path,
    model_path: Path,
    output_dir: Path,
    sample_ids: Sequence[str],
    device_index: int,
    master_port: int,
) -> None:
    """Build moving/static pairs while reusing all existing conditioning."""
    requested = tuple(dict.fromkeys(str(value) for value in sample_ids))
    if not requested or any(not value for value in requested):
        raise ValueError("sample_ids must contain at least one non-empty ID")
    paths = sorted(input_dir.rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No H3 PRoPE parquet shards found under {input_dir}")

    locations: dict[str, tuple[Path, int]] = {}
    requested_set = set(requested)
    for path in paths:
        ids = _read_ids(path)
        for row_index, sample_id in enumerate(ids):
            if sample_id in requested_set:
                if sample_id in locations:
                    raise ValueError(f"Duplicate requested sample ID: {sample_id}")
                locations[sample_id] = (path, row_index)
    missing = requested_set - set(locations)
    if missing:
        raise KeyError(f"Requested sample IDs are absent from encoded data: {sorted(missing)}")

    device = _initialize_worker(device_index, master_port)
    model_path = model_path.resolve()
    model_index = _load_model_index(model_path)
    vae = _load_component("vae", model_path, model_index, _fastvideo_args(model_path))
    dataset = H3VWMRawDataset(H3VWMDataConfig(manifest_path=manifest_path, split="all"))
    raw_indices = {_record_identity(dataset, index): index for index in range(len(dataset))}

    records: list[dict[str, Any]] = []
    cached_tables: dict[Path, pa.Table] = {}
    for sample_id in requested:
        if sample_id not in raw_indices:
            raise KeyError(f"Requested sample ID is absent from VWM train_v2: {sample_id}")
        path, row_index = locations[sample_id]
        if path not in cached_tables:
            schema = pq.read_schema(path)
            if not schema.equals(pyarrow_schema_minimax_h3_worldmodel_prope, check_metadata=False):
                raise ValueError(f"Unexpected H3 PRoPE schema in {path}")
            cached_tables[path] = pq.read_table(path, schema=pyarrow_schema_minimax_h3_worldmodel_prope)
        motion_record = cached_tables[path].slice(row_index, 1).to_pylist()[0]
        raw_sample = dataset[raw_indices[sample_id]]
        raw_video = raw_sample["video"]
        assert isinstance(raw_video, torch.Tensor)
        static_video = raw_video[:, :1].expand(-1, raw_video.shape[1], -1, -1).contiguous()
        static_latent, static_first = encode_h3_worldmodel_video(vae, static_video, device=device)
        records.extend((
            motion_record,
            static_control_record(
                motion_record,
                static_vae_latent=static_latent.numpy(),
                static_first_frame_latent=static_first.numpy(),
            ),
        ))
        print(f"paired moving/static target for {sample_id}", flush=True)
        del static_video, static_latent, static_first

    output_path = output_dir / "data_000000.parquet"
    _atomic_write_records(records, pyarrow_schema_minimax_h3_worldmodel_prope, output_path)
    print(f"wrote {len(records)} moving/static rows to {output_path}", flush=True)


def preprocess_legacy_cache(
    *,
    metadata_path: Path,
    cache_dir: Path,
    vwm_manifest_path: Path,
    output_dir: Path,
    worker_index: int,
    num_workers: int,
    samples_per_shard: int,
    max_samples: int | None,
) -> None:
    """Convert existing DiffSynth H3 tensors without running VAE/Qwen again."""
    dataset = LegacyH3WorldModelCache(
        metadata_path=metadata_path,
        cache_dir=cache_dir,
        vwm_manifest_path=vwm_manifest_path,
    )
    worker_dir = output_dir / f"worker_{worker_index:03d}"
    for shard_index, sample_indices in iter_worker_chunks(
            len(dataset),
            worker_index=worker_index,
            num_workers=num_workers,
            samples_per_shard=samples_per_shard,
            max_samples=max_samples,
    ):
        output_path = worker_dir / f"data_{shard_index:06d}.parquet"
        expected_ids = [dataset.identity(index) for index in sample_indices]
        if _reuse_completed_shard(
                output_path,
                expected_ids,
                pyarrow_schema_minimax_h3_worldmodel_prope,
        ):
            print(f"worker {worker_index}: reuse {output_path}", flush=True)
            continue
        records = [dataset.record(index) for index in sample_indices]
        _atomic_write_records(
            records,
            pyarrow_schema_minimax_h3_worldmodel_prope,
            output_path,
        )
        print(
            f"worker {worker_index}: converted {len(records)} legacy H3 rows to {output_path}",
            flush=True,
        )


def verify_dataset(
    input_dir: Path,
    *,
    expected_rows: int | None = None,
    conditioning: Literal["additive", "prope"] = "additive",
) -> None:
    """Validate final schemas, row counts, unique IDs, and audio exclusion."""
    paths = sorted(input_dir.rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No final parquet shards found under {input_dir}")
    seen: set[str] = set()
    row_count = 0
    _, final_schema = _worldmodel_schemas(conditioning)
    for path in paths:
        schema = pq.read_schema(path)
        if not schema.equals(final_schema, check_metadata=False):
            raise ValueError(f"Unexpected final schema in {path}")
        if any("audio" in name for name in schema.names):
            raise ValueError(f"Audio field found in {path}")
        ids = _read_ids(path)
        duplicates = seen.intersection(ids)
        if duplicates:
            raise ValueError(f"Duplicate sample IDs found in {path}: {sorted(duplicates)[:3]}")
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate sample IDs occur within {path}")
        if conditioning == "prope":
            rows = pq.read_table(
                path, columns=["id", "camera_to_world_bytes", "camera_to_world_shape",
                               "camera_to_world_dtype", "camera_intrinsics_bytes",
                               "camera_intrinsics_shape", "camera_intrinsics_dtype"]).to_pylist()
            for row in rows:
                if row["camera_to_world_dtype"] != "float32":
                    raise ValueError(f"Camera dtype is not float32 for {row['id']}")
                camera = np.frombuffer(row["camera_to_world_bytes"],
                                       dtype=np.float32).reshape(row["camera_to_world_shape"])
                if not np.allclose(camera[0], np.eye(4), atol=5.0e-3):
                    raise ValueError(f"Camera is not rebased to the clip's first frame for {row['id']}")
                if row["camera_intrinsics_dtype"] != "float32":
                    raise ValueError(f"Camera intrinsics dtype is not float32 for {row['id']}")
                intrinsics = np.frombuffer(row["camera_intrinsics_bytes"],
                                           dtype=np.float32).reshape(row["camera_intrinsics_shape"])
                # Crop-adjusted normalized telephoto K can exceed 10; 32 matches the source-K audit.
                if (intrinsics.shape != (camera.shape[0], 3, 3) or not np.isfinite(intrinsics).all()
                        or np.any(intrinsics[..., 0, 0] <= 0.0) or np.any(intrinsics[..., 1, 1] <= 0.0)
                        or np.max(intrinsics[..., 0, 0]) > 32.0 or np.max(intrinsics[..., 1, 1]) > 32.0
                        or np.max(np.abs(intrinsics[..., :2, 2])) > 2.0):
                    raise ValueError(f"Camera intrinsics are not normalized and valid for {row['id']}")
        seen.update(ids)
        row_count += len(ids)
    if expected_rows is not None and row_count != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, found {row_count}")
    print(f"validated {row_count} audio-free rows across {len(paths)} shards in {input_dir}")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _add_worker_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--num-workers", type=_positive_int, default=1)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--master-port", type=_positive_int, default=29531)


def _add_conditioning_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--conditioning", choices=("additive", "prope"), default="additive")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="phase", required=True)

    vae = subparsers.add_parser("vae", help="encode video VAE and camera actions")
    vae.add_argument("--manifest", type=Path, required=True)
    vae.add_argument("--model-path", type=Path, required=True)
    vae.add_argument("--output-dir", type=Path, required=True)
    vae.add_argument("--split", choices=("train", "validation", "all"), default="train")
    vae.add_argument("--samples-per-shard", type=_positive_int, default=16)
    vae.add_argument("--max-samples", type=_positive_int)
    vae.add_argument("--validation-fraction", type=float, default=0.01)
    vae.add_argument("--min-source-duration-seconds", type=float, default=4.5)
    vae.add_argument("--split-seed", type=int, default=20260828)
    _add_conditioning_argument(vae)
    _add_worker_arguments(vae)

    condition = subparsers.add_parser("condition", help="add Qwen3-VL first-frame conditioning")
    condition.add_argument("--input-dir", type=Path, required=True)
    condition.add_argument("--model-path", type=Path, required=True)
    condition.add_argument("--output-dir", type=Path, required=True)
    _add_conditioning_argument(condition)
    _add_worker_arguments(condition)

    augment = subparsers.add_parser("augment", help="append PRoPE pose/K to existing additive parquet")
    augment.add_argument("--input-dir", type=Path, required=True)
    augment.add_argument("--manifest", type=Path, required=True)
    augment.add_argument("--output-dir", type=Path, required=True)
    augment.add_argument("--worker-index", type=int, default=0)
    augment.add_argument("--num-workers", type=_positive_int, default=1)

    legacy = subparsers.add_parser(
        "legacy-cache",
        help="reuse DiffSynth H3 PTH tensors and rebuild audio-free PRoPE records",
    )
    legacy.add_argument("--metadata", type=Path, required=True)
    legacy.add_argument("--cache-dir", type=Path, required=True)
    legacy.add_argument("--vwm-manifest", type=Path, required=True)
    legacy.add_argument("--output-dir", type=Path, required=True)
    legacy.add_argument("--samples-per-shard", type=_positive_int, default=16)
    legacy.add_argument("--max-samples", type=_positive_int)
    _add_worker_arguments(legacy)

    static_pairs = subparsers.add_parser(
        "static-pairs",
        help="pair selected moving records with VAE-encoded static targets",
    )
    static_pairs.add_argument("--input-dir", type=Path, required=True)
    static_pairs.add_argument("--manifest", type=Path, required=True)
    static_pairs.add_argument("--model-path", type=Path, required=True)
    static_pairs.add_argument("--output-dir", type=Path, required=True)
    static_pairs.add_argument("--sample-id", action="append", required=True)
    static_pairs.add_argument("--device-index", type=int, default=0)
    static_pairs.add_argument("--master-port", type=_positive_int, default=29531)

    verify = subparsers.add_parser("verify", help="verify a completed training parquet directory")
    verify.add_argument("--input-dir", type=Path, required=True)
    verify.add_argument("--expected-rows", type=_positive_int)
    _add_conditioning_argument(verify)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        if args.phase == "vae":
            preprocess_vae(
                manifest_path=args.manifest,
                model_path=args.model_path,
                output_dir=args.output_dir,
                split=args.split,
                worker_index=args.worker_index,
                num_workers=args.num_workers,
                device_index=args.device_index,
                samples_per_shard=args.samples_per_shard,
                max_samples=args.max_samples,
                validation_fraction=args.validation_fraction,
                min_source_duration_seconds=args.min_source_duration_seconds,
                split_seed=args.split_seed,
                master_port=args.master_port,
                conditioning=args.conditioning,
            )
        elif args.phase == "condition":
            preprocess_condition(
                input_dir=args.input_dir,
                model_path=args.model_path,
                output_dir=args.output_dir,
                worker_index=args.worker_index,
                num_workers=args.num_workers,
                device_index=args.device_index,
                master_port=args.master_port,
                conditioning=args.conditioning,
            )
        elif args.phase == "augment":
            augment_prope_geometry(
                input_dir=args.input_dir,
                manifest_path=args.manifest,
                output_dir=args.output_dir,
                worker_index=args.worker_index,
                num_workers=args.num_workers,
            )
        elif args.phase == "legacy-cache":
            preprocess_legacy_cache(
                metadata_path=args.metadata,
                cache_dir=args.cache_dir,
                vwm_manifest_path=args.vwm_manifest,
                output_dir=args.output_dir,
                worker_index=args.worker_index,
                num_workers=args.num_workers,
                samples_per_shard=args.samples_per_shard,
                max_samples=args.max_samples,
            )
        elif args.phase == "static-pairs":
            build_static_control_pairs(
                input_dir=args.input_dir,
                manifest_path=args.manifest,
                model_path=args.model_path,
                output_dir=args.output_dir,
                sample_ids=args.sample_id,
                device_index=args.device_index,
                master_port=args.master_port,
            )
        else:
            verify_dataset(
                args.input_dir,
                expected_rows=args.expected_rows,
                conditioning=args.conditioning,
            )
    finally:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
