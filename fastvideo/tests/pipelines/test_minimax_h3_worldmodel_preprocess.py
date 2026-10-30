# SPDX-License-Identifier: Apache-2.0
"""CPU tests for resumable MiniMax-H3 world-model preprocessing."""

from io import BytesIO
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
import pytest
import torch

from fastvideo.pipelines.basic.minimax_h3.packing import patchify_video_latents
from fastvideo.pipelines.preprocess.minimax_h3_worldmodel.legacy_cache import (
    extract_legacy_h3_cache_payload,
    recover_legacy_h3_keyframe_latent,
)
from fastvideo.pipelines.preprocess.minimax_h3_worldmodel.preprocess import (
    _encode_png,
    _atomic_write_records,
    _reuse_completed_shard,
    build_parser,
    iter_worker_chunks,
    repair_camera_parquet_shard,
    rebase_prope_vae_record_camera,
    static_control_record,
    verify_dataset,
)
from fastvideo.dataset.dataloader.record_schema import (
    minimax_h3_worldmodel_prope_conditioned_record_creator,
    minimax_h3_worldmodel_prope_vae_record_creator,
)
from fastvideo.dataset.dataloader.schema import pyarrow_schema_minimax_h3_worldmodel_prope


def test_first_frame_png_round_trips_without_legacy_mode_argument():
    frame = torch.from_numpy(np.full((3, 4, 5), 127, dtype=np.uint8))
    encoded = _encode_png(frame)

    with Image.open(BytesIO(encoded)) as image:
        assert image.mode == "RGB"
        assert image.size == (5, 4)


def test_condition_phase_repairs_old_camera_origin_without_changing_relative_path():
    camera = np.broadcast_to(np.eye(4, dtype=np.float32), (2, 4, 4)).copy()
    camera[0, :3, 3] = [2.0, 3.0, 4.0]
    camera[1, :3, 3] = [2.0, 3.0, 5.0]
    original = {
        "id": "old-cache",
        "camera_to_world_bytes": camera.tobytes(),
        "camera_to_world_shape": [2, 4, 4],
        "camera_to_world_dtype": "float32",
    }

    repaired = rebase_prope_vae_record_camera(original)
    corrected = np.frombuffer(repaired["camera_to_world_bytes"], dtype=np.float32).reshape(2, 4, 4)

    assert np.allclose(corrected[0], np.eye(4))
    assert np.allclose(corrected[1, :3, 3], [0.0, 0.0, 1.0])
    assert np.frombuffer(original["camera_to_world_bytes"], dtype=np.float32).reshape(2, 4, 4)[0, 2, 3] == 4.0
    assert rebase_prope_vae_record_camera(repaired) is repaired


def test_existing_camera_parquet_is_repaired_atomically_and_idempotently(tmp_path: Path):
    camera = np.broadcast_to(np.eye(4, dtype=np.float32), (2, 4, 4)).copy()
    camera[:, 0, 3] = [2.0, 3.0]
    schema = pa.schema([
        pa.field("id", pa.string()),
        pa.field("camera_to_world_bytes", pa.binary()),
        pa.field("camera_to_world_shape", pa.list_(pa.int64())),
        pa.field("camera_to_world_dtype", pa.string()),
        pa.field("payload", pa.binary()),
    ])
    path = tmp_path / "old.parquet"
    _atomic_write_records([{
        "id": "old",
        "camera_to_world_bytes": camera.tobytes(),
        "camera_to_world_shape": [2, 4, 4],
        "camera_to_world_dtype": "float32",
        "payload": b"unchanged",
    }], schema, path)

    assert repair_camera_parquet_shard(path) == (1, 1)
    assert repair_camera_parquet_shard(path) == (1, 0)
    record = pq.read_table(path).to_pylist()[0]
    corrected = np.frombuffer(record["camera_to_world_bytes"], dtype=np.float32).reshape(2, 4, 4)
    assert np.allclose(corrected[0], np.eye(4))
    assert np.allclose(corrected[1, 0, 3], 1.0)
    assert record["payload"] == b"unchanged"


def test_worker_chunks_partition_a_global_pilot_without_overlap():
    worker_chunks = [
        list(
            iter_worker_chunks(
                100,
                worker_index=worker,
                num_workers=3,
                samples_per_shard=4,
                max_samples=17,
            )
        )
        for worker in range(3)
    ]
    flattened = [index for chunks in worker_chunks for _, chunk in chunks for index in chunk]

    assert sorted(flattened) == list(range(17))
    assert len(flattened) == len(set(flattened))
    assert all(len(chunk) <= 4 for chunks in worker_chunks for _, chunk in chunks)


def test_vae_parser_accepts_all_records_and_explicit_short_clip_policy():
    args = build_parser().parse_args([
        "vae",
        "--manifest",
        "train.jsonl",
        "--model-path",
        "model",
        "--output-dir",
        "encoded",
        "--split",
        "all",
        "--min-source-duration-seconds",
        "0",
    ])

    assert args.split == "all"
    assert args.min_source_duration_seconds == 0.0


def test_legacy_cache_parser_requires_explicit_sources():
    args = build_parser().parse_args([
        "legacy-cache",
        "--metadata",
        "legacy.jsonl",
        "--cache-dir",
        "legacy-cache",
        "--vwm-manifest",
        "train.jsonl",
        "--output-dir",
        "encoded",
        "--max-samples",
        "2",
    ])

    assert args.phase == "legacy-cache"
    assert args.max_samples == 2
    assert args.samples_per_shard == 16


def test_atomic_shard_can_be_verified_and_reused(tmp_path: Path):
    schema = pa.schema([pa.field("id", pa.string()), pa.field("value", pa.int64())])
    output_path = tmp_path / "worker_000" / "data_000000.parquet"
    _atomic_write_records(
        [{"id": "a", "value": 1}, {"id": "b", "value": 2}],
        schema,
        output_path,
    )

    assert _reuse_completed_shard(output_path, ["a", "b"], schema)
    assert not output_path.with_suffix(".parquet.tmp").exists()


def test_static_control_pair_has_static_pose_action_and_visual_target():
    frames = 2
    vae_record = minimax_h3_worldmodel_prope_vae_record_creator(
        sample_id="source/sample",
        source_name="source",
        caption="camera pans right",
        vae_latent=np.ones((24, frames, 2, 2), dtype=np.float32),
        first_frame_latent=np.ones((24, 1, 2, 2), dtype=np.float32),
        first_frame_png=b"png",
        action_continuous=np.ones((frames, 6), dtype=np.float32),
        action_valid_mask=np.ones(frames, dtype=np.float32),
        source_frame_indices=np.arange(124),
        width=32,
        height=32,
        fps=24.0,
        source_fps=30.0,
        camera_to_world=np.stack((np.eye(4), np.eye(4))).astype(np.float32),
        camera_intrinsics=np.stack((np.eye(3), np.eye(3))).astype(np.float32),
    )
    motion = minimax_h3_worldmodel_prope_conditioned_record_creator(
        vae_record,
        text_embedding=np.zeros((2, 5120), dtype=np.float32),
        text_token_tags=np.array([0, 1], dtype=np.float32),
    )

    static = static_control_record(
        motion,
        static_vae_latent=np.full((24, frames, 2, 2), 3.0, dtype=np.float32),
        static_first_frame_latent=np.full((24, 1, 2, 2), 4.0, dtype=np.float32),
    )

    assert static["id"] == "source/sample#static"
    action = np.frombuffer(static["action_continuous_bytes"], dtype=np.float32)
    latent = np.frombuffer(static["vae_latent_bytes"], dtype=np.float32)
    assert np.all(action == 0.0)
    assert np.all(latent == 3.0)


def test_verify_accepts_normalized_telephoto_intrinsics_but_rejects_pixel_units(tmp_path: Path):
    camera = np.stack((np.eye(4), np.eye(4))).astype(np.float32)
    intrinsics = np.stack((np.eye(3), np.eye(3))).astype(np.float32)
    intrinsics[:, 0, 0] = 6.4
    intrinsics[:, 1, 1] = 11.3
    intrinsics[:, 0, 2] = 0.5
    intrinsics[:, 1, 2] = 0.5
    record = minimax_h3_worldmodel_prope_vae_record_creator(
        sample_id="telephoto",
        source_name="source",
        caption="scene",
        vae_latent=np.ones((24, 2, 2, 2), dtype=np.float32),
        first_frame_latent=np.ones((24, 1, 2, 2), dtype=np.float32),
        first_frame_png=b"png",
        action_continuous=np.zeros((2, 6), dtype=np.float32),
        action_valid_mask=np.ones(2, dtype=np.float32),
        source_frame_indices=np.arange(124),
        width=832,
        height=480,
        fps=24.0,
        source_fps=30.0,
        camera_to_world=camera,
        camera_intrinsics=intrinsics,
    )
    conditioned = minimax_h3_worldmodel_prope_conditioned_record_creator(
        record,
        text_embedding=np.zeros((2, 5120), dtype=np.float32),
        text_token_tags=np.array([0, 1], dtype=np.float32),
    )
    path = tmp_path / "valid" / "data_000000.parquet"
    _atomic_write_records([conditioned], pyarrow_schema_minimax_h3_worldmodel_prope, path)
    verify_dataset(path.parent, expected_rows=1, conditioning="prope")

    intrinsics[:, 1, 1] = 400.0
    record["camera_intrinsics_bytes"] = intrinsics.tobytes()
    conditioned = minimax_h3_worldmodel_prope_conditioned_record_creator(
        record,
        text_embedding=np.zeros((2, 5120), dtype=np.float32),
        text_token_tags=np.array([0, 1], dtype=np.float32),
    )
    bad_path = tmp_path / "pixel" / "data_000000.parquet"
    _atomic_write_records([conditioned], pyarrow_schema_minimax_h3_worldmodel_prope, bad_path)
    with pytest.raises(ValueError, match="not normalized"):
        verify_dataset(bad_path.parent, expected_rows=1, conditioning="prope")


def _legacy_keyframe_anchor(clean: torch.Tensor) -> torch.Tensor:
    generator = torch.Generator("cpu").manual_seed(42)
    noise = torch.randn((1, 24, 38, clean.shape[-2], clean.shape[-1]),
                        generator=generator, dtype=torch.bfloat16)[:, :, :1]
    clean_rows = patchify_video_latents(clean, (1, 2, 2))
    noise_rows = patchify_video_latents(noise, (1, 2, 2))
    strength = torch.tensor(0.999, dtype=torch.bfloat16)
    return strength * clean_rows + (1.0 - strength) * noise_rows


def test_legacy_keyframe_recovery_undoes_noise_anchor():
    generator = torch.Generator("cpu").manual_seed(7)
    clean = torch.randn((1, 24, 1, 4, 6), generator=generator, dtype=torch.bfloat16)
    anchor = _legacy_keyframe_anchor(clean)

    recovered = recover_legacy_h3_keyframe_latent(
        anchor,
        (24, 37, 4, 6),
        noise_strength=0.999,
    )

    assert recovered.shape == (24, 1, 4, 6)
    assert torch.allclose(recovered, clean[0].float(), atol=2.0e-2, rtol=2.0e-2)


def test_legacy_payload_drops_audio_and_action_text(tmp_path: Path):
    clean = torch.zeros((1, 24, 1, 30, 52), dtype=torch.bfloat16)
    prompt = torch.arange(39 * 5120, dtype=torch.float32).reshape(39, 5120).to(torch.bfloat16)
    spans = [(2 + index, 3 + index) for index in range(37)]
    token_tags = torch.ones(39, dtype=torch.long)
    token_tags[0] = 0
    cache_path = tmp_path / "0.pth"
    torch.save(
        (
            {
                "input_latents": torch.ones((1, 24, 37, 30, 52), dtype=torch.bfloat16),
                "audio_input_latents": torch.full((2, 32, 207), 123.0, dtype=torch.bfloat16),
                "keyframe_cond_anchor": _legacy_keyframe_anchor(clean),
                "imgvid_cond_noise_aug": 0.999,
            },
            {
                "prompt_embeds": prompt,
                "packed": {
                    "action_text_spans_local": spans,
                    "text_pos": torch.arange(39),
                    "token_tags": token_tags,
                },
            },
            {},
        ),
        cache_path,
    )

    video, first_frame, text, tags = extract_legacy_h3_cache_payload(cache_path)

    assert video.shape == (24, 37, 30, 52)
    assert first_frame.shape == (24, 1, 30, 52)
    assert text.shape == (2, 5120)
    assert torch.equal(text, prompt[:2].float())
    assert tags.tolist() == [0, 1]
    assert all(tensor.dtype in (torch.float32, torch.int64)
               for tensor in (video, first_frame, text, tags))
