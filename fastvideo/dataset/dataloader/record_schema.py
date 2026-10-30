from typing import Any

import numpy as np

from fastvideo.pipelines.pipeline_batch_info import PreprocessBatch


def _float32_tensor_fields(name: str, value: np.ndarray) -> dict[str, Any]:
    """Serialize one tensor for the current float32-only parquet collator."""
    array = np.ascontiguousarray(value, dtype=np.float32)
    return {
        f"{name}_bytes": array.tobytes(),
        f"{name}_shape": list(array.shape),
        f"{name}_dtype": "float32",
    }


def _minimax_h3_worldmodel_base_record(
    *,
    sample_id: str,
    source_name: str,
    caption: str,
    vae_latent: np.ndarray,
    first_frame_latent: np.ndarray,
    action_continuous: np.ndarray,
    action_valid_mask: np.ndarray,
    source_frame_indices: np.ndarray,
    width: int,
    height: int,
    fps: float,
    source_fps: float,
) -> dict[str, Any]:
    """Validate and serialize fields shared by both preprocessing phases."""
    vae_latent = np.asarray(vae_latent)
    first_frame_latent = np.asarray(first_frame_latent)
    action_continuous = np.asarray(action_continuous)
    action_valid_mask = np.asarray(action_valid_mask)
    source_frame_indices = np.asarray(source_frame_indices, dtype=np.int64)

    if vae_latent.ndim != 4 or vae_latent.shape[0] != 24:
        raise ValueError(f"vae_latent must have shape [24,T,H,W], got {vae_latent.shape}")
    if first_frame_latent.shape != (24, 1, vae_latent.shape[2], vae_latent.shape[3]):
        raise ValueError(
            "first_frame_latent must have shape [24,1,H,W] matching vae_latent, "
            f"got {first_frame_latent.shape}"
        )
    expected_action_shape = (vae_latent.shape[1], 6)
    if action_continuous.shape != expected_action_shape:
        raise ValueError(
            f"action_continuous must have shape {expected_action_shape}, got {action_continuous.shape}"
        )
    if action_valid_mask.shape != (vae_latent.shape[1],):
        raise ValueError("action_valid_mask must have one entry per video latent frame")
    if source_frame_indices.ndim != 1 or source_frame_indices.size <= 0:
        raise ValueError("source_frame_indices must be a non-empty vector")
    if int(width) <= 0 or int(height) <= 0 or float(fps) <= 0.0 or float(source_fps) <= 0.0:
        raise ValueError("width, height, fps, and source_fps must be positive")
    if not str(sample_id) or not str(source_name) or not str(caption).strip():
        raise ValueError("sample_id, source_name, and caption must be non-empty")

    record: dict[str, Any] = {
        "id": str(sample_id),
        "source_name": str(source_name),
        "caption": str(caption),
        "media_type": "video_worldmodel",
        "width": int(width),
        "height": int(height),
        "num_frames": int(source_frame_indices.size),
        "duration_sec": float(source_frame_indices.size) / float(fps),
        "fps": float(fps),
        "source_fps": float(source_fps),
        "source_frame_indices": source_frame_indices.tolist(),
    }
    for name, value in {
        "vae_latent": vae_latent,
        "first_frame_latent": first_frame_latent,
        "action_continuous": action_continuous,
        "action_valid_mask": action_valid_mask,
    }.items():
        record.update(_float32_tensor_fields(name, value))
    return record


def minimax_h3_worldmodel_vae_record_creator(
    *,
    first_frame_png: bytes,
    **kwargs: Any,
) -> dict[str, Any]:
    """Create one resumable H3 VAE/action intermediate record."""
    if not isinstance(first_frame_png, bytes) or not first_frame_png:
        raise ValueError("first_frame_png must contain one lossless RGB image")
    record = _minimax_h3_worldmodel_base_record(**kwargs)
    record["first_frame_png"] = first_frame_png
    return record


def _minimax_h3_worldmodel_pose_fields(
    camera_to_world: np.ndarray,
    camera_intrinsics: np.ndarray,
    *,
    num_latent_frames: int,
) -> dict[str, Any]:
    """Validate and serialize latent-aligned calibrated camera geometry."""
    camera_to_world = np.asarray(camera_to_world, dtype=np.float32)
    camera_intrinsics = np.asarray(camera_intrinsics, dtype=np.float32)
    if camera_to_world.shape != (num_latent_frames, 4, 4):
        raise ValueError(
            f"camera_to_world must have shape [{num_latent_frames},4,4], got {camera_to_world.shape}"
        )
    if camera_intrinsics.shape != (num_latent_frames, 3, 3):
        raise ValueError(
            f"camera_intrinsics must have shape [{num_latent_frames},3,3], got {camera_intrinsics.shape}"
        )
    if not np.isfinite(camera_to_world).all() or not np.isfinite(camera_intrinsics).all():
        raise ValueError("camera geometry must contain only finite values")
    expected_bottom_row = np.broadcast_to(
        np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        camera_to_world[:, 3, :].shape,
    )
    if not np.allclose(camera_to_world[:, 3, :], expected_bottom_row, atol=1.0e-4):
        raise ValueError("camera_to_world must contain homogeneous SE(3) matrices")
    if np.any(camera_intrinsics[:, 0, 0] <= 0.0) or np.any(camera_intrinsics[:, 1, 1] <= 0.0):
        raise ValueError("camera intrinsics must have positive focal lengths")
    fields: dict[str, Any] = {}
    fields.update(_float32_tensor_fields("camera_to_world", camera_to_world))
    fields.update(_float32_tensor_fields("camera_intrinsics", camera_intrinsics))
    return fields


def minimax_h3_worldmodel_prope_vae_record_creator(
    *,
    first_frame_png: bytes,
    camera_to_world: np.ndarray,
    camera_intrinsics: np.ndarray,
    **kwargs: Any,
) -> dict[str, Any]:
    """Create one resumable H3 VAE record with PRoPE camera geometry."""
    record = minimax_h3_worldmodel_vae_record_creator(first_frame_png=first_frame_png, **kwargs)
    record.update(
        _minimax_h3_worldmodel_pose_fields(
            camera_to_world,
            camera_intrinsics,
            num_latent_frames=int(np.asarray(kwargs["vae_latent"]).shape[1]),
        )
    )
    return record


def _minimax_h3_worldmodel_condition_fields(
    text_embedding: np.ndarray,
    text_token_tags: np.ndarray,
) -> dict[str, Any]:
    """Validate and serialize Qwen3-VL conditioning fields."""
    text_embedding = np.asarray(text_embedding)
    text_token_tags = np.asarray(text_token_tags)
    if text_embedding.ndim != 2 or text_embedding.shape[1] != 5120:
        raise ValueError(f"text_embedding must have shape [L,5120], got {text_embedding.shape}")
    if text_token_tags.shape != (text_embedding.shape[0],):
        raise ValueError("text_token_tags must have one entry per text embedding row")
    if not np.isin(text_token_tags, (0, 1)).all():
        raise ValueError("text_token_tags may contain only H3 video=0 and text=1 tags")
    fields: dict[str, Any] = {}
    for name, value in {
        "text_embedding": text_embedding,
        "text_token_tags": text_token_tags,
    }.items():
        fields.update(_float32_tensor_fields(name, value))
    return fields


def minimax_h3_worldmodel_conditioned_record_creator(
    vae_record: dict[str, Any],
    *,
    text_embedding: np.ndarray,
    text_token_tags: np.ndarray,
) -> dict[str, Any]:
    """Complete one staged VAE record without copying its large tensors."""
    from fastvideo.dataset.dataloader.schema import (
        pyarrow_schema_minimax_h3_worldmodel,
        pyarrow_schema_minimax_h3_worldmodel_vae,
    )

    if set(vae_record) != set(pyarrow_schema_minimax_h3_worldmodel_vae.names):
        raise ValueError("vae_record does not match the H3 world-model VAE schema")
    record = {
        name: vae_record[name]
        for name in pyarrow_schema_minimax_h3_worldmodel.names
        if name in vae_record
    }
    record.update(_minimax_h3_worldmodel_condition_fields(text_embedding, text_token_tags))
    return record


def minimax_h3_worldmodel_prope_conditioned_record_creator(
    vae_record: dict[str, Any],
    *,
    text_embedding: np.ndarray,
    text_token_tags: np.ndarray,
) -> dict[str, Any]:
    """Complete one staged PRoPE VAE record with native H3 conditioning."""
    from fastvideo.dataset.dataloader.schema import (
        pyarrow_schema_minimax_h3_worldmodel_prope,
        pyarrow_schema_minimax_h3_worldmodel_prope_vae,
    )

    if set(vae_record) != set(pyarrow_schema_minimax_h3_worldmodel_prope_vae.names):
        raise ValueError("vae_record does not match the H3 PRoPE world-model VAE schema")
    record = {
        name: vae_record[name]
        for name in pyarrow_schema_minimax_h3_worldmodel_prope.names
        if name in vae_record
    }
    record.update(_minimax_h3_worldmodel_condition_fields(text_embedding, text_token_tags))
    return record


def minimax_h3_worldmodel_prope_record_from_additive(
    additive_record: dict[str, Any],
    *,
    camera_to_world: np.ndarray,
    camera_intrinsics: np.ndarray,
) -> dict[str, Any]:
    """Add calibrated geometry without recomputing identical H3 encodings."""
    from fastvideo.dataset.dataloader.schema import (
        pyarrow_schema_minimax_h3_worldmodel,
        pyarrow_schema_minimax_h3_worldmodel_prope,
    )

    if set(additive_record) != set(pyarrow_schema_minimax_h3_worldmodel.names):
        raise ValueError("additive_record does not match the H3 world-model schema")
    num_latent_frames = int(additive_record["vae_latent_shape"][1])
    record = dict(additive_record)
    record.update(
        _minimax_h3_worldmodel_pose_fields(
            camera_to_world,
            camera_intrinsics,
            num_latent_frames=num_latent_frames,
        )
    )
    if set(record) != set(pyarrow_schema_minimax_h3_worldmodel_prope.names):
        raise RuntimeError("augmented record does not match the H3 PRoPE schema")
    return record


def minimax_h3_worldmodel_record_creator(
    *,
    text_embedding: np.ndarray,
    text_token_tags: np.ndarray,
    **kwargs: Any,
) -> dict[str, Any]:
    """Create one audio-free MiniMax-H3 Stage-0 world-model record."""
    record = _minimax_h3_worldmodel_base_record(**kwargs)
    record.update(_minimax_h3_worldmodel_condition_fields(text_embedding, text_token_tags))
    return record


def basic_t2v_record_creator(batch: PreprocessBatch) -> list[dict[str, Any]]:
    """Create a record for the Parquet dataset from PreprocessBatch."""
    # For batch processing, we need to handle the case where some fields might be single values
    # or lists depending on the batch size

    assert isinstance(batch.prompt, list)
    assert isinstance(batch.width, list)
    assert isinstance(batch.height, list)
    assert isinstance(batch.fps, list)
    assert isinstance(batch.num_frames, list)

    records = []
    for idx, video_name in enumerate(batch.video_file_name):
        width = batch.width[idx] if batch.width is not None else 0
        height = batch.height[idx] if batch.height is not None else 0

        # Get FPS - single value in PreprocessBatch
        fps_val = float(batch.fps[idx]) if batch.fps is not None else 0.0

        # For duration, we need to calculate it or use a default since it's not in PreprocessBatch
        # duration = num_frames / fps if available
        duration_val = 0.0
        if batch.num_frames[idx] and batch.fps[idx] and batch.fps[idx] > 0:
            duration_val = float(batch.num_frames[idx]) / float(batch.fps[idx])

        record = {
            "id": video_name,
            "vae_latent_bytes": batch.latents[idx].tobytes(),
            "vae_latent_shape": list(batch.latents[idx].shape),
            "vae_latent_dtype": str(batch.latents[idx].dtype),
            "text_embedding_bytes": batch.prompt_embeds[idx].tobytes(),
            "text_embedding_shape": list(batch.prompt_embeds[idx].shape),
            "text_embedding_dtype": str(batch.prompt_embeds[idx].dtype),
            "file_name": video_name,
            "caption": batch.prompt[idx],
            "media_type": "video",
            "width": int(width),
            "height": int(height),
            "num_frames": batch.latents[idx].shape[1] if len(batch.latents[idx].shape) > 1 else 0,
            "duration_sec": duration_val,
            "fps": fps_val,
        }
        records.append(record)

    return records


def i2v_record_creator(batch: PreprocessBatch) -> list[dict[str, Any]]:
    """Create a record for the Parquet dataset with CLIP features."""
    records = basic_t2v_record_creator(batch)

    if len(batch.image_embeds) > 0:
        assert len(batch.image_embeds) == 1, "image embedding should be a single tensor"
        image_embeds = batch.image_embeds[0]
    else:
        image_embeds = None
    image_latent = batch.image_latent
    pil_image = batch.pil_image

    for idx, record in enumerate(records):
        if image_embeds is not None:
            record.update({
                "clip_feature_bytes": image_embeds[idx].tobytes(),
                "clip_feature_shape": list(image_embeds[idx].shape),
                "clip_feature_dtype": str(image_embeds[idx].dtype),
            })
        else:
            record.update({
                "clip_feature_bytes": b"",
                "clip_feature_shape": [],
                "clip_feature_dtype": "",
            })

        if image_latent is not None:
            record.update({
                "first_frame_latent_bytes": image_latent[idx].tobytes(),
                "first_frame_latent_shape": list(image_latent[idx].shape),
                "first_frame_latent_dtype": str(image_latent[idx].dtype),
            })
        else:
            record.update({
                "first_frame_latent_bytes": b"",
                "first_frame_latent_shape": [],
                "first_frame_latent_dtype": "",
            })

        if pil_image is not None:
            record.update({
                "pil_image_bytes": pil_image[idx].tobytes(),
                "pil_image_shape": list(pil_image[idx].shape),
                "pil_image_dtype": str(pil_image[idx].dtype),
            })
        else:
            record.update({
                "pil_image_bytes": b"",
                "pil_image_shape": [],
                "pil_image_dtype": "",
            })

    return records


def ode_text_only_record_creator(video_name: str, text_embedding: np.ndarray, caption: str,
                                 trajectory_latents: np.ndarray, trajectory_timesteps: np.ndarray) -> dict[str, Any]:
    """Create a text-only ODE trajectory record matching pyarrow_schema_ode_trajectory_text_only.

    Args:
        video_name: Base name/id for the sample (without extension).
        text_embedding: Text encoder output array [SeqLen, Dim].
        caption: Original text prompt.
        trajectory_latents: Collected trajectory latents array.
        trajectory_timesteps: Collected timesteps array.

    Returns:
        dict suitable for records_to_table(…, pyarrow_schema_ode_trajectory_text_only)
    """
    assert trajectory_latents is not None, "trajectory_latents is required"
    assert trajectory_timesteps is not None, "trajectory_timesteps is required"

    record = {
        "id": f"text_{video_name}",
        "text_embedding_bytes": text_embedding.tobytes(),
        "text_embedding_shape": list(text_embedding.shape),
        "text_embedding_dtype": str(text_embedding.dtype),
        "file_name": video_name,
        "caption": caption,
        "media_type": "text",
    }

    record.update({
        "trajectory_latents_bytes": trajectory_latents.tobytes(),
        "trajectory_latents_shape": list(trajectory_latents.shape),
        "trajectory_latents_dtype": str(trajectory_latents.dtype),
    })

    record.update({
        "trajectory_timesteps_bytes": trajectory_timesteps.tobytes(),
        "trajectory_timesteps_shape": list(trajectory_timesteps.shape),
        "trajectory_timesteps_dtype": str(trajectory_timesteps.dtype),
    })

    return record


def text_only_record_creator(text_name: str, text_embedding: np.ndarray, caption: str) -> dict[str, Any]:
    """Create a text-only record matching pyarrow_schema_text_only.

    Args:
        text_name: Base id/name for the text sample.
        text_embedding: Text encoder output array [SeqLen, Dim].
        caption: Original text prompt.

    Returns:
        dict suitable for records_to_table(…, pyarrow_schema_text_only)
    """
    record = {
        "id": f"text_{text_name}",
        "text_embedding_bytes": text_embedding.tobytes(),
        "text_embedding_shape": list(text_embedding.shape),
        "text_embedding_dtype": str(text_embedding.dtype),
        "caption": caption,
    }
    return record


def matrixgame2_ode_record_creator(video_name: str,
                                   clip_feature: np.ndarray,
                                   first_frame_latent: np.ndarray,
                                   trajectory_latents: np.ndarray,
                                   trajectory_timesteps: np.ndarray,
                                   pil_image: np.ndarray | None = None,
                                   keyboard_cond: np.ndarray | None = None,
                                   mouse_cond: np.ndarray | None = None,
                                   caption: str = "") -> dict[str, Any]:
    """Create a ODE trajectory record matching pyarrow_schema_matrixgame2_ode_trajectory.

    Args:
        video_name: Base name/id for the sample.
        clip_feature: CLIP image features.
        first_frame_latent: First frame VAE latent.
        trajectory_latents: Collected trajectory latents array.
        trajectory_timesteps: Collected timesteps array.
        pil_image: Optional PIL image array for validation.
        keyboard_cond: Optional keyboard action array.
        mouse_cond: Optional mouse action array.
        caption: Optional caption.

    Returns:
        dict suitable for records_to_table(…, pyarrow_schema_matrixgame2_ode_trajectory)
    """
    assert trajectory_latents is not None, "trajectory_latents is required"
    assert trajectory_timesteps is not None, "trajectory_timesteps is required"
    assert clip_feature is not None, "clip_feature is required"
    assert first_frame_latent is not None, "first_frame_latent is required"

    record = {
        "id": video_name,
        "file_name": video_name,
        "caption": caption,
        "media_type": "video",
    }

    # I2V features
    record.update({
        "clip_feature_bytes": clip_feature.tobytes(),
        "clip_feature_shape": list(clip_feature.shape),
        "clip_feature_dtype": str(clip_feature.dtype),
    })

    record.update({
        "first_frame_latent_bytes": first_frame_latent.tobytes(),
        "first_frame_latent_shape": list(first_frame_latent.shape),
        "first_frame_latent_dtype": str(first_frame_latent.dtype),
    })

    # Optional PIL Image
    if pil_image is not None:
        record.update({
            "pil_image_bytes": pil_image.tobytes(),
            "pil_image_shape": list(pil_image.shape),
            "pil_image_dtype": str(pil_image.dtype),
        })
    else:
        record.update({
            "pil_image_bytes": b"",
            "pil_image_shape": [],
            "pil_image_dtype": "",
        })

    # Actions
    if keyboard_cond is not None:
        record.update({
            "keyboard_cond_bytes": keyboard_cond.tobytes(),
            "keyboard_cond_shape": list(keyboard_cond.shape),
            "keyboard_cond_dtype": str(keyboard_cond.dtype),
        })
    else:
        record.update({
            "keyboard_cond_bytes": b"",
            "keyboard_cond_shape": [],
            "keyboard_cond_dtype": "",
        })

    if mouse_cond is not None:
        record.update({
            "mouse_cond_bytes": mouse_cond.tobytes(),
            "mouse_cond_shape": list(mouse_cond.shape),
            "mouse_cond_dtype": str(mouse_cond.dtype),
        })
    else:
        record.update({
            "mouse_cond_bytes": b"",
            "mouse_cond_shape": [],
            "mouse_cond_dtype": "",
        })

    record.update({
        "trajectory_latents_bytes": trajectory_latents.tobytes(),
        "trajectory_latents_shape": list(trajectory_latents.shape),
        "trajectory_latents_dtype": str(trajectory_latents.dtype),
    })

    record.update({
        "trajectory_timesteps_bytes": trajectory_timesteps.tobytes(),
        "trajectory_timesteps_shape": list(trajectory_timesteps.shape),
        "trajectory_timesteps_dtype": str(trajectory_timesteps.dtype),
    })

    return record
