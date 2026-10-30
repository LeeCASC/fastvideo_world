# SPDX-License-Identifier: Apache-2.0
# schema.py
"""
Unified data schema and format for saving and loading image/video data after
preprocessing.

It uses apache arrow in-memory format that can be consumed by modern data
frameworks that can handle parquet or lance file.
"""

import pyarrow as pa

pyarrow_schema_i2v = pa.schema([
    pa.field("id", pa.string()),
    # --- Image/Video VAE latents ---
    # Tensors are stored as raw bytes with shape and dtype info for loading
    pa.field("vae_latent_bytes", pa.binary()),
    # e.g., [C, T, H, W] or [C, H, W]
    pa.field("vae_latent_shape", pa.list_(pa.int64())),
    # e.g., 'float32'
    pa.field("vae_latent_dtype", pa.string()),
    # --- Text encoder output tensor ---
    # Tensors are stored as raw bytes with shape and dtype info for loading
    pa.field("text_embedding_bytes", pa.binary()),
    # e.g., [SeqLen, Dim]
    pa.field("text_embedding_shape", pa.list_(pa.int64())),
    # e.g., 'bfloat16' or 'float32'
    pa.field("text_embedding_dtype", pa.string()),
    #I2V
    pa.field("clip_feature_bytes", pa.binary()),
    pa.field("clip_feature_shape", pa.list_(pa.int64())),
    pa.field("clip_feature_dtype", pa.string()),
    pa.field("first_frame_latent_bytes", pa.binary()),
    pa.field("first_frame_latent_shape", pa.list_(pa.int64())),
    pa.field("first_frame_latent_dtype", pa.string()),
    # I2V Validation
    pa.field("pil_image_bytes", pa.binary()),
    pa.field("pil_image_shape", pa.list_(pa.int64())),
    pa.field("pil_image_dtype", pa.string()),
    # --- Metadata ---
    pa.field("file_name", pa.string()),
    pa.field("caption", pa.string()),
    pa.field("media_type", pa.string()),  # 'image' or 'video'
    pa.field("width", pa.int64()),
    pa.field("height", pa.int64()),
    # -- Video-specific (can be null/default for images) ---
    # Number of frames processed (e.g., 1 for image, N for video)
    pa.field("num_frames", pa.int64()),
    pa.field("duration_sec", pa.float64()),
    pa.field("fps", pa.float64()),
])

pyarrow_schema_t2v = pa.schema([
    pa.field("id", pa.string()),
    # --- Image/Video VAE latents ---
    # Tensors are stored as raw bytes with shape and dtype info for loading
    pa.field("vae_latent_bytes", pa.binary()),
    # e.g., [C, T, H, W] or [C, H, W]
    pa.field("vae_latent_shape", pa.list_(pa.int64())),
    # e.g., 'float32'
    pa.field("vae_latent_dtype", pa.string()),
    # --- Text encoder output tensor ---
    # Tensors are stored as raw bytes with shape and dtype info for loading
    pa.field("text_embedding_bytes", pa.binary()),
    # e.g., [SeqLen, Dim]
    pa.field("text_embedding_shape", pa.list_(pa.int64())),
    # e.g., 'bfloat16' or 'float32'
    pa.field("text_embedding_dtype", pa.string()),
    # --- Metadata ---
    pa.field("file_name", pa.string()),
    pa.field("caption", pa.string()),
    pa.field("media_type", pa.string()),  # 'image' or 'video'
    pa.field("width", pa.int64()),
    pa.field("height", pa.int64()),
    # -- Video-specific (can be null/default for images) ---
    # Number of frames processed (e.g., 1 for image, N for video)
    pa.field("num_frames", pa.int64()),
    pa.field("duration_sec", pa.float64()),
    pa.field("fps", pa.float64()),
])

# One text-to-video-and-audio (T2VA) row owns synchronized video, audio, and text tensors so
# collate_rows_from_parquet_schema cannot pair targets from different samples.
# Each tensor uses the bytes/shape/dtype triplet that the collator discovers.
pyarrow_schema_t2va = pa.schema([
    pa.field("id", pa.string()),
    # --- Video VAE latent tensor ---
    pa.field("vae_latent_bytes", pa.binary()),
    pa.field("vae_latent_shape", pa.list_(pa.int64())),
    pa.field("vae_latent_dtype", pa.string()),
    # --- Stereo audio VAE latent tensor ---
    pa.field("audio_latent_bytes", pa.binary()),
    pa.field("audio_latent_shape", pa.list_(pa.int64())),
    pa.field("audio_latent_dtype", pa.string()),
    # --- Text encoder output tensor ---
    pa.field("text_embedding_bytes", pa.binary()),
    pa.field("text_embedding_shape", pa.list_(pa.int64())),
    pa.field("text_embedding_dtype", pa.string()),
    # --- Paired audio-video metadata ---
    pa.field("file_name", pa.string()),
    pa.field("caption", pa.string()),
    pa.field("media_type", pa.string()),
    pa.field("width", pa.int64()),
    pa.field("height", pa.int64()),
    pa.field("num_frames", pa.int64()),
    pa.field("duration_sec", pa.float64()),
    pa.field("fps", pa.float64()),
    pa.field("audio_sample_rate", pa.int64()),
])

# MiniMax-H3 Stage-0 world-model rows are deliberately video-only. The first
# frame is represented both in Qwen3-VL conditioning (embedding + token tags)
# and as one native H3 VAE keyframe latent. Actions align with H3 video latent
# frames rather than raw pixels.
pyarrow_schema_minimax_h3_worldmodel = pa.schema([
    pa.field("id", pa.string()),
    pa.field("vae_latent_bytes", pa.binary()),
    pa.field("vae_latent_shape", pa.list_(pa.int64())),
    pa.field("vae_latent_dtype", pa.string()),
    pa.field("first_frame_latent_bytes", pa.binary()),
    pa.field("first_frame_latent_shape", pa.list_(pa.int64())),
    pa.field("first_frame_latent_dtype", pa.string()),
    pa.field("text_embedding_bytes", pa.binary()),
    pa.field("text_embedding_shape", pa.list_(pa.int64())),
    pa.field("text_embedding_dtype", pa.string()),
    pa.field("text_token_tags_bytes", pa.binary()),
    pa.field("text_token_tags_shape", pa.list_(pa.int64())),
    pa.field("text_token_tags_dtype", pa.string()),
    pa.field("action_continuous_bytes", pa.binary()),
    pa.field("action_continuous_shape", pa.list_(pa.int64())),
    pa.field("action_continuous_dtype", pa.string()),
    pa.field("action_valid_mask_bytes", pa.binary()),
    pa.field("action_valid_mask_shape", pa.list_(pa.int64())),
    pa.field("action_valid_mask_dtype", pa.string()),
    pa.field("source_name", pa.string()),
    pa.field("caption", pa.string()),
    pa.field("media_type", pa.string()),
    pa.field("width", pa.int64()),
    pa.field("height", pa.int64()),
    pa.field("num_frames", pa.int64()),
    pa.field("duration_sec", pa.float64()),
    pa.field("fps", pa.float64()),
    pa.field("source_fps", pa.float64()),
    pa.field("source_frame_indices", pa.list_(pa.int64())),
])

# PRoPE conditioning consumes calibrated camera geometry rather than the
# lossy 6-DoF delta representation. Keep this contract separate so existing
# additive-action parquet remains readable and reproducible.
pyarrow_schema_minimax_h3_worldmodel_prope = pa.schema([
    *pyarrow_schema_minimax_h3_worldmodel,
    pa.field("camera_to_world_bytes", pa.binary()),
    pa.field("camera_to_world_shape", pa.list_(pa.int64())),
    pa.field("camera_to_world_dtype", pa.string()),
    pa.field("camera_intrinsics_bytes", pa.binary()),
    pa.field("camera_intrinsics_shape", pa.list_(pa.int64())),
    pa.field("camera_intrinsics_dtype", pa.string()),
])

# An audited same-first-frame partner supplies a second *real* video target.
# The partner's camera and intrinsics are stored alongside its latent so the
# paired objective never treats a counterfactual pose as a positive label.
pyarrow_schema_minimax_h3_worldmodel_prope_pair = pa.schema([
    *pyarrow_schema_minimax_h3_worldmodel_prope,
    pa.field("pair_id", pa.string()),
    pa.field("pair_direction", pa.string()),
    pa.field("pair_vae_latent_bytes", pa.binary()),
    pa.field("pair_vae_latent_shape", pa.list_(pa.int64())),
    pa.field("pair_vae_latent_dtype", pa.string()),
    pa.field("pair_camera_to_world_bytes", pa.binary()),
    pa.field("pair_camera_to_world_shape", pa.list_(pa.int64())),
    pa.field("pair_camera_to_world_dtype", pa.string()),
    pa.field("pair_camera_intrinsics_bytes", pa.binary()),
    pa.field("pair_camera_intrinsics_shape", pa.list_(pa.int64())),
    pa.field("pair_camera_intrinsics_dtype", pa.string()),
    pa.field("pair_action_valid_mask_bytes", pa.binary()),
    pa.field("pair_action_valid_mask_shape", pa.list_(pa.int64())),
    pa.field("pair_action_valid_mask_dtype", pa.string()),
])

# The first preprocessing phase stores everything that does not require the
# Qwen3-VL conditioner, plus a lossless first-frame image for the second phase.
# Keeping this as a strict schema makes staged/resumable preprocessing as
# verifiable as the final training dataset.
pyarrow_schema_minimax_h3_worldmodel_vae = pa.schema([
    field
    for field in pyarrow_schema_minimax_h3_worldmodel
    if not field.name.startswith(("text_embedding_", "text_token_tags_"))
] + [pa.field("first_frame_png", pa.binary())])

pyarrow_schema_minimax_h3_worldmodel_prope_vae = pa.schema([
    field
    for field in pyarrow_schema_minimax_h3_worldmodel_prope
    if not field.name.startswith(("text_embedding_", "text_token_tags_"))
] + [pa.field("first_frame_png", pa.binary())])

pyarrow_schema_ode_trajectory_text_only = pa.schema([
    pa.field("id", pa.string()),
    # --- Text encoder output tensor ---
    # Tensors are stored as raw bytes with shape and dtype info for loading
    pa.field("text_embedding_bytes", pa.binary()),
    # e.g., [SeqLen, Dim]
    pa.field("text_embedding_shape", pa.list_(pa.int64())),
    # e.g., 'bfloat16' or 'float32'
    pa.field("text_embedding_dtype", pa.string()),
    # --- ODE Trajectory ---
    pa.field("trajectory_latents_bytes", pa.binary()),
    pa.field("trajectory_latents_shape", pa.list_(pa.int64())),
    pa.field("trajectory_latents_dtype", pa.string()),
    pa.field("trajectory_timesteps_bytes", pa.binary()),
    pa.field("trajectory_timesteps_shape", pa.list_(pa.int64())),
    pa.field("trajectory_timesteps_dtype", pa.string()),
    # --- Metadata ---
    pa.field("file_name", pa.string()),
    pa.field("caption", pa.string()),
    pa.field("media_type", pa.string()),  # Always 'text' for text-only
])

pyarrow_schema_text_only = pa.schema([
    pa.field("id", pa.string()),
    # --- Text encoder output tensor ---
    # Tensors are stored as raw bytes with shape and dtype info for loading
    pa.field("text_embedding_bytes", pa.binary()),
    # e.g., [SeqLen, Dim]
    pa.field("text_embedding_shape", pa.list_(pa.int64())),
    # e.g., 'bfloat16' or 'float32'
    pa.field("text_embedding_dtype", pa.string()),
    # --- Metadata ---
    pa.field("caption", pa.string()),
])

pyarrow_schema_matrixgame2 = pa.schema([
    pa.field("id", pa.string()),
    # --- Image/Video VAE latents ---
    # Tensors are stored as raw bytes with shape and dtype info for loading
    pa.field("vae_latent_bytes", pa.binary()),
    # e.g., [C, T, H, W] or [C, H, W]
    pa.field("vae_latent_shape", pa.list_(pa.int64())),
    # e.g., 'float32'
    pa.field("vae_latent_dtype", pa.string()),
    #I2V
    pa.field("clip_feature_bytes", pa.binary()),
    pa.field("clip_feature_shape", pa.list_(pa.int64())),
    pa.field("clip_feature_dtype", pa.string()),
    pa.field("first_frame_latent_bytes", pa.binary()),
    pa.field("first_frame_latent_shape", pa.list_(pa.int64())),
    pa.field("first_frame_latent_dtype", pa.string()),
    # --- Action ---
    pa.field("mouse_cond_bytes", pa.binary()),
    pa.field("mouse_cond_shape", pa.list_(pa.int64())),  # [T, 2]
    pa.field("mouse_cond_dtype", pa.string()),
    pa.field("keyboard_cond_bytes", pa.binary()),
    pa.field("keyboard_cond_shape", pa.list_(pa.int64())),  # [T, 4]
    pa.field("keyboard_cond_dtype", pa.string()),
    # I2V Validation
    pa.field("pil_image_bytes", pa.binary()),
    pa.field("pil_image_shape", pa.list_(pa.int64())),
    pa.field("pil_image_dtype", pa.string()),
    # --- Metadata ---
    pa.field("file_name", pa.string()),
    pa.field("caption", pa.string()),
    pa.field("media_type", pa.string()),  # 'image' or 'video'
    pa.field("width", pa.int64()),
    pa.field("height", pa.int64()),
    # -- Video-specific (can be null/default for images) ---
    # Number of frames processed (e.g., 1 for image, N for video)
    pa.field("num_frames", pa.int64()),
    pa.field("duration_sec", pa.float64()),
    pa.field("fps", pa.float64()),
])

pyarrow_schema_matrixgame2_ode_trajectory = pa.schema([
    pa.field("id", pa.string()),
    # --- Action ---
    pa.field("mouse_cond_bytes", pa.binary()),
    pa.field("mouse_cond_shape", pa.list_(pa.int64())),  # [T, 2]
    pa.field("mouse_cond_dtype", pa.string()),
    pa.field("keyboard_cond_bytes", pa.binary()),
    pa.field("keyboard_cond_shape", pa.list_(pa.int64())),  # [T, 4]
    pa.field("keyboard_cond_dtype", pa.string()),
    #I2V
    pa.field("clip_feature_bytes", pa.binary()),
    pa.field("clip_feature_shape", pa.list_(pa.int64())),
    pa.field("clip_feature_dtype", pa.string()),
    pa.field("first_frame_latent_bytes", pa.binary()),
    pa.field("first_frame_latent_shape", pa.list_(pa.int64())),
    pa.field("first_frame_latent_dtype", pa.string()),
    # I2V Validation
    pa.field("pil_image_bytes", pa.binary()),
    pa.field("pil_image_shape", pa.list_(pa.int64())),
    pa.field("pil_image_dtype", pa.string()),
    # --- ODE Trajectory ---
    pa.field("trajectory_latents_bytes", pa.binary()),
    pa.field("trajectory_latents_shape", pa.list_(pa.int64())),
    pa.field("trajectory_latents_dtype", pa.string()),
    pa.field("trajectory_timesteps_bytes", pa.binary()),
    pa.field("trajectory_timesteps_shape", pa.list_(pa.int64())),
    pa.field("trajectory_timesteps_dtype", pa.string()),
    # --- Metadata ---
    pa.field("file_name", pa.string()),
    pa.field("caption", pa.string()),
    pa.field("media_type", pa.string()),  # 'image' or 'video'
])
