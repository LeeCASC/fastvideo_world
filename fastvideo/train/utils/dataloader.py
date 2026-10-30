# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from fastvideo.train.utils.training_config import (
        DataConfig, )


def build_parquet_t2v_train_dataloader(
    data_config: DataConfig,
    *,
    text_len: int,
    parquet_schema: Any,
    global_indices: Sequence[int] | None = None,
    allow_duplicate_global_indices: bool = False,
) -> Any:
    """Build a parquet dataloader for T2V-style datasets."""

    from fastvideo.dataset import (
        build_parquet_map_style_dataloader, )

    _dataset, dataloader = (build_parquet_map_style_dataloader(
        data_config.data_path,
        data_config.train_batch_size,
        num_data_workers=(data_config.dataloader_num_workers),
        parquet_schema=parquet_schema,
        cfg_rate=data_config.training_cfg_rate,
        drop_last=True,
        text_padding_length=int(text_len),
        seed=int(data_config.seed or 0),
        global_indices=global_indices,
        allow_duplicate_global_indices=allow_duplicate_global_indices,
    ))
    return dataloader


def build_motion_balanced_indices(
    data_path: str,
    *,
    action_channel_scales: Sequence[float],
    seed: int,
    num_buckets: int = 4,
) -> list[int]:
    """Interleave equal-count action-motion quantiles for DP/SP sampling."""
    import pyarrow.parquet as pq

    root = Path(data_path).expanduser().resolve()
    parquet_files = sorted(root.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {root}")
    scales = np.asarray(action_channel_scales, dtype=np.float32)
    if scales.shape != (6, ) or bool((scales <= 0.0).any()):
        raise ValueError("action_channel_scales must contain six positive values")
    if num_buckets <= 1:
        raise ValueError("motion-balanced sampling needs at least two buckets")

    scores: list[float] = []
    columns = [
        "action_continuous_bytes",
        "action_continuous_shape",
        "action_valid_mask_bytes",
        "action_valid_mask_shape",
    ]
    for parquet_path in parquet_files:
        data = pq.read_table(parquet_path, columns=columns).to_pydict()
        rows = zip(*(data[column] for column in columns), strict=True)
        for action_bytes, action_shape, mask_bytes, mask_shape in rows:
            actions = np.frombuffer(action_bytes, dtype=np.float32).reshape(action_shape)
            valid = np.frombuffer(mask_bytes, dtype=np.float32).reshape(mask_shape) > 0.0
            normalized = np.clip(actions / scales, -5.0, 5.0)
            magnitudes = np.linalg.vector_norm(normalized, axis=-1)
            scores.append(float(magnitudes[valid].mean()) if bool(valid.any()) else 0.0)

    sorted_indices = np.argsort(np.asarray(scores), kind="stable")
    buckets = [bucket.copy() for bucket in np.array_split(sorted_indices, num_buckets)]
    bucket_size = min(len(bucket) for bucket in buckets)
    if bucket_size == 0:
        raise ValueError(f"Dataset with {len(scores)} samples is too small for {num_buckets} buckets")
    rng = np.random.default_rng(seed)
    for bucket in buckets:
        rng.shuffle(bucket)
    interleaved = np.stack([bucket[:bucket_size] for bucket in buckets], axis=1).reshape(-1)
    return [int(index) for index in interleaved]


def build_parquet_matrixgame2_train_dataloader(
    data_config: DataConfig,
    *,
    parquet_schema: Any,
) -> Any:
    """Build a parquet dataloader for Matrix-Game 2.0 datasets."""

    from fastvideo.dataset import (
        build_parquet_map_style_dataloader, )

    _dataset, dataloader = (build_parquet_map_style_dataloader(
        data_config.data_path,
        data_config.train_batch_size,
        num_data_workers=(data_config.dataloader_num_workers),
        parquet_schema=parquet_schema,
        cfg_rate=float(data_config.training_cfg_rate or 0.0),
        drop_last=True,
        text_padding_length=512,
        seed=int(data_config.seed or 0),
    ))
    return dataloader
