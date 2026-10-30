# SPDX-License-Identifier: Apache-2.0
"""Architecture configuration for the MiniMax H3 joint audio-video DiT."""

from __future__ import annotations

from dataclasses import dataclass, field

from fastvideo.configs.models.dits.base import DiTArchConfig, DiTConfig
from fastvideo.platforms import AttentionBackendEnum


def _is_minimax_h3_block(name: str, module: object) -> bool:
    """Select the main and text-refiner transformer blocks for FSDP."""
    del module
    parts = name.split(".")
    return ((len(parts) == 2 and parts[0] == "transformer_blocks" and parts[1].isdigit())
            or (len(parts) == 3 and parts[:2] == ["token_refiner", "refiner_blocks"] and parts[2].isdigit()))


@dataclass
class MiniMaxH3ArchConfig(DiTArchConfig):
    """One-to-one representation of the released transformer config."""

    _fsdp_shard_conditions: list = field(default_factory=lambda: [_is_minimax_h3_block])
    _supported_attention_backends: tuple[AttentionBackendEnum, ...] = (
        AttentionBackendEnum.TORCH_SDPA,
        AttentionBackendEnum.FLASH_ATTN,
        # FP4-quantized QK attention (fa4_fp4 on sm_100/sm_103, cutlass on
        # sm_12x). Enabled for speed experiments; output quality against the
        # SSIM references is not yet validated.
        AttentionBackendEnum.ATTN_QAT_INFER,
        AttentionBackendEnum.VIDEO_SPARSE_ATTN_H3,
    )

    param_names_mapping: dict = field(
        default_factory=lambda: {
            r"^time_embedder\.linear_1\.(.*)$": r"time_embedder.fc_in.\1",
            r"^time_embedder\.linear_2\.(.*)$": r"time_embedder.fc_out.\1",
            r"^(.*)\.attn\.to_out\.0\.(.*)$": r"\1.attn.to_out.\2",
            r"^(.*)\.ff\.net\.0\.proj\.(.*)$": r"\1.ff.fc_in.\2",
            r"^(.*)\.ff\.net\.2\.(.*)$": r"\1.ff.fc_out.\2",
        })
    num_attention_heads: int = 56
    attention_head_dim: int = 128
    hidden_size: int = 5376
    num_layers: int = 50
    num_refiner_layers: int = 2
    ffn_dim: int = 14336
    in_channels: int = 24
    audio_in_channels: int = 32
    # Disabled for released H3 checkpoints. World-model training enables a
    # zero-initialized per-video-row camera-action projection with dimension 6.
    action_dim: int = 0
    action_modulation_mode: str = "additive"
    action_adapter_hidden_dim: int = 0
    # Scale and re-apply the projected action residual throughout the main
    # transformer. Interval zero preserves input-only legacy conditioning.
    action_injection_interval: int = 0
    action_post_adaln: bool = False
    # Give each block independent channel-wise camera scale/shift strength.
    # Disabled for existing checkpoints, whose gates contain one scalar.
    action_channelwise_gates: bool = False
    action_residual_scale: float = 1.0
    # Smooth per-channel bound on the projected residual. Zero disables it.
    action_residual_clip: float = 0.0
    # Optional camera-projective positional subspace in every main attention
    # head. H3 native RoPE occupies [0,96), so Stage-0 PRoPE uses [96,108).
    prope_dim_range: tuple[int, int] | None = None
    # Gated preserves the zero-initialized compatibility path. Exact follows
    # Cosmos PRoPE and applies the complete projective basis change.
    prope_mode: str = "gated"
    # Positive: preserve released H3 attention and add a parallel camera
    # attention residual every N blocks. Zero retains in-place PRoPE.
    prope_parallel_residual_interval: int = 0
    patch_size: tuple[int, int, int] = (1, 2, 2)
    text_dim: int = 5120
    freq_dim: int = 256
    time_embed_hidden_dim: int = 5376
    time_embed_dim: int = 2688
    adaln_rank: int | None = None
    rope_freq_dim: int = 16
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    qk_norm_eps: float = 1e-5
    final_norm_eps: float = 1e-5

    def __post_init__(self) -> None:
        super().__post_init__()
        if len(self.patch_size) != 3:
            raise ValueError(f"MiniMax H3 patch_size must have three axes, got {self.patch_size}.")
        self.patch_size = (self.patch_size[0], self.patch_size[1], self.patch_size[2])
        self.num_channels_latents = self.in_channels
        self.out_channels = self.in_channels
        if self.action_dim < 0:
            raise ValueError(f"MiniMax H3 action_dim must be non-negative, got {self.action_dim}.")
        if self.action_modulation_mode not in {"additive", "film"}:
            raise ValueError("MiniMax H3 action_modulation_mode must be additive or film")
        if self.action_adapter_hidden_dim < 0:
            raise ValueError("MiniMax H3 action_adapter_hidden_dim must be non-negative")
        if self.action_modulation_mode == "film" and (self.action_dim == 0 or self.action_adapter_hidden_dim == 0):
            raise ValueError("MiniMax H3 camera FiLM requires positive action_dim and action_adapter_hidden_dim")
        if self.action_injection_interval < 0:
            raise ValueError("MiniMax H3 action_injection_interval must be non-negative, "
                             f"got {self.action_injection_interval}.")
        if self.action_post_adaln and (self.action_modulation_mode != "film" or self.action_injection_interval == 0):
            raise ValueError("action_post_adaln requires film modulation and a positive injection interval")
        if self.action_channelwise_gates and self.action_modulation_mode != "film":
            raise ValueError("action_channelwise_gates requires film modulation")
        if self.action_residual_scale <= 0.0:
            raise ValueError(f"MiniMax H3 action_residual_scale must be positive, got {self.action_residual_scale}.")
        if self.action_residual_clip < 0.0:
            raise ValueError(f"MiniMax H3 action_residual_clip must be non-negative, got {self.action_residual_clip}.")
        if self.prope_parallel_residual_interval < 0:
            raise ValueError("MiniMax H3 prope_parallel_residual_interval must be non-negative, "
                             f"got {self.prope_parallel_residual_interval}.")
        if self.prope_mode not in {"gated", "exact", "residual"}:
            raise ValueError(f"MiniMax H3 prope_mode must be gated, exact, or residual, got {self.prope_mode!r}.")
        if self.prope_mode == "exact" and self.prope_parallel_residual_interval > 0:
            raise ValueError("Exact MiniMax H3 PRoPE is incompatible with parallel residual attention.")
        if self.prope_parallel_residual_interval > 0 and self.prope_dim_range is None:
            raise ValueError("MiniMax H3 parallel PRoPE residuals require prope_dim_range.")
        if self.prope_dim_range is not None:
            if len(self.prope_dim_range) != 2:
                raise ValueError("MiniMax H3 prope_dim_range must contain [start,end].")
            self.prope_dim_range = (int(self.prope_dim_range[0]), int(self.prope_dim_range[1]))
            prope_start, prope_end = self.prope_dim_range
            prope_width = prope_end - prope_start
            if not 0 <= prope_start < prope_end <= self.attention_head_dim:
                raise ValueError("MiniMax H3 prope_dim_range must lie inside one attention head; "
                                 f"got {self.prope_dim_range} for width {self.attention_head_dim}.")
            if self.prope_mode == "residual" and self.prope_dim_range != (0, self.attention_head_dim):
                raise ValueError("Residual PRoPE must transform every attention-head channel")
            if prope_width % (4 if self.prope_mode == "residual" else 12):
                raise ValueError("MiniMax H3 PRoPE width must be divisible by its homogeneous 4-vector groups, "
                                 f"got {prope_width}.")
        if self.adaln_rank is not None and not 0 < self.adaln_rank <= self.time_embed_dim:
            raise ValueError(f"MiniMax H3 adaln_rank must be in (0, time_embed_dim={self.time_embed_dim}], "
                             f"got {self.adaln_rank}.")
        rotary_dim = 2 * 3 * self.rope_freq_dim
        if rotary_dim > self.attention_head_dim or rotary_dim % 2:
            raise ValueError(f"MiniMax H3 rotary width must be even and no larger than the head width; got "
                             f"rotary_dim={rotary_dim}, attention_head_dim={self.attention_head_dim}.")
        if (self.prope_dim_range is not None and self.prope_mode != "residual"
                and self.prope_dim_range[0] < rotary_dim):
            raise ValueError("MiniMax H3 PRoPE must not overlap native RoPE; "
                             f"got rotary_dim={rotary_dim}, prope_dim_range={self.prope_dim_range}.")


@dataclass
class MiniMaxH3Config(DiTConfig):
    """FastVideo component configuration for MiniMax H3 transformers."""

    arch_config: MiniMaxH3ArchConfig = field(default_factory=MiniMaxH3ArchConfig)
    prefix: str = "minimax_h3"
    # FastVideo's Fully Sharded Data Parallel (FSDP) loading path requires one
    # parameter dtype, while H3 inference keeps boundary projections in FP32.
    uniform_parameter_dtype: bool = False
