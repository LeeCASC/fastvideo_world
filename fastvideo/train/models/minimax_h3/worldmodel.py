# SPDX-License-Identifier: Apache-2.0
"""Video-only Stage-0 world-model training on the MiniMax-H3 backbone."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, TYPE_CHECKING

import torch

from fastvideo.distributed import get_sp_group
from fastvideo.forward_context import set_forward_context
from fastvideo.logger import init_logger
from fastvideo.models.dits.minimax_h3_prope import camera_projection_matrices, relative_camera_actions
from fastvideo.pipelines import TrainingBatch
from fastvideo.pipelines.basic.minimax_h3.packing import (
    MINIMAX_H3_KEYFRAME_NOISE_AUG,
    MINIMAX_H3_VIDEO_TAG,
    MiniMaxH3PackedLayout,
    build_packed_sequence,
    build_row_timesteps,
    keyframe_condition_noise,
    patchify_video_latents,
    unpatchify_video_tokens,
)
from fastvideo.train.models.base import NoisePrediction
from fastvideo.train.models.minimax_h3.minimax_h3 import (
    _AUDIO_LATENT_CHANNELS,
    _VIDEO_LATENT_CHANNELS,
    MiniMaxH3Model,
)

if TYPE_CHECKING:
    from fastvideo.train.utils.training_config import TrainingConfig

logger = init_logger(__name__)


class MiniMaxH3WorldModel(MiniMaxH3Model):
    """Train H3 video flow from first-frame, text, and camera actions only."""

    _preprocessed_data_type = "h3_worldmodel"
    camera_conditioning = "additive"
    normalize_camera_translation = True
    action_representation = "incremental"
    minimum_video_noise_sigma = 0.0
    fixed_video_noise_sigma: float | None = None
    action_fourier_frequencies = 0
    train_lora = True

    def __init__(
            self,
            *,
            camera_conditioning: Literal["additive", "prope", "hybrid"] = "additive",
            action_representation: Literal["incremental", "relative_pose", "relative_pose_delta"] = "incremental",
            paired_supervision: bool = False,
            trainable_modules: tuple[str, ...] | list[str] | None = None,
            action_channel_scales: tuple[float, ...] | list[float] = (
                0.03,
                0.04,
                0.02,
                0.01,
                0.05,
                0.005,
            ),
            action_normalized_clip: float = 5.0,
            action_normalization_mode: Literal["clamp", "asinh"] = "clamp",
            action_embedding_scale: float = 16.0,
            action_fourier_frequencies: int = 0,
            action_learning_rate: float = 1.0e-4,
            prope_learning_rate: float = 1.0e-4,
            prope_strength: float = 1.0,
            prope_base_strength: float = 0.0,
            prope_use_camera_intrinsics: bool = False,
            normalize_camera_translation: bool = True,
            require_rebased_camera: bool = False,
            action_motion_loss_weight: float = 2.0,
            minimum_video_noise_sigma: float = 0.0,
            fixed_video_noise_sigma: float | None = None,
            neutralize_text_conditioning: bool = False,
            vision_only_conditioning: bool = False,
            train_lora: bool = True,
            motion_balanced_sampling: bool = False,
            motion_balanced_num_buckets: int = 4,
            sampling_index_path: str = "",
            init_from_training_checkpoint: str = "",
            checkpoint_init_modules: tuple[str, ...] | list[str] = (),
            **kwargs: Any,
    ) -> None:
        self.camera_conditioning = str(camera_conditioning)
        if self.camera_conditioning not in {"additive", "prope", "hybrid"}:
            raise ValueError("camera_conditioning must be additive, prope, or hybrid")
        self.action_representation = str(action_representation)
        if self.action_representation not in {"incremental", "relative_pose", "relative_pose_delta"}:
            raise ValueError("action_representation must be incremental, relative_pose, or relative_pose_delta")
        if self.action_representation in {"relative_pose", "relative_pose_delta"
                                          } and self.camera_conditioning != "hybrid":
            raise ValueError("relative-pose actions require hybrid camera conditioning")
        self.paired_supervision = bool(paired_supervision)
        if self.paired_supervision and self.camera_conditioning != "hybrid":
            raise ValueError("paired_supervision requires hybrid camera conditioning")
        self._preprocessed_data_type = ("h3_worldmodel_prope_pair" if self.paired_supervision else "h3_worldmodel_prope"
                                        if self.camera_conditioning in {"prope", "hybrid"} else "h3_worldmodel")
        super().__init__(**kwargs)
        arch = self.transformer.config.arch_config
        self.action_fourier_frequencies = int(action_fourier_frequencies)
        if self.action_fourier_frequencies < 0:
            raise ValueError("action_fourier_frequencies must be non-negative")
        if self.action_representation == "relative_pose_delta" and self.action_fourier_frequencies != 0:
            raise ValueError("relative_pose_delta requires action_fourier_frequencies=0")
        action_feature_dim = (12 if self.action_representation == "relative_pose_delta" else
                              6 if self.action_fourier_frequencies == 0 else
                              (6 * (1 + 2 * self.action_fourier_frequencies) + 1))
        if self.camera_conditioning == "additive":
            if arch.action_dim != action_feature_dim:
                raise ValueError("additive MiniMaxH3WorldModel action_dim must match its action feature encoder; "
                                 f"expected {action_feature_dim}, got {arch.action_dim}")
            if arch.prope_dim_range is not None:
                raise ValueError("additive MiniMaxH3WorldModel must disable prope_dim_range")
        elif self.camera_conditioning == "prope":
            if arch.action_dim != 0:
                raise ValueError("PRoPE MiniMaxH3WorldModel requires pipeline.dit_config.action_dim=0")
            if arch.prope_dim_range is None:
                raise ValueError("PRoPE MiniMaxH3WorldModel requires pipeline.dit_config.prope_dim_range")
        else:
            if arch.action_dim != action_feature_dim or arch.prope_dim_range is None:
                raise ValueError("hybrid MiniMaxH3WorldModel action_dim must match its action feature encoder "
                                 f"({action_feature_dim}) and prope_dim_range must be enabled")
        self.train_lora = bool(train_lora)
        if self._num_lora_layers <= 0:
            raise ValueError("MiniMaxH3WorldModel requires enabled LoRA configuration")
        if not self.train_lora:
            for name, parameter in self.transformer.named_parameters():
                if name.endswith((".lora_A", ".lora_B")):
                    parameter.requires_grad_(False)
        if len(action_channel_scales) != 6 or any(float(value) <= 0.0 for value in action_channel_scales):
            raise ValueError("action_channel_scales must contain six positive values")
        if action_normalized_clip <= 0.0 or action_embedding_scale <= 0.0:
            raise ValueError("action_normalized_clip and action_embedding_scale must be positive")
        if (action_learning_rate <= 0.0 or prope_learning_rate <= 0.0 or prope_strength <= 0.0
                or action_motion_loss_weight < 0.0):
            raise ValueError("adapter learning rates and prope_strength must be positive; "
                             "action_motion_loss_weight must be non-negative")
        self.action_channel_scales = tuple(float(value) for value in action_channel_scales)
        self.action_normalized_clip = float(action_normalized_clip)
        self.action_normalization_mode = str(action_normalization_mode)
        if self.action_normalization_mode not in {"clamp", "asinh"}:
            raise ValueError("action_normalization_mode must be clamp or asinh")
        self.action_embedding_scale = float(action_embedding_scale)
        self.action_learning_rate = float(action_learning_rate)
        self.prope_learning_rate = float(prope_learning_rate)
        self.prope_strength = float(prope_strength)
        self.prope_base_strength = float(prope_base_strength)
        if not 0.0 <= self.prope_base_strength <= 1.0:
            raise ValueError("prope_base_strength must be in [0,1]")
        self.prope_use_camera_intrinsics = bool(prope_use_camera_intrinsics)
        self.normalize_camera_translation = bool(normalize_camera_translation)
        self.require_rebased_camera = bool(require_rebased_camera)
        self.action_motion_loss_weight = float(action_motion_loss_weight)
        self.minimum_video_noise_sigma = float(minimum_video_noise_sigma)
        if not 0.0 <= self.minimum_video_noise_sigma < 1.0:
            raise ValueError("minimum_video_noise_sigma must be in [0,1)")
        self.fixed_video_noise_sigma = None if fixed_video_noise_sigma is None else float(fixed_video_noise_sigma)
        if self.fixed_video_noise_sigma is not None and not 0.0 < self.fixed_video_noise_sigma <= 1.0:
            raise ValueError("fixed_video_noise_sigma must be in (0,1]")
        self.neutralize_text_conditioning = bool(neutralize_text_conditioning)
        self.vision_only_conditioning = bool(vision_only_conditioning)
        if self.neutralize_text_conditioning and self.vision_only_conditioning:
            raise ValueError("neutralize_text_conditioning and vision_only_conditioning are mutually exclusive")
        self.motion_balanced_sampling = bool(motion_balanced_sampling)
        # YAML null should mean that no fixed sampling plan is used.  Converting
        # None directly to the string "None" makes init_preprocessors try to
        # open a file with that name and abort every distributed worker.
        self.sampling_index_path = "" if sampling_index_path is None else str(sampling_index_path)
        if self.motion_balanced_sampling and self.sampling_index_path:
            raise ValueError("motion_balanced_sampling and sampling_index_path are mutually exclusive")
        self.motion_balanced_num_buckets = int(motion_balanced_num_buckets)
        if self.motion_balanced_num_buckets <= 1:
            raise ValueError("motion_balanced_num_buckets must be at least two")
        if trainable_modules is None:
            trainable_modules = (("action_adapter", "proj_in", "norm_out",
                                  "proj_out") if self.camera_conditioning in {"additive", "hybrid"} else
                                 ("proj_in", "norm_out", "proj_out"))
        self.trainable_modules = tuple(str(name) for name in trainable_modules)
        checkpoint_init_modules = tuple(str(name) for name in checkpoint_init_modules)
        self._enable_named_modules((*self.trainable_modules, *checkpoint_init_modules))
        if self.camera_conditioning in {"prope", "hybrid"}:
            prope_blocks = [
                block for block in self.transformer.transformer_blocks if block.attn.prope_dim_range is not None
            ]
            if not prope_blocks:
                raise RuntimeError("H3 PRoPE configuration did not enable any attention blocks")
            for block in prope_blocks:
                if block.attn.prope_gate is not None:
                    block.attn.prope_gate.requires_grad_(True)
                if block.attn.prope_o is not None:
                    block.attn.prope_o.requires_grad_(True)
                block.attn.prope_strength = self.prope_strength
                block.attn.prope_base_strength = self.prope_base_strength
        if init_from_training_checkpoint:
            self._load_training_checkpoint(init_from_training_checkpoint)
        frozen_checkpoint_modules = set(checkpoint_init_modules) - set(self.trainable_modules)
        self._set_named_modules_trainable(frozen_checkpoint_modules, trainable=False)
        self._audit_trainable_parameters()

    def optimizer_param_groups(
        self,
        *,
        learning_rate: float,
        weight_decay: float,
    ) -> list[dict[str, Any]]:
        """Give new camera-conditioning parameters a faster unregularized group."""
        parameter_groups: list[tuple[str, list[torch.nn.Parameter], float]] = []
        if self.camera_conditioning in {"additive", "hybrid"}:
            action_parameters: list[torch.nn.Parameter] = []
            for module_name in ("action_adapter", "action_encoder", "action_block_gates"):
                module = getattr(self.transformer, module_name, None)
                if module is not None:
                    action_parameters.extend(parameter for parameter in module.parameters() if parameter.requires_grad)
            parameter_groups.append(("action_adapter", action_parameters, self.action_learning_rate))
        if self.camera_conditioning in {"prope", "hybrid"}:
            prope_parameters = [
                block.attn.prope_gate for block in self.transformer.transformer_blocks
                if block.attn.prope_gate is not None and block.attn.prope_gate.requires_grad
            ]
            prope_parameters.extend(parameter for block in self.transformer.transformer_blocks
                                    if block.attn.prope_o is not None for parameter in block.attn.prope_o.parameters()
                                    if parameter.requires_grad)
            if prope_parameters:
                parameter_groups.append(("prope_camera", prope_parameters, self.prope_learning_rate))
        empty_groups = [name for name, parameters, _ in parameter_groups if not parameters]
        if empty_groups:
            raise RuntimeError(f"H3 Stage-0 optimizer has empty camera parameter groups: {empty_groups}")
        adapter_parameters = [parameter for _, parameters, _ in parameter_groups for parameter in parameters]
        adapter_parameter_ids = {id(parameter) for parameter in adapter_parameters}
        backbone_parameters = [
            parameter for parameter in self.transformer.parameters()
            if parameter.requires_grad and id(parameter) not in adapter_parameter_ids
        ]
        groups: list[dict[str, Any]] = []
        summary: list[str] = []
        if backbone_parameters:
            groups.append({
                "params": backbone_parameters,
                "lr": float(learning_rate),
                "weight_decay": float(weight_decay),
                "name": "pretrained_and_lora",
            })
            summary.append(f"pretrained/LoRA lr={float(learning_rate):.3g} wd={float(weight_decay):.3g}")
        for name, parameters, group_learning_rate in parameter_groups:
            groups.append({
                "params": parameters,
                "lr": group_learning_rate,
                "weight_decay": 0.0,
                "name": name,
            })
            summary.append(f"{name} lr={group_learning_rate:.3g} wd=0")
        logger.info("MiniMaxH3 Stage-0 optimizer groups: %s", "; ".join(summary))
        return groups

    def _normalized_actions(self, actions: torch.Tensor) -> torch.Tensor:
        scales = actions.new_tensor(self.action_channel_scales)
        normalized = actions.float() / scales.float()
        if getattr(self, "action_normalization_mode", "clamp") == "asinh":
            # Unlike clipping, this remains monotone for large cumulative camera motion.
            normalized = torch.asinh(normalized)
        else:
            normalized = normalized.clamp(min=-self.action_normalized_clip, max=self.action_normalized_clip)
        return (normalized * self.action_embedding_scale).to(actions.dtype)

    def _action_features(self, actions: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
        """Encode cumulative pose and distinguish controlled-static from no control."""
        if actions.shape[-1] == 12:
            normalized = torch.cat(
                (self._normalized_actions(actions[..., :6]), self._normalized_actions(actions[..., 6:])),
                dim=-1).float()
        else:
            normalized = self._normalized_actions(actions).float()
        valid = action_mask.unsqueeze(-1).float()
        if self.action_fourier_frequencies == 0:
            return (normalized * valid).to(actions.dtype)
        unit_pose = normalized / self.action_embedding_scale
        features = [normalized]
        for frequency in range(self.action_fourier_frequencies):
            phase = unit_pose * (2**frequency) * torch.pi
            features.extend(
                (phase.sin() * self.action_embedding_scale, (phase.cos() - 1.0) * self.action_embedding_scale))
        # Static targets remain explicitly controlled through this last
        # channel, whereas first-frame condition rows are constructed as zero.
        features.append(valid * self.action_embedding_scale)
        return (torch.cat(features, dim=-1) * valid).to(actions.dtype)

    def _video_loss_weights(self, actions: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
        """Return mean-one frame weights emphasizing camera-motion targets."""
        normalized = self._normalized_actions(actions).float() / self.action_embedding_scale
        motion = normalized.square().sum(dim=-1).sqrt() * action_mask.float()
        mean_motion = motion.mean(dim=1, keepdim=True)
        relative_motion = torch.where(
            mean_motion > 1.0e-6,
            motion / mean_motion.clamp_min(1.0e-6),
            torch.zeros_like(motion),
        )
        weights = 1.0 + self.action_motion_loss_weight * relative_motion
        return weights / weights.mean(dim=1, keepdim=True)

    def _enable_named_modules(self, module_names: tuple[str, ...]) -> None:
        named_modules = dict(self.transformer.named_modules())
        resolved = {name: self._resolve_trainable_module_name(name, named_modules) for name in module_names}
        missing = [name for name, actual in resolved.items() if actual is None]
        if missing:
            raise ValueError(f"Unknown H3 modules: {missing}")
        for actual in resolved.values():
            assert actual is not None
            named_modules[actual].requires_grad_(True)

    @staticmethod
    def _resolve_trainable_module_name(name: str, named_modules: dict[str, torch.nn.Module]) -> str | None:
        if name in named_modules:
            return name
        if name.startswith("transformer_blocks."):
            block, separator, remainder = name.partition(".attn.")
            if separator:
                wrapped = f"{block}._checkpoint_wrapped_module.attn.{remainder}"
                if wrapped in named_modules:
                    return wrapped
        return None

    def _set_named_modules_trainable(self, module_names: set[str], *, trainable: bool) -> None:
        named_modules = dict(self.transformer.named_modules())
        for name in module_names:
            actual = self._resolve_trainable_module_name(name, named_modules)
            if actual is None:
                raise ValueError(f"Unknown H3 module: {name}")
            named_modules[actual].requires_grad_(trainable)

    def _load_training_checkpoint(self, checkpoint_path: str) -> None:
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner

        from fastvideo.training.checkpointing_utils import ModelWrapper

        checkpoint = Path(checkpoint_path).expanduser().resolve()
        dcp_path = checkpoint if checkpoint.name == "dcp" else checkpoint / "dcp"
        if not dcp_path.is_dir():
            raise FileNotFoundError(f"Training checkpoint DCP directory not found: {dcp_path}")
        state = {"roles.student.transformer": ModelWrapper(self.transformer)}
        # A Stage-0 experiment may add new action/PRoPE parameters (for example,
        # enabling gates on every H3 block) while reusing an earlier Stage-0
        # checkpoint.  DCP's default planner is strict about keys present in the
        # current model but absent from the checkpoint, even though the model
        # has already initialized those parameters.  Partial loading preserves
        # all matching pretrained/LoRA weights and leaves genuinely new
        # parameters at their initialized values.
        dcp.load(
            state,
            checkpoint_id=str(dcp_path),
            planner=DefaultLoadPlanner(allow_partial_load=True),
        )
        logger.info("Initialized MiniMax-H3 model weights from %s without optimizer state", checkpoint)

    def _audit_trainable_parameters(self) -> None:
        """Fail closed if Stage-0 accidentally unfreezes H3 backbone weights."""
        trainable: list[tuple[str, int]] = []
        unexpected: list[str] = []
        important_counts = {name: 0 for name in self.trainable_modules}
        lora_count = 0
        prope_gate_count = 0
        for name, parameter in self.transformer.named_parameters():
            if not parameter.requires_grad:
                continue
            trainable.append((name, parameter.numel()))
            canonical_name = name.replace("._checkpoint_wrapped_module", "")
            important = next(
                (module_name for module_name in self.trainable_modules
                 if canonical_name == module_name or canonical_name.startswith(f"{module_name}.")),
                None,
            )
            if important is not None:
                important_counts[important] += parameter.numel()
            elif name.endswith((".lora_A", ".lora_B")):
                lora_count += parameter.numel()
            elif self.camera_conditioning in {"prope", "hybrid"} and name.endswith(
                    ".prope_gate") or self.camera_conditioning in {"prope", "hybrid"} and ".prope_o." in name:
                prope_gate_count += parameter.numel()
            else:
                unexpected.append(name)

        if not trainable:
            raise RuntimeError("MiniMaxH3WorldModel has no trainable parameters")
        empty_modules = [name for name, count in important_counts.items() if count == 0]
        if empty_modules:
            raise RuntimeError(f"H3 trainable modules contain no parameters: {empty_modules}")
        if self.train_lora and lora_count == 0:
            raise RuntimeError("MiniMaxH3WorldModel has no trainable LoRA parameters")
        if not self.train_lora and lora_count != 0:
            raise RuntimeError("MiniMaxH3WorldModel train_lora=false left trainable LoRA parameters")
        if (self.camera_conditioning in {"prope", "hybrid"}
                and self.transformer.config.arch_config.prope_mode in {"gated", "residual"} and prope_gate_count == 0):
            raise RuntimeError("MiniMaxH3WorldModel has no trainable PRoPE camera parameters")
        if unexpected:
            preview = unexpected[:20]
            suffix = "" if len(unexpected) <= len(preview) else f" (+{len(unexpected) - len(preview)} more)"
            raise RuntimeError(f"Unexpected trainable H3 backbone parameters: {preview}{suffix}")

        total = sum(count for _, count in trainable)
        logger.info(
            "MiniMaxH3 Stage-0 trainable audit passed: %.3fM parameters "
            "(LoRA %.3fM; PRoPE gates %.3fM; selected layers %.3fM)",
            total / 1e6,
            lora_count / 1e6,
            prope_gate_count / 1e6,
            (total - lora_count - prope_gate_count) / 1e6,
        )

    def init_preprocessors(self, training_config: TrainingConfig) -> None:
        from fastvideo.dataset.dataloader.schema import (
            pyarrow_schema_minimax_h3_worldmodel,
            pyarrow_schema_minimax_h3_worldmodel_prope,
            pyarrow_schema_minimax_h3_worldmodel_prope_pair,
        )
        from fastvideo.train.utils.dataloader import (
            build_motion_balanced_indices,
            build_parquet_t2v_train_dataloader,
        )

        self.sp_group = get_sp_group()
        text_config = training_config.pipeline_config.text_encoder_configs[0]  # type: ignore[union-attr]
        global_indices = None
        if self.sampling_index_path:
            global_indices = json.loads(Path(self.sampling_index_path).read_text())
            if not isinstance(global_indices, list) or not all(type(index) is int for index in global_indices):
                raise ValueError("sampling_index_path must contain one JSON array of integer parquet indices")
            logger.info("Using audited, repeat-aware H3 sampling plan with %d draws", len(global_indices))
        elif self.motion_balanced_sampling:
            if not isinstance(training_config.data.data_path, str):
                raise TypeError("motion-balanced sampling requires one parquet dataset path")
            global_indices = build_motion_balanced_indices(
                training_config.data.data_path,
                action_channel_scales=self.action_channel_scales,
                seed=int(training_config.data.seed),
                num_buckets=self.motion_balanced_num_buckets,
            )
            logger.info(
                "Enabled motion-balanced sampling with %d buckets over %d rows",
                self.motion_balanced_num_buckets,
                len(global_indices),
            )
        self.dataloader = build_parquet_t2v_train_dataloader(
            training_config.data,
            text_len=int(text_config.arch_config.text_len),
            parquet_schema=(pyarrow_schema_minimax_h3_worldmodel_prope_pair
                            if self.paired_supervision else pyarrow_schema_minimax_h3_worldmodel_prope if
                            self.camera_conditioning in {"prope", "hybrid"} else pyarrow_schema_minimax_h3_worldmodel),
            global_indices=global_indices,
            allow_duplicate_global_indices=bool(self.sampling_index_path),
        )
        self.start_step = 0

    def prepare_batch(
        self,
        raw_batch: dict[str, Any],
        *,
        generator: torch.Generator,
        latents_source: Literal["data", "zeros"] = "data",
    ) -> TrainingBatch:
        if latents_source != "data":
            raise ValueError("MiniMaxH3WorldModel supports only data latents")
        device, dtype = self.device, torch.bfloat16
        video = raw_batch["vae_latent"]
        first = raw_batch["first_frame_latent"]
        if video.ndim != 5 or tuple(video.shape[:2]) != (1, _VIDEO_LATENT_CHANNELS):
            raise ValueError(f"vae_latent must have shape [1,24,T,H,W], got {tuple(video.shape)}")
        if first.shape != (1, 24, 1, video.shape[3], video.shape[4]):
            raise ValueError("first_frame_latent must align with vae_latent spatial geometry")
        video = video.to(device=device, dtype=dtype)
        first = first.to(device=device, dtype=dtype)

        text = raw_batch["text_embedding"]
        text_mask = raw_batch["text_attention_mask"][0].to(torch.bool)
        tags = raw_batch["text_token_tags"][0].to(torch.long)
        valid_text_tokens = int(text_mask.sum())
        # The parquet collator crops embeddings to the model's 512-token
        # context window, while tags are read from the uncropped row.  This
        # is common for Qwen3-VL rows with long scene captions (the new union
        # cache has 358 such rows).  Keep the tag prefix corresponding to the
        # retained embedding rows; a shorter tag array still indicates a
        # corrupt cache and must fail loudly.
        if tags.ndim != 1 or tags.numel() < valid_text_tokens:
            raise ValueError("text_token_tags must cover all non-padded text_embedding rows")
        if tags.numel() != valid_text_tokens:
            logger.warning(
                "Cropping text_token_tags from %d to %d rows to match the padded text embedding",
                tags.numel(), valid_text_tokens,
            )
            tags = tags[:valid_text_tokens]
        actions = raw_batch["action_continuous"].to(device=device, dtype=dtype)
        action_mask = raw_batch["action_valid_mask"].to(device=device, dtype=torch.bool)
        if actions.shape != (1, video.shape[2], 6) or action_mask.shape != actions.shape[:2]:
            raise ValueError("camera actions must have shape [1,T,6] with an aligned validity mask")
        camera_to_world: torch.Tensor | None = None
        camera_intrinsics: torch.Tensor | None = None
        if self.camera_conditioning in {"prope", "hybrid"}:
            camera_to_world = raw_batch["camera_to_world"].to(device=device, dtype=torch.float32)
            camera_intrinsics = raw_batch["camera_intrinsics"].to(device=device, dtype=torch.float32)
            expected_c2w = (1, video.shape[2], 4, 4)
            expected_intrinsics = (1, video.shape[2], 3, 3)
            if camera_to_world.shape != expected_c2w or camera_intrinsics.shape != expected_intrinsics:
                raise ValueError(f"PRoPE camera geometry must have shapes {expected_c2w} and {expected_intrinsics}; "
                                 f"got {tuple(camera_to_world.shape)} and {tuple(camera_intrinsics.shape)}")
            if getattr(self, "require_rebased_camera", False):
                identity = torch.eye(4, device=device, dtype=torch.float32)
                if not bool(torch.isfinite(camera_to_world).all() and torch.isfinite(camera_intrinsics).all()):
                    raise ValueError("PRoPE camera geometry contains non-finite values")
                if not bool(torch.allclose(camera_to_world[:, 0], identity[None], atol=1.0e-3, rtol=0.0)):
                    raise ValueError("PRoPE camera trajectory must be rebased to the first latent frame")
                rotations = camera_to_world[..., :3, :3]
                orthogonality = rotations.transpose(-1, -2) @ rotations
                if not bool(torch.allclose(orthogonality, identity[:3, :3], atol=2.0e-2, rtol=0.0)):
                    raise ValueError("PRoPE camera rotations must be orthogonal")
                if not bool((torch.linalg.det(rotations) > 0.95).all()):
                    raise ValueError("PRoPE camera rotations must be proper, not reflected")
                if not bool(torch.allclose(camera_to_world[..., 3, :], identity[3, :], atol=1.0e-3, rtol=0.0)):
                    raise ValueError("PRoPE camera matrices must have homogeneous bottom rows")
                if not bool((camera_intrinsics[..., 0, 0] > 0).all() and (camera_intrinsics[..., 1, 1] > 0).all()):
                    raise ValueError("PRoPE camera intrinsics must have positive focal lengths")

        noise = torch.randn(video.shape, generator=generator, device=device, dtype=dtype)
        noise_amount, _ = self._sample_noise_amounts(generator, device)
        if self.fixed_video_noise_sigma is not None:
            noise_amount = torch.full_like(noise_amount, self.fixed_video_noise_sigma)
        else:
            noise_amount = noise_amount.clamp_min(self.minimum_video_noise_sigma)
        sigma = noise_amount.to(dtype).view(1, 1, 1, 1, 1)
        noisy = (1.0 - sigma) * video + sigma * noise
        _, _, frames, height, width = video.shape
        selected_text = text[:, text_mask].to(device=device, dtype=dtype)
        if getattr(self, "vision_only_conditioning", False):
            vision_mask = tags == MINIMAX_H3_VIDEO_TAG
            if not bool(vision_mask.any()):
                raise ValueError("vision_only_conditioning requires first-frame vision tokens")
            selected_text = selected_text[:, vision_mask.to(selected_text.device)]
            tags = tags[vision_mask]
        elif getattr(self, "neutralize_text_conditioning", False):
            selected_text = torch.zeros_like(selected_text)
        layout = build_packed_sequence(tags, frames, height, width, 0, self.transformer.patch_size, ("first", ))
        first_rows = patchify_video_latents(first, self.transformer.patch_size)
        condition_noise = keyframe_condition_noise(
            ((1, height, width), ),
            self.transformer.patch_size,
            _VIDEO_LATENT_CHANNELS,
            generator=generator,
            device=device,
            dtype=dtype,
        )
        condition_rows = self.noise_scheduler.scale_noise(
            first_rows,
            MINIMAX_H3_KEYFRAME_NOISE_AUG,
            condition_noise,
        )

        input_kwargs: dict[str, Any] = {
            "condition_video_rows": condition_rows,
            "actions": actions,
            "action_mask": action_mask,
            "video_loss_weights": self._video_loss_weights(actions, action_mask),
        }
        if camera_to_world is not None and camera_intrinsics is not None:
            input_kwargs.update({
                "camera_to_world": camera_to_world,
                "camera_intrinsics": camera_intrinsics,
            })
        if "pair_vae_latent" in raw_batch:
            partner = raw_batch["pair_vae_latent"].to(device=device, dtype=dtype)
            pair_camera = raw_batch["pair_camera_to_world"].to(device=device, dtype=torch.float32)
            pair_intrinsics = raw_batch["pair_camera_intrinsics"].to(device=device, dtype=torch.float32)
            pair_mask = raw_batch["pair_action_valid_mask"].to(device=device, dtype=torch.bool)
            if partner.shape != video.shape or pair_camera.shape != camera_to_world.shape or pair_intrinsics.shape != camera_intrinsics.shape:
                raise ValueError("paired video and camera tensors must align with the primary sample")
            if pair_mask.shape != action_mask.shape or not bool(
                    torch.allclose(pair_camera[:, 0], torch.eye(4, device=device)[None], atol=1.0e-3)):
                raise ValueError("paired action mask or first-frame camera is invalid")
            input_kwargs.update({
                "pair_latents": partner.permute(0, 2, 1, 3, 4),
                "pair_camera_to_world": pair_camera,
                "pair_camera_intrinsics": pair_intrinsics,
                "pair_action_mask": pair_mask,
            })
        batch = TrainingBatch(
            latents=video.permute(0, 2, 1, 3, 4),
            raw_latent_shape=tuple(video.shape),
            encoder_hidden_states=selected_text,
            encoder_attention_mask=torch.ones((1, len(tags)), device=device, dtype=dtype),
            infos=raw_batch.get("info_list"),
            noisy_model_input=noisy,
            noise=noise,
            sigmas=sigma,
            timesteps=1.0 - noise_amount,
            minimax_h3_layout=layout,
            input_kwargs=input_kwargs,
        )
        return batch

    def _packed_prope_matrices(
        self,
        layout: MiniMaxH3PackedLayout,
        camera_to_world: torch.Tensor,
        camera_intrinsics: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Expand latent-frame camera projections over H3 packed rows."""
        projection, inverse = camera_projection_matrices(
            camera_to_world,
            camera_intrinsics,
            image_width=int(self.training_config.data.num_width),
            image_height=int(self.training_config.data.num_height),
            use_camera_intrinsics=self.prope_use_camera_intrinsics,
            normalize_translation=self.normalize_camera_translation,
        )
        rows_per_frame = (layout.latent_height // self.transformer.patch_size[1]) * (layout.latent_width //
                                                                                     self.transformer.patch_size[2])
        target_projection = projection.repeat_interleave(rows_per_frame, dim=1)
        target_inverse = inverse.repeat_interleave(rows_per_frame, dim=1)
        condition_projection = projection[:, :1].expand(-1, layout.num_condition_video_rows, -1, -1)
        condition_inverse = inverse[:, :1].expand(-1, layout.num_condition_video_rows, -1, -1)
        video_projection = torch.cat((condition_projection, target_projection), dim=1)
        video_inverse = torch.cat((condition_inverse, target_inverse), dim=1)
        sequence_length = int(layout.position_ids.shape[0])
        # Give non-video tokens the reference-camera transform. With a static
        # trajectory this makes PRoPE an exact global basis change for the
        # entire packed attention document, including text-video attention.
        packed_projection = projection[:, :1].expand(-1, sequence_length, -1, -1).clone()
        packed_inverse = inverse[:, :1].expand(-1, sequence_length, -1, -1).clone()
        video_indices = layout.video_indices.to(camera_to_world.device)
        packed_projection.index_copy_(1, video_indices, video_projection)
        packed_inverse.index_copy_(1, video_indices, video_inverse)
        return packed_projection, packed_inverse

    def predict_noise(
        self,
        noisy_latents: torch.Tensor,
        timestep: torch.Tensor,
        batch: TrainingBatch,
        *,
        conditional: bool,
        cfg_uncond: dict[str, Any] | None = None,
        attn_kind: Literal["dense", "vsa"] = "dense",
    ) -> NoisePrediction:
        del timestep
        if not conditional or cfg_uncond is not None or attn_kind != "dense":
            raise ValueError("MiniMaxH3WorldModel requires one dense conditional sample")
        layout = batch.minimax_h3_layout
        if not isinstance(layout, MiniMaxH3PackedLayout) or batch.input_kwargs is None:
            raise RuntimeError("prepare_batch() must set the H3 layout and world-model inputs")
        condition_rows = batch.input_kwargs["condition_video_rows"]
        video_bcthw = noisy_latents.permute(0, 2, 1, 3, 4).to(torch.bfloat16)
        target_rows = patchify_video_latents(video_bcthw, self.transformer.patch_size)
        video_rows = torch.cat((condition_rows, target_rows))[None]
        action_rows: torch.Tensor | None = None
        prope_projection: torch.Tensor | None = None
        prope_inverse: torch.Tensor | None = None
        if self.camera_conditioning in {"additive", "hybrid"}:
            if self.action_representation in {"relative_pose", "relative_pose_delta"}:
                actions = relative_camera_actions(
                    batch.input_kwargs["camera_to_world"],
                    normalize_translation=self.normalize_camera_translation,
                )
                if self.action_representation == "relative_pose_delta":
                    delta = torch.cat((torch.zeros_like(actions[:, :1]), actions[:, 1:] - actions[:, :-1]), dim=1)
                    actions = torch.cat((actions, delta * max(1, actions.shape[1] - 1)), dim=-1)
            else:
                actions = batch.input_kwargs["actions"]
            actions = self._action_features(actions, batch.input_kwargs["action_mask"])
            rows_per_frame = (layout.latent_height // self.transformer.patch_size[1]) * (layout.latent_width //
                                                                                         self.transformer.patch_size[2])
            action_rows = actions.repeat_interleave(rows_per_frame, dim=1)
            action_rows = torch.cat(
                (
                    torch.zeros(
                        (1, layout.num_condition_video_rows, actions.shape[-1]),
                        device=self.device,
                        dtype=action_rows.dtype,
                    ),
                    action_rows,
                ),
                dim=1,
            )
        if self.camera_conditioning in {"prope", "hybrid"}:
            prope_projection, prope_inverse = self._packed_prope_matrices(
                layout,
                batch.input_kwargs["camera_to_world"],
                batch.input_kwargs["camera_intrinsics"],
            )
        unique, inverse = build_row_timesteps(
            layout,
            video_timestep=float(batch.timesteps[0]),
            audio_timestep=float(batch.timesteps[0]),
            condition_video_timestep=max(float(batch.timesteps[0]), MINIMAX_H3_KEYFRAME_NOISE_AUG),
            condition_audio_timestep=1.0,
        )
        with (
                torch.autocast(self.device.type, dtype=torch.bfloat16),
                set_forward_context(current_timestep=unique.to(self.device), attn_metadata=None),
        ):
            velocity, _ = self.transformer(
                hidden_states=video_rows,
                audio_hidden_states=torch.empty((1, 0, _AUDIO_LATENT_CHANNELS),
                                                device=self.device,
                                                dtype=torch.bfloat16),
                encoder_hidden_states=batch.encoder_hidden_states,
                timestep=unique.to(self.device),
                timestep_indices=inverse.to(self.device),
                token_tags=layout.token_tags.to(self.device),
                position_ids=layout.position_ids.to(self.device),
                video_indices=layout.video_indices.to(self.device),
                audio_indices=layout.audio_indices.to(self.device),
                text_indices=layout.text_indices.to(self.device),
                action_hidden_states=action_rows,
                prope_projection=prope_projection,
                prope_inverse=prope_inverse,
            )
        target_velocity = velocity[:, layout.num_condition_video_rows:]
        prediction = unpatchify_video_tokens(
            target_velocity,
            layout.num_video_latent_frames,
            layout.latent_height,
            layout.latent_width,
            _VIDEO_LATENT_CHANNELS,
            self.transformer.patch_size,
        ).permute(0, 2, 1, 3, 4)
        return -prediction


__all__ = ["MiniMaxH3WorldModel"]
