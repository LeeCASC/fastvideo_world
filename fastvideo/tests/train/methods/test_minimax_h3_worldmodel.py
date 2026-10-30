# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for video-only MiniMax-H3 world-model training."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from fastvideo.models.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler
from fastvideo.train.models.minimax_h3 import MiniMaxH3WorldModel
from fastvideo.train.methods.fine_tuning.worldmodel import action_response_bounds
from fastvideo.train.utils.config import load_run_config

_FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "minimax_h3_worldmodel_min.yaml"
_STAGE0_CONFIG = Path(__file__).resolve().parents[4] / "examples/train/configs/minimax_h3_worldmodel_stage0.yaml"
_STAGE0_V2_CONFIG = Path(__file__).resolve().parents[4] / "examples/train/configs/minimax_h3_worldmodel_stage0_v2.yaml"
_STAGE0_PROPE_CONFIG = (
    Path(__file__).resolve().parents[4] / "examples/train/configs/minimax_h3_worldmodel_stage0_prope.yaml"
)
_RECOVERY_CONFIG = (
    Path(__file__).resolve().parents[4] / "examples/train/configs/minimax_h3_worldmodel_stage0_prope_recovery.yaml"
)
_HYBRID_PROBE_CONFIG = (
    Path(__file__).resolve().parents[4] / "examples/train/configs/minimax_h3_worldmodel_stage0_hybrid_probe.yaml"
)
_CAUSAL_ACQUISITION_CONFIG = (
    Path(__file__).resolve().parents[4] / "examples/train/configs/minimax_h3_worldmodel_stage0_causal_acquisition.yaml"
)
_CONTROL_REPAIR_V2_CONFIG = (
    Path(__file__).resolve().parents[4] / "examples/train/configs/minimax_h3_worldmodel_stage0_control_repair_v2.yaml"
)
_DIRECTIONAL_V3_CONFIG = (
    Path(__file__).resolve().parents[4] / "examples/train/configs/minimax_h3_worldmodel_stage0_directional_v3.yaml"
)
_DENOISING_V4_CONFIG = (
    Path(__file__).resolve().parents[4] / "examples/train/configs/minimax_h3_worldmodel_stage0_denoising_v4.yaml"
)
_FULL_PROPE_LORA_V5_CONFIG = (
    Path(__file__).resolve().parents[4] / "examples/train/configs/minimax_h3_worldmodel_stage0_full_prope_lora_v5.yaml"
)
_PARALLEL_PROPE_V6_CONFIG = (
    Path(__file__).resolve().parents[4] / "examples/train/configs/minimax_h3_worldmodel_stage0_parallel_prope_v6.yaml"
)
_PAIRED_STATIC_V7_CONFIG = (
    Path(__file__).resolve().parents[4] / "examples/train/configs/minimax_h3_worldmodel_stage0_paired_static_v7.yaml"
)
_ACTION_OVERFIT_V4_CONFIGS = tuple(
    Path(__file__).resolve().parents[4] / "examples/train/configs" / name
    for name in (
        "minimax_h3_worldmodel_action_overfit_v4_lora_gpu2_7.yaml",
        "minimax_h3_worldmodel_action_overfit_v4_boundary_gpu2_7.yaml",
        "minimax_h3_worldmodel_action_overfit_v4_deep_gpu2_7.yaml",
    )
)
_ACTION_OVERFIT_V5_CONFIG = (
    Path(__file__).resolve().parents[4]
    / "examples/train/configs/minimax_h3_worldmodel_action_overfit_v5_paired_static_multisigma_gpu4_7.yaml"
)
_STAGE0_V5_CACHE20668_CONFIG = (
    Path(__file__).resolve().parents[4]
    / "examples/train/configs/minimax_h3_worldmodel_stage0_v5_h3_cache20668_gpu4_7_bs64.yaml"
)
_JS_EASY_POSE_OVERFIT_CONFIG = (
    Path(__file__).resolve().parents[4]
    / "examples/train/configs/minimax_h3_worldmodel_js_easy_pose_overfit.yaml"
)
_DOWNLOAD_SCRIPT = Path(__file__).resolve().parents[4] / "examples/train/download_minimax_h3_worldmodel.sh"
_CHECKSUM_FILE = Path(__file__).resolve().parents[4] / "examples/train/checksums/minimax_h3_worldmodel.sha256"


class _IdentityVideoTransformer:
    patch_size = (1, 2, 2)

    def __call__(self, **kwargs):
        assert kwargs["audio_hidden_states"].shape == (1, 0, 32)
        assert kwargs["action_hidden_states"].shape == (1, kwargs["hidden_states"].shape[1], 6)
        return kwargs["hidden_states"], kwargs["audio_hidden_states"]


class _IdentityPropeTransformer:
    patch_size = (1, 2, 2)

    def __call__(self, **kwargs):
        assert kwargs["action_hidden_states"] is None
        sequence_length = kwargs["position_ids"].shape[0]
        assert kwargs["prope_projection"].shape == (1, sequence_length, 4, 4)
        assert kwargs["prope_inverse"].shape == (1, sequence_length, 4, 4)
        return kwargs["hidden_states"], kwargs["audio_hidden_states"]


class _IdentityHybridTransformer(_IdentityPropeTransformer):
    def __call__(self, **kwargs):
        sequence_length = kwargs["position_ids"].shape[0]
        assert kwargs["action_hidden_states"].shape == (1, kwargs["hidden_states"].shape[1], 6)
        assert kwargs["prope_projection"].shape == (1, sequence_length, 4, 4)
        assert kwargs["prope_inverse"].shape == (1, sequence_length, 4, 4)
        return kwargs["hidden_states"], kwargs["audio_hidden_states"]


class _TrainableAuditTransformer(torch.nn.Module):
    def __init__(self, *, unexpected: bool = False):
        super().__init__()
        self.action_adapter = torch.nn.Linear(1, 1)
        self.proj_in = torch.nn.Linear(1, 1)
        self.norm_out = torch.nn.LayerNorm(1)
        self.proj_out = torch.nn.Linear(1, 1)
        self.backbone = torch.nn.Module()
        self.backbone.register_parameter("lora_A", torch.nn.Parameter(torch.ones(1)))
        self.backbone.register_parameter("lora_B", torch.nn.Parameter(torch.ones(1)))
        self.frozen = torch.nn.Linear(1, 1)
        self.frozen.requires_grad_(unexpected)


def test_h3_worldmodel_fixture_selects_action_lora_contract():
    config = load_run_config(str(_FIXTURE))

    assert config.models["student"]["_target_"].endswith("MiniMaxH3WorldModel")
    assert config.models["student"]["lora"]["enable"] is True
    assert config.training.data.preprocessed_data_type == "h3_worldmodel"
    assert config.training.pipeline_config.dit_config.arch_config.action_dim == 6


def test_h3_worldmodel_stage0_config_uses_camera_loss_validation():
    config = load_run_config(str(_STAGE0_CONFIG))

    student = config.models["student"]
    assert student["trainable_modules"] == ["action_adapter", "proj_in", "norm_out", "proj_out"]
    distributed = config.training.distributed
    assert distributed.num_gpus == 8
    assert distributed.sp_size == 2
    assert distributed.hsdp_replicate_dim == 4
    assert distributed.hsdp_shard_dim == 2
    data_parallel_degree = distributed.num_gpus // distributed.sp_size
    effective_total_batch = (
        config.training.data.train_batch_size * data_parallel_degree * config.training.loop.gradient_accumulation_steps
    )
    assert effective_total_batch == 64
    assert config.training.data.num_height == 480
    assert config.training.data.num_width == 832
    assert config.callbacks["worldmodel_validation"]["data_path"].endswith("/validation")


def test_h3_worldmodel_stage0_v2_strengthens_action_signal_and_keeps_total_batch_64():
    config = load_run_config(str(_STAGE0_V2_CONFIG))
    student = config.models["student"]

    assert student["action_channel_scales"] == [0.03, 0.04, 0.02, 0.01, 0.05, 0.005]
    assert student["action_embedding_scale"] == 16.0
    assert student["action_learning_rate"] > config.training.optimizer.learning_rate
    assert config.callbacks["worldmodel_validation"]["num_batches"] == 8
    data_parallel_degree = config.training.distributed.num_gpus // config.training.distributed.sp_size
    total_batch = (
        config.training.data.train_batch_size * data_parallel_degree * config.training.loop.gradient_accumulation_steps
    )
    assert total_batch == 64


def test_h3_worldmodel_stage0_prope_uses_pose_schema_lora16_and_total_batch_64():
    config = load_run_config(str(_STAGE0_PROPE_CONFIG))
    student = config.models["student"]
    arch = config.training.pipeline_config.dit_config.arch_config

    assert student["camera_conditioning"] == "prope"
    assert student["lora"]["rank"] == student["lora"]["alpha"] == 16
    assert student["prope_use_camera_intrinsics"] is False
    assert "action_adapter" not in student["trainable_modules"]
    assert config.training.data.preprocessed_data_type == "h3_worldmodel_prope"
    assert arch.action_dim == 0 and arch.prope_dim_range == (96, 108)
    data_parallel_degree = config.training.distributed.num_gpus // config.training.distributed.sp_size
    total_batch = (
        config.training.data.train_batch_size * data_parallel_degree * config.training.loop.gradient_accumulation_steps
    )
    assert total_batch == 64


def test_h3_worldmodel_recovery_config_freezes_boundaries_and_resets_training_state():
    config = load_run_config(str(_RECOVERY_CONFIG))
    student = config.models["student"]
    method = config.method

    assert student["trainable_modules"] == []
    assert student["checkpoint_init_modules"] == ["proj_in", "norm_out", "proj_out"]
    assert student["init_from_training_checkpoint"].endswith("checkpoint-250")
    assert student["prope_strength"] == 4.0
    assert student["prope_base_strength"] == 1.0
    assert student["motion_balanced_sampling"] is True
    assert method["temporal_delta_weight"] == 0.25
    assert method["camera_ranking_weight"] == 1.0
    assert method["camera_ranking_every_microbatches"] == 1
    assert config.training.checkpoint.resume_from_checkpoint == ""
    assert config.training.checkpoint.training_state_checkpointing_steps == 10


def test_h3_worldmodel_hybrid_probe_config_starts_fresh_and_keeps_total_batch_64():
    config = load_run_config(str(_HYBRID_PROBE_CONFIG))
    student = config.models["student"]
    method = config.method
    arch = config.training.pipeline_config.dit_config.arch_config

    assert student["camera_conditioning"] == "hybrid"
    assert student["action_representation"] == "relative_pose"
    assert "init_from_training_checkpoint" not in student
    assert student["trainable_modules"] == ["action_adapter"]
    assert student["minimum_video_noise_sigma"] == 0.85
    assert method["latent_motion_weight"] > 0.0
    assert method["camera_ranking_weight"] == 0.0
    assert arch.action_dim == 6 and arch.prope_dim_range == (96, 108)
    data_parallel_degree = config.training.distributed.num_gpus // config.training.distributed.sp_size
    total_batch = (
        config.training.data.train_batch_size * data_parallel_degree * config.training.loop.gradient_accumulation_steps
    )
    assert total_batch == 64


def test_h3_worldmodel_causal_acquisition_freezes_lora_and_removes_future_leakage():
    config = load_run_config(str(_CAUSAL_ACQUISITION_CONFIG))
    student = config.models["student"]
    arch = config.training.pipeline_config.dit_config.arch_config

    assert student["train_lora"] is False
    assert student["trainable_modules"] == ["action_adapter"]
    assert student["fixed_video_noise_sigma"] == 1.0
    assert student["action_fourier_frequencies"] == 4
    assert arch.action_dim == 55
    assert arch.action_injection_interval == 4
    assert arch.action_residual_scale == 1.0
    data_parallel_degree = config.training.distributed.num_gpus // config.training.distributed.sp_size
    diagnostic_batch = (
        config.training.data.train_batch_size * data_parallel_degree * config.training.loop.gradient_accumulation_steps
    )
    assert diagnostic_batch == 4


def test_h3_worldmodel_control_repair_bounds_action_and_targets_final_lora_blocks():
    config = load_run_config(str(_CONTROL_REPAIR_V2_CONFIG))
    student = config.models["student"]
    arch = config.training.pipeline_config.dit_config.arch_config

    assert student["train_lora"] is True
    assert student["action_embedding_scale"] == 1.0
    assert student["action_fourier_frequencies"] == 2
    assert "fixed_video_noise_sigma" not in student
    assert student["minimum_video_noise_sigma"] == 0.0
    assert student["lora"]["target_modules"] == [f"transformer_blocks.{index}.attn" for index in range(42, 50)]
    assert arch.action_dim == 31
    assert arch.action_injection_interval == 8
    assert arch.action_residual_scale == 0.25
    assert arch.action_residual_clip == 0.1
    assert config.method["temporal_delta_weight"] == 0.1


def test_h3_worldmodel_directional_repair_removes_leakage_and_uses_reverse_counterfactual():
    config = load_run_config(str(_DIRECTIONAL_V3_CONFIG))
    student = config.models["student"]
    method = config.method

    assert student["fixed_video_noise_sigma"] == 1.0
    assert method["camera_counterfactual_variants"] == ["static", "reverse"]
    assert method["latent_motion_weight"] > 0.0
    assert config.training.data.data_path.endswith("/camera_moving_v3_32")
    assert config.training.loop.gradient_accumulation_steps == 1


def test_h3_worldmodel_denoising_repair_continues_v3_across_rollout_noise_levels():
    config = load_run_config(str(_DENOISING_V4_CONFIG))
    student = config.models["student"]
    arch = config.training.pipeline_config.dit_config.arch_config

    assert student["init_from_training_checkpoint"].endswith("directional_v3/checkpoints/checkpoint-100")
    assert "fixed_video_noise_sigma" not in student
    assert student["minimum_video_noise_sigma"] == 0.5
    assert student["action_embedding_scale"] == 4.0
    assert student["lora"]["rank"] == 8
    assert config.method["camera_counterfactual_variants"] == ["static", "reverse"]
    assert config.method["latent_motion_weight"] == 2.0
    assert config.training.data.data_path.endswith("/camera_moving_v3_32")
    assert config.training.loop.gradient_accumulation_steps == 1
    assert arch.action_dim == 31 and arch.prope_dim_range == (96, 108)


def test_h3_worldmodel_full_prope_lora_adapts_every_attention_block():
    config = load_run_config(str(_FULL_PROPE_LORA_V5_CONFIG))
    student = config.models["student"]
    arch = config.training.pipeline_config.dit_config.arch_config

    assert "init_from_training_checkpoint" not in student
    assert student["camera_conditioning"] == "hybrid"
    assert student["lora"]["rank"] == student["lora"]["alpha"] == 16
    assert student["lora"]["target_modules"] == ["to_q", "to_k", "to_v", "to_out"]
    assert student["minimum_video_noise_sigma"] == 0.5
    assert config.method["camera_counterfactual_variants"] == ["static", "reverse"]
    assert config.training.data.data_path.endswith("/camera_moving_v3_32")
    assert config.training.loop.gradient_accumulation_steps == 1
    assert arch.action_dim == 31 and arch.prope_dim_range == (96, 108)


def test_h3_worldmodel_parallel_prope_preserves_base_with_sparse_camera_branches():
    config = load_run_config(str(_PARALLEL_PROPE_V6_CONFIG))
    student = config.models["student"]
    arch = config.training.pipeline_config.dit_config.arch_config

    assert "init_from_training_checkpoint" not in student
    assert student["prope_use_camera_intrinsics"] is False
    assert student["prope_base_strength"] == 0.25
    assert student["prope_learning_rate"] > config.training.optimizer.learning_rate
    assert arch.prope_parallel_residual_interval == 6
    assert arch.prope_dim_range == (96, 108)
    assert config.method["camera_counterfactual_variants"] == ["static", "reverse"]
    assert config.training.loop.max_train_steps == 100


def test_h3_worldmodel_paired_static_uses_pure_noise_and_direct_counterfactual_target():
    config = load_run_config(str(_PAIRED_STATIC_V7_CONFIG))
    student = config.models["student"]

    assert student["init_from_training_checkpoint"].endswith("parallel_prope_v6/checkpoints/checkpoint-100")
    assert student["fixed_video_noise_sigma"] == 1.0
    assert config.method["camera_counterfactual_variants"] == ["static"]
    assert config.method["camera_static_reconstruction_weight"] == 1.0
    assert config.training.loop.max_train_steps == 100


def test_h3_action_overfit_configs_isolate_trainable_capacity():
    configs = [load_run_config(str(path)) for path in _ACTION_OVERFIT_V4_CONFIGS]
    expected_modules = [
        ["action_adapter"],
        ["action_adapter", "proj_in", "norm_out", "proj_out"],
        [
            "action_adapter",
            "proj_in",
            "norm_out",
            "proj_out",
            "transformer_blocks.48",
            "transformer_blocks.49",
        ],
    ]
    assert [config.models["student"]["trainable_modules"] for config in configs] == expected_modules
    for config in configs:
        student = config.models["student"]
        method = config.method
        arch = config.training.pipeline_config.dit_config.arch_config
        assert student["fixed_video_noise_sigma"] == 1.0
        assert student["neutralize_text_conditioning"] is True
        assert "init_from_training_checkpoint" not in student
        assert method["camera_counterfactual_same_input"] is True
        assert method["camera_counterfactual_variants"] == ["static", "reverse"]
        assert method["camera_sensitivity_weight"] > 0.0
        assert method["latent_motion_direction_weight"] > 0.0
        assert arch.prope_mode == "exact"
        assert config.training.distributed.num_gpus == 6
        assert config.training.distributed.sp_size == 2
        assert config.training.distributed.hsdp_replicate_dim == 3
        assert config.training.loop.gradient_accumulation_steps == 1


def test_h3_action_overfit_v5_resumes_boundary_model_with_paired_static_multisigma():
    config = load_run_config(str(_ACTION_OVERFIT_V5_CONFIG))
    student = config.models["student"]
    method = config.method
    arch = config.training.pipeline_config.dit_config.arch_config

    assert student["trainable_modules"] == ["action_adapter", "proj_in", "norm_out", "proj_out"]
    assert student["lora"]["target_modules"] == ["to_q", "to_k", "to_v", "to_out"]
    assert "fixed_video_noise_sigma" not in student
    assert student["minimum_video_noise_sigma"] == 0.05
    assert student["neutralize_text_conditioning"] is True
    assert method["camera_counterfactual_variants"] == ["static", "reverse", "shuffled"]
    assert method["camera_counterfactual_same_input"] is True
    assert method["camera_static_paired_target"] is True
    assert method["camera_static_reconstruction_weight"] == 3.0
    assert method["static_latent_motion_weight"] == 3.0
    assert config.training.checkpoint.resume_from_checkpoint.endswith(
        "minimax_h3_action_overfit_v4_boundary_gpu2_7/checkpoints/checkpoint-300"
    )
    assert "action_overfit_v5_paired_static_multisigma" in config.training.checkpoint.output_dir
    data_parallel_degree = config.training.distributed.num_gpus // config.training.distributed.sp_size
    global_batch = (
        config.training.data.train_batch_size
        * data_parallel_degree
        * config.training.loop.gradient_accumulation_steps
    )
    assert global_batch == 4
    assert config.training.loop.max_train_steps == 800
    assert arch.prope_mode == "exact"
    assert arch.action_injection_interval == 1


def test_h3_stage0_v5_cache20668_starts_fresh_on_four_gpus_with_global_batch_64():
    config = load_run_config(str(_STAGE0_V5_CACHE20668_CONFIG))
    student = config.models["student"]
    method = config.method
    distributed = config.training.distributed
    arch = config.training.pipeline_config.dit_config.arch_config

    assert student["init_from"].endswith("/MiniMax-H3")
    assert "init_from_training_checkpoint" not in student
    assert student["trainable_modules"] == ["action_adapter", "proj_in", "norm_out", "proj_out"]
    assert student["lora"]["target_modules"] == ["to_q", "to_k", "to_v", "to_out"]
    assert student["motion_balanced_sampling"] is True
    assert method["camera_counterfactual_variants"] == ["static", "reverse", "shuffled"]
    assert method["camera_static_paired_target"] is True
    assert method["camera_response_action_epsilon"] == 1.0e-6
    assert method["fail_on_nonfinite_loss"] is True
    assert config.callbacks["grad_clip"]["error_if_nonfinite"] is True
    assert config.callbacks["grad_clip"]["log_grad_norms"] is True
    assert config.training.data.data_path.endswith("/h3_legacy_20668_v5/preprocessed/all")
    assert config.training.data.preprocessed_data_type == "h3_worldmodel_prope"
    assert distributed.num_gpus == 4
    assert distributed.sp_size == 2
    assert distributed.hsdp_replicate_dim == 2
    assert distributed.hsdp_shard_dim == 2
    data_parallel_degree = distributed.num_gpus // distributed.sp_size
    global_batch = (
        config.training.data.train_batch_size
        * data_parallel_degree
        * config.training.loop.gradient_accumulation_steps
    )
    assert global_batch == 64
    assert config.training.checkpoint.resume_from_checkpoint == ""
    assert arch.prope_mode == "exact"
    assert arch.action_injection_interval == 1


def test_h3_worldmodel_download_is_pinned_verified_and_audio_free():
    script = _DOWNLOAD_SCRIPT.read_text()
    checksum_rows = _CHECKSUM_FILE.read_text().splitlines()

    assert "42ed227ee7df40d41602854ae760620d6eb651fe" in script
    assert "https://hf-mirror.com" in script
    assert "audio_vae" not in script and "audio_scheduler" not in script
    assert all(pattern in script for pattern in ("'FL2VA/*'", "'Ref2VA/*'", "'transformer_ref/*'", "'audio_*'"))
    assert "--http1.1" in script and "--retry-all-errors" in script
    assert "sha256sum --check" in script
    assert len(checksum_rows) == 31
    assert all(len(row.split(maxsplit=1)[0]) == 64 for row in checksum_rows)


def test_h3_worldmodel_trainable_audit_rejects_unlisted_backbone_weights():
    model = MiniMaxH3WorldModel.__new__(MiniMaxH3WorldModel)
    model.trainable_modules = ("action_adapter", "proj_in", "norm_out", "proj_out")
    model.transformer = _TrainableAuditTransformer(unexpected=False)
    model._audit_trainable_parameters()

    model.transformer = _TrainableAuditTransformer(unexpected=True)
    with pytest.raises(RuntimeError, match="Unexpected trainable H3 backbone parameters"):
        model._audit_trainable_parameters()


def test_h3_worldmodel_optimizer_separates_action_adapter_learning_rate():
    model = MiniMaxH3WorldModel.__new__(MiniMaxH3WorldModel)
    model.transformer = _TrainableAuditTransformer()
    model.action_learning_rate = 1.0e-4

    groups = model.optimizer_param_groups(learning_rate=2.0e-5, weight_decay=0.01)

    assert [group["name"] for group in groups] == ["pretrained_and_lora", "action_adapter"]
    assert [group["lr"] for group in groups] == [2.0e-5, 1.0e-4]
    assert [group["weight_decay"] for group in groups] == [0.01, 0.0]
    assert list(groups[1]["params"]) == list(model.transformer.action_adapter.parameters())


def test_h3_worldmodel_prepares_video_only_first_frame_and_actions(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(MiniMaxH3WorldModel, "device", property(lambda _self: torch.device("cpu")))
    model = MiniMaxH3WorldModel.__new__(MiniMaxH3WorldModel)
    model.transformer = _IdentityVideoTransformer()
    model.noise_scheduler = MiniMaxH3Scheduler(shift=12.0)
    model.training_config = SimpleNamespace(distributed=SimpleNamespace(sp_size=1))
    model.sp_group = None
    model.action_channel_scales = (0.03, 0.04, 0.02, 0.01, 0.05, 0.005)
    model.action_normalized_clip = 5.0
    model.action_embedding_scale = 16.0
    model.action_motion_loss_weight = 2.0
    raw = {
        "vae_latent": torch.zeros((1, 24, 2, 4, 4)),
        "first_frame_latent": torch.zeros((1, 24, 1, 4, 4)),
        "text_embedding": torch.zeros((1, 4, 5120)),
        "text_attention_mask": torch.tensor([[1, 1, 0, 0]], dtype=torch.float32),
        "text_token_tags": torch.tensor([[0, 1]], dtype=torch.float32),
        "action_continuous": torch.ones((1, 2, 6)),
        "action_valid_mask": torch.tensor([[1, 0]], dtype=torch.float32),
    }

    batch = model.prepare_batch(raw, generator=torch.Generator(device="cpu").manual_seed(9))
    prediction = model.predict_noise(
        batch.noisy_model_input.permute(0, 2, 1, 3, 4),
        batch.timesteps,
        batch,
        conditional=True,
    )

    assert isinstance(prediction, torch.Tensor)
    assert prediction.shape == batch.latents.shape == (1, 2, 24, 4, 4)
    assert batch.audio_latents is None and batch.audio_noise is None
    assert batch.minimax_h3_layout.num_audio_latents == 0
    assert batch.minimax_h3_layout.num_condition_video_rows == 4


def test_action_only_keeps_first_frame_vision_and_removes_text(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(MiniMaxH3WorldModel, "device", property(lambda _self: torch.device("cpu")))
    model = MiniMaxH3WorldModel.__new__(MiniMaxH3WorldModel)
    model.transformer = _IdentityVideoTransformer()
    model.noise_scheduler = MiniMaxH3Scheduler(shift=12.0)
    model.training_config = SimpleNamespace(distributed=SimpleNamespace(sp_size=1))
    model.sp_group = None
    model.vision_only_conditioning = True
    model.neutralize_text_conditioning = False
    model.action_channel_scales = (1.0,) * 6
    model.action_normalized_clip = 4.0
    model.action_embedding_scale = 4.0
    model.action_motion_loss_weight = 0.0
    raw = {
        "vae_latent": torch.zeros((1, 24, 2, 4, 4)),
        "first_frame_latent": torch.zeros((1, 24, 1, 4, 4)),
        "text_embedding": torch.stack((torch.ones(5120), torch.full((5120,), 2.0)))[None],
        "text_attention_mask": torch.ones((1, 2)),
        "text_token_tags": torch.tensor([[0, 1]], dtype=torch.float32),
        "action_continuous": torch.zeros((1, 2, 6)),
        "action_valid_mask": torch.ones((1, 2)),
    }

    batch = model.prepare_batch(raw, generator=torch.Generator(device="cpu").manual_seed(9))

    assert batch.encoder_hidden_states.shape == (1, 1, 5120)
    assert torch.equal(batch.encoder_hidden_states.float(), torch.ones((1, 1, 5120)))
    assert batch.encoder_attention_mask.shape == (1, 1)


def test_action_response_magnitude_keeps_audited_translation_scale():
    camera = torch.eye(4).repeat(1, 3, 1, 1)
    camera[:, 1, 0, 3] = 1.0
    camera[:, 2, 0, 3] = 2.0
    static = camera[:, :1].expand_as(camera).clone()

    _, _, normalized = action_response_bounds(camera, static, 0.1, 0.5)
    _, _, calibrated = action_response_bounds(
        camera, static, 0.1, 0.5, normalize_translation=False
    )

    assert torch.isclose(calibrated, normalized * 2.0, atol=1.0e-5)


def test_h3_worldmodel_prope_prepares_and_injects_calibrated_pose(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(MiniMaxH3WorldModel, "device", property(lambda _self: torch.device("cpu")))
    model = MiniMaxH3WorldModel.__new__(MiniMaxH3WorldModel)
    model.camera_conditioning = "prope"
    model.transformer = _IdentityPropeTransformer()
    model.noise_scheduler = MiniMaxH3Scheduler(shift=12.0)
    model.training_config = SimpleNamespace(
        distributed=SimpleNamespace(sp_size=1),
        data=SimpleNamespace(num_width=64, num_height=64),
    )
    model.sp_group = None
    model.action_channel_scales = (1.0,) * 6
    model.action_normalized_clip = 5.0
    model.action_embedding_scale = 1.0
    model.action_motion_loss_weight = 0.0
    model.prope_use_camera_intrinsics = False
    c2w = torch.eye(4).repeat(1, 2, 1, 1)
    c2w[:, 1, 0, 3] = 1.0
    intrinsics = torch.eye(3).repeat(1, 2, 1, 1)
    intrinsics[..., 0, 0] = 0.8
    intrinsics[..., 1, 1] = 0.8
    intrinsics[..., 0, 2] = 0.5
    intrinsics[..., 1, 2] = 0.5
    raw = {
        "vae_latent": torch.zeros((1, 24, 2, 4, 4)),
        "first_frame_latent": torch.zeros((1, 24, 1, 4, 4)),
        "text_embedding": torch.zeros((1, 4, 5120)),
        "text_attention_mask": torch.tensor([[1, 1, 0, 0]], dtype=torch.float32),
        "text_token_tags": torch.tensor([[0, 1]], dtype=torch.float32),
        "action_continuous": torch.zeros((1, 2, 6)),
        "action_valid_mask": torch.ones((1, 2)),
        "camera_to_world": c2w,
        "camera_intrinsics": intrinsics,
    }

    batch = model.prepare_batch(raw, generator=torch.Generator(device="cpu").manual_seed(11))
    prediction = model.predict_noise(
        batch.noisy_model_input.permute(0, 2, 1, 3, 4),
        batch.timesteps,
        batch,
        conditional=True,
    )

    assert isinstance(prediction, torch.Tensor)
    assert prediction.shape == (1, 2, 24, 4, 4)
    assert torch.equal(batch.input_kwargs["camera_to_world"], c2w)

    model.require_rebased_camera = True
    shifted = c2w.clone()
    shifted[..., 0, 3] += 0.25
    with pytest.raises(ValueError, match="rebased to the first latent frame"):
        model.prepare_batch({**raw, "camera_to_world": shifted}, generator=torch.Generator().manual_seed(11))
    reflected = c2w.clone()
    reflected[:, 1, 0, 0] = -1.0
    with pytest.raises(ValueError, match="proper, not reflected"):
        model.prepare_batch({**raw, "camera_to_world": reflected}, generator=torch.Generator().manual_seed(11))
    model.prepare_batch(raw, generator=torch.Generator().manual_seed(11))


def test_js_easy_pose_overfit_config_uses_exact_prope_and_flow_only():
    config = load_run_config(str(_JS_EASY_POSE_OVERFIT_CONFIG))
    student = config.models["student"]
    arch = config.training.pipeline_config.dit_config.arch_config

    assert student["camera_conditioning"] == "prope"
    assert student["require_rebased_camera"] is True
    assert student["normalize_camera_translation"] is False
    assert student["prope_use_camera_intrinsics"] is True
    assert student["vision_only_conditioning"] is True
    assert student["action_motion_loss_weight"] == 0.0
    assert student["sampling_index_path"].endswith("stage1_jl_indices.json")
    assert config.method["_target_"].endswith("finetune.FineTuneMethod")
    assert arch.prope_mode == "exact" and arch.prope_dim_range == (96, 108)
    assert arch.action_dim == 0
    assert config.training.loop.gradient_accumulation_steps == 2


def test_h3_worldmodel_hybrid_injects_relative_pose_and_prope(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(MiniMaxH3WorldModel, "device", property(lambda _self: torch.device("cpu")))
    model = MiniMaxH3WorldModel.__new__(MiniMaxH3WorldModel)
    model.camera_conditioning = "hybrid"
    model.action_representation = "relative_pose"
    model.transformer = _IdentityHybridTransformer()
    model.noise_scheduler = MiniMaxH3Scheduler(shift=12.0)
    model.training_config = SimpleNamespace(
        distributed=SimpleNamespace(sp_size=1),
        data=SimpleNamespace(num_width=64, num_height=64),
    )
    model.sp_group = None
    model.action_channel_scales = (1.0,) * 6
    model.action_normalized_clip = 1.0
    model.action_embedding_scale = 4.0
    model.action_motion_loss_weight = 0.0
    model.prope_use_camera_intrinsics = False
    c2w = torch.eye(4).repeat(1, 2, 1, 1)
    c2w[:, 1, 0, 3] = 1.0
    intrinsics = torch.eye(3).repeat(1, 2, 1, 1)
    intrinsics[..., 0, 0] = intrinsics[..., 1, 1] = 0.5
    intrinsics[..., 0, 2] = intrinsics[..., 1, 2] = 0.5
    raw = {
        "vae_latent": torch.zeros((1, 24, 2, 4, 4)),
        "first_frame_latent": torch.zeros((1, 24, 1, 4, 4)),
        "text_embedding": torch.zeros((1, 4, 5120)),
        "text_attention_mask": torch.tensor([[1, 1, 0, 0]], dtype=torch.float32),
        "text_token_tags": torch.tensor([[0, 1]], dtype=torch.float32),
        "action_continuous": torch.zeros((1, 2, 6)),
        "action_valid_mask": torch.ones((1, 2)),
        "camera_to_world": c2w,
        "camera_intrinsics": intrinsics,
    }

    batch = model.prepare_batch(raw, generator=torch.Generator(device="cpu").manual_seed(13))
    prediction = model.predict_noise(
        batch.noisy_model_input.permute(0, 2, 1, 3, 4),
        batch.timesteps,
        batch,
        conditional=True,
    )

    assert isinstance(prediction, torch.Tensor)
    assert prediction.shape == (1, 2, 24, 4, 4)


def test_h3_worldmodel_action_normalization_preserves_zero_and_clamps_outliers():
    model = MiniMaxH3WorldModel.__new__(MiniMaxH3WorldModel)
    model.action_channel_scales = (0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
    model.action_normalized_clip = 2.0
    model.action_embedding_scale = 4.0

    normalized = model._normalized_actions(torch.tensor([[[0.0, 0.5, -0.5, 1.5, -1.5, 0.25]]]))

    torch.testing.assert_close(normalized, torch.tensor([[[0.0, 4.0, -4.0, 8.0, -8.0, 2.0]]]))


def test_h3_worldmodel_asinh_action_normalization_preserves_large_magnitudes():
    model = MiniMaxH3WorldModel.__new__(MiniMaxH3WorldModel)
    model.action_channel_scales = (1.0,) * 6
    model.action_normalized_clip = 2.0
    model.action_normalization_mode = "asinh"
    model.action_embedding_scale = 1.0

    values = torch.tensor([[[0.0, 1.0, 2.0, 10.0, -10.0, -2.0]]])
    normalized = model._normalized_actions(values)

    torch.testing.assert_close(normalized, torch.asinh(values))
    assert normalized[0, 0, 3] > normalized[0, 0, 2]


def test_h3_multiaction_postadaln_config_has_live_action_supervision():
    path = Path(__file__).resolve().parents[4] / "examples/train/configs/minimax_h3_multiaction_postadaln_20261002.yaml"
    config = load_run_config(str(path))
    student = config.models["student"]
    arch = config.training.pipeline_config.dit_config.arch_config
    assert arch.action_post_adaln is True
    assert student["action_normalization_mode"] == "asinh"
    assert student["paired_supervision"] is False
    assert config.method["camera_ranking_weight"] > 0
    assert config.method["camera_counterfactual_same_input"] is True
    assert config.method["camera_sensitivity_weight"] > 0
    assert len([name for name in student["trainable_modules"] if name.endswith(".attn.to_out")]) == 4
    assert (config.training.distributed.num_gpus // config.training.distributed.sp_size *
            config.training.data.train_batch_size * config.training.loop.gradient_accumulation_steps) == 8


def test_h3_trainable_block_name_resolves_gradient_checkpoint_wrapper():
    name = "transformer_blocks.46.attn.to_out"
    wrapped = "transformer_blocks.46._checkpoint_wrapped_module.attn.to_out"
    actual = MiniMaxH3WorldModel._resolve_trainable_module_name(name, {wrapped: torch.nn.Identity()})
    assert actual == wrapped


def test_h3_worldmodel_fourier_features_keep_static_control_explicit():
    model = MiniMaxH3WorldModel.__new__(MiniMaxH3WorldModel)
    model.action_channel_scales = (1.0,) * 6
    model.action_normalized_clip = 1.0
    model.action_embedding_scale = 2.0
    model.action_fourier_frequencies = 2
    actions = torch.zeros((1, 2, 6))
    mask = torch.tensor([[True, False]])

    features = model._action_features(actions, mask)

    assert features.shape == (1, 2, 31)
    assert features[0, 0, -1] == 2.0
    assert int(torch.count_nonzero(features[0, 0])) == 1
    torch.testing.assert_close(features[0, 1], torch.zeros(31))


def test_h3_worldmodel_optimizer_allows_frozen_lora_acquisition_stage():
    model = MiniMaxH3WorldModel.__new__(MiniMaxH3WorldModel)
    model.transformer = _TrainableAuditTransformer()
    model.action_learning_rate = 1.0e-4
    model.transformer.proj_in.requires_grad_(False)
    model.transformer.norm_out.requires_grad_(False)
    model.transformer.proj_out.requires_grad_(False)
    model.transformer.backbone.requires_grad_(False)

    groups = model.optimizer_param_groups(learning_rate=2.0e-5, weight_decay=0.01)

    assert [group["name"] for group in groups] == ["action_adapter"]


def test_h3_worldmodel_motion_loss_weights_are_mean_one_and_static_safe():
    model = MiniMaxH3WorldModel.__new__(MiniMaxH3WorldModel)
    model.action_channel_scales = (1.0,) * 6
    model.action_normalized_clip = 5.0
    model.action_embedding_scale = 1.0
    model.action_motion_loss_weight = 2.0
    mask = torch.ones((1, 3), dtype=torch.bool)

    static_weights = model._video_loss_weights(torch.zeros((1, 3, 6)), mask)
    moving_actions = torch.zeros((1, 3, 6))
    moving_actions[0, 2, 0] = 1.0
    moving_weights = model._video_loss_weights(moving_actions, mask)

    torch.testing.assert_close(static_weights, torch.ones_like(static_weights))
    torch.testing.assert_close(moving_weights.mean(), torch.tensor(1.0))
    assert moving_weights[0, 2] > moving_weights[0, 0]
