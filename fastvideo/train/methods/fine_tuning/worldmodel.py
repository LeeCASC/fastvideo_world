# SPDX-License-Identifier: Apache-2.0
"""Camera-sensitive objectives for MiniMax-H3 world-model fine-tuning."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

from fastvideo.models.dits.minimax_h3_prope import (
    make_camera_trajectory_variant,
    relative_camera_actions,
)
from fastvideo.train.methods.fine_tuning.finetune import FineTuneMethod, video_mse_loss
from fastvideo.train.models.base import NoisePrediction
from fastvideo.train.models.minimax_h3.camera_motion_probe import (
    latent_soft_translation,
    masked_motion_loss,
)


def predicted_clean_from_noisy(
    prediction: torch.Tensor,
    noisy_model_input: torch.Tensor,
    sigmas: torch.Tensor,
) -> torch.Tensor:
    """Convert H3 flow velocity to a clean estimate for a supplied noisy input."""
    noisy = noisy_model_input.permute(0, 2, 1, 3, 4).float()
    return noisy - prediction.float() * sigmas.float()


def predicted_clean_video(prediction: torch.Tensor, training_batch: Any) -> torch.Tensor:
    """Convert the H3 rectified-flow velocity into a clean-video estimate."""
    if training_batch.noisy_model_input is None or training_batch.sigmas is None:
        raise RuntimeError("world-model auxiliary losses require noisy input and sigmas")
    return predicted_clean_from_noisy(
        prediction,
        training_batch.noisy_model_input,
        training_batch.sigmas,
    )


def temporal_delta_error(
    predicted_clean: torch.Tensor,
    clean: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Measure robust latent-frame velocity error on valid camera transitions."""
    if predicted_clean.shape != clean.shape or predicted_clean.ndim != 5:
        raise ValueError("predicted and clean video latents must share shape [B,T,C,H,W]")
    if valid_mask.shape != predicted_clean.shape[:2]:
        raise ValueError("camera validity mask must have shape [B,T]")
    predicted_delta = predicted_clean[:, 1:] - predicted_clean[:, :-1]
    clean_delta = clean.float()[:, 1:] - clean.float()[:, :-1]
    transition_mask = (valid_mask[:, 1:] & valid_mask[:, :-1]).float()
    per_transition = F.smooth_l1_loss(predicted_delta, clean_delta, reduction="none").mean(dim=(2, 3, 4))
    denominator = transition_mask.sum().clamp_min(1.0)
    return (per_transition * transition_mask).sum() / denominator


def lowpass_video(video: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Suppress spatial texture before measuring global latent motion."""
    if kernel_size <= 1:
        return video.float()
    if kernel_size > min(video.shape[-2:]):
        raise ValueError("latent motion low-pass kernel exceeds the latent spatial size")
    flattened = video.float().flatten(0, 1)
    pooled = F.avg_pool2d(flattened, kernel_size=kernel_size, stride=kernel_size)
    return pooled.unflatten(0, video.shape[:2])


def masked_motion_direction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Penalize angular disagreement only where target motion is observable."""
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("prediction and target must share shape [B,T,D]")
    if valid_mask.shape != prediction.shape[:2]:
        raise ValueError("valid_mask must have shape [B,T]")
    observable = target.float().norm(dim=-1) > 1.0e-4
    weights = (valid_mask & observable).float()
    cosine = F.cosine_similarity(prediction.float(), target.float(), dim=-1, eps=1.0e-6)
    return ((1.0 - cosine) * weights).sum() / weights.sum().clamp_min(1.0)


def _stable_root_mean_square(value: torch.Tensor) -> torch.Tensor:
    """Return an exact-zero RMS with a finite gradient at zero."""
    if value.numel() == 0:
        raise ValueError("RMS requires at least one value")
    return torch.linalg.vector_norm(value.float()) / math.sqrt(value.numel())


def paired_clean_difference_loss(
    first_prediction: torch.Tensor,
    second_prediction: torch.Tensor,
    first_noisy: torch.Tensor,
    second_noisy: torch.Tensor,
    sigma: torch.Tensor,
    first_target: torch.Tensor,
    second_target: torch.Tensor,
) -> torch.Tensor:
    """Supervise the signed difference between two *real* paired videos."""
    first_clean = predicted_clean_from_noisy(first_prediction, first_noisy, sigma)
    second_clean = predicted_clean_from_noisy(second_prediction, second_noisy, sigma)
    target_difference = first_target.float() - second_target.float()
    return F.mse_loss(first_clean - second_clean, target_difference)


def action_response_bounds(
    correct_camera: torch.Tensor,
    counterfactual_camera: torch.Tensor,
    minimum_scale: float,
    maximum_scale: float,
    *,
    normalize_translation: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map an action perturbation to response bounds and its magnitude."""
    action_delta = relative_camera_actions(
        correct_camera, normalize_translation=normalize_translation
    ) - relative_camera_actions(
        counterfactual_camera, normalize_translation=normalize_translation
    )
    magnitude = _stable_root_mean_square(action_delta)
    return magnitude * minimum_scale, magnitude * maximum_scale, magnitude


def paired_static_training_inputs(
    clean_latents: torch.Tensor,
    noise: torch.Tensor,
    sigmas: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a repeated-first-frame target and its matched noisy input."""
    if clean_latents.ndim != 5:
        raise ValueError("clean latents must have shape [B,T,C,H,W]")
    static_latents = clean_latents[:, :1].expand_as(clean_latents)
    static_bcthw = static_latents.permute(0, 2, 1, 3, 4)
    if noise.shape != static_bcthw.shape or sigmas.ndim != 5:
        raise ValueError("noise and sigmas must align with paired static latents")
    static_noisy_model_input = ((1.0 - sigmas.float()) * static_bcthw.float() + sigmas.float() * noise.float()).to(
        noise.dtype)
    return static_latents, static_noisy_model_input


class MiniMaxH3WorldModelFineTuneMethod(FineTuneMethod):
    """Add temporal motion and camera-counterfactual objectives."""

    def __init__(self, *, cfg: Any, role_models: dict[str, Any]) -> None:
        method_config = dict(cfg.method)
        self.temporal_delta_weight = float(method_config.get("temporal_delta_weight", 0.0))
        self.paired_flow_weight = float(method_config.get("paired_flow_weight", 0.0))
        self.paired_difference_weight = float(method_config.get("paired_difference_weight", 0.0))
        self.paired_require_identical_noisy_input = bool(
            method_config.get("paired_require_identical_noisy_input", False))
        self.temporal_delta_warmup_steps = int(method_config.get("temporal_delta_warmup_steps", 0))
        self.camera_ranking_weight = float(method_config.get("camera_ranking_weight", 0.0))
        self.camera_ranking_margin_ratio = float(method_config.get("camera_ranking_margin_ratio", 0.03))
        self.camera_ranking_every_microbatches = int(method_config.get("camera_ranking_every_microbatches", 4))
        self.camera_counterfactual_variants = tuple(
            str(variant) for variant in method_config.get("camera_counterfactual_variants", ("static", )))
        self.latent_motion_weight = float(method_config.get("latent_motion_weight", 0.0))
        self.static_latent_motion_weight = float(method_config.get("static_latent_motion_weight", 1.0))
        self.latent_motion_direction_weight = float(method_config.get("latent_motion_direction_weight", 0.0))
        self.latent_motion_lowpass_kernel = int(method_config.get("latent_motion_lowpass_kernel", 1))
        self.camera_counterfactual_same_input = bool(method_config.get("camera_counterfactual_same_input", False))
        self.camera_static_paired_target = bool(
            method_config.get("camera_static_paired_target", not self.camera_counterfactual_same_input))
        self.camera_sensitivity_weight = float(method_config.get("camera_sensitivity_weight", 0.0))
        self.camera_response_range_weight = float(method_config.get("camera_response_range_weight", 0.0))
        self.camera_sensitivity_min_scale = float(method_config.get("camera_sensitivity_min_scale", 0.0))
        self.camera_sensitivity_max_scale = float(method_config.get("camera_sensitivity_max_scale", 1.0))
        self.camera_response_action_epsilon = float(method_config.get("camera_response_action_epsilon", 1.0e-6))
        self.camera_static_reconstruction_weight = float(method_config.get("camera_static_reconstruction_weight", 0.0))
        self.fail_on_nonfinite_loss = bool(method_config.get("fail_on_nonfinite_loss", True))
        self.latent_motion_max_shift = int(method_config.get("latent_motion_max_shift", 4))
        self.latent_motion_temperature = float(method_config.get("latent_motion_temperature", 0.02))
        self.precondition_outputs = bool(cfg.training.model.precondition_outputs)
        if (min(
                self.temporal_delta_weight,
                self.paired_flow_weight,
                self.paired_difference_weight,
                self.camera_ranking_weight,
                self.latent_motion_weight,
                self.latent_motion_direction_weight,
                self.camera_sensitivity_weight,
                self.camera_response_range_weight,
                self.static_latent_motion_weight,
                self.camera_static_reconstruction_weight,
        ) < 0.0):
            raise ValueError("world-model auxiliary loss weights must be non-negative")
        if self.temporal_delta_warmup_steps < 0:
            raise ValueError("temporal_delta_warmup_steps must be non-negative")
        if self.camera_ranking_margin_ratio < 0.0 or self.camera_ranking_every_microbatches <= 0:
            raise ValueError("camera ranking margin must be non-negative and cadence must be positive")
        if not self.camera_counterfactual_variants or any(variant not in {"static", "reverse", "shuffled"}
                                                          for variant in self.camera_counterfactual_variants):
            raise ValueError("camera_counterfactual_variants must contain only static, reverse, and/or shuffled")
        if self.latent_motion_max_shift <= 0 or self.latent_motion_temperature <= 0.0:
            raise ValueError("latent motion max shift and temperature must be positive")
        if self.latent_motion_lowpass_kernel <= 0:
            raise ValueError("latent_motion_lowpass_kernel must be positive")
        if not 0.0 <= self.camera_sensitivity_min_scale < self.camera_sensitivity_max_scale:
            raise ValueError("camera sensitivity scales must satisfy 0 <= minimum < maximum")
        if self.camera_response_action_epsilon < 0.0:
            raise ValueError("camera_response_action_epsilon must be non-negative")
        if (self.camera_sensitivity_weight > 0.0
                or self.camera_response_range_weight > 0.0) and not self.camera_counterfactual_same_input:
            raise ValueError("camera sensitivity losses require camera_counterfactual_same_input=true")
        if self.camera_static_reconstruction_weight > 0.0 and not self.camera_static_paired_target:
            raise ValueError("static reconstruction requires camera_static_paired_target=true")
        if self.camera_static_reconstruction_weight > 0.0 and "static" not in self.camera_counterfactual_variants:
            raise ValueError("static reconstruction requires a static counterfactual variant")
        self._worldmodel_microbatch = 0
        super().__init__(cfg=cfg, role_models=role_models)
        if getattr(self.student, "camera_conditioning", None) not in {"prope", "hybrid"}:
            raise TypeError("MiniMaxH3WorldModelFineTuneMethod requires PRoPE or hybrid camera conditioning")

    def augment_loss_map(
        self,
        loss_map: dict[str, torch.Tensor],
        prediction: NoisePrediction,
        training_batch: Any,
        iteration: int,
    ) -> dict[str, torch.Tensor]:
        if not isinstance(prediction, torch.Tensor):
            raise TypeError("MiniMax-H3 world-model training requires a video-only prediction")
        if training_batch.latents is None or training_batch.input_kwargs is None:
            raise RuntimeError("world-model auxiliary losses require clean latents and input kwargs")
        valid_mask = training_batch.input_kwargs["action_mask"].to(torch.bool)
        predicted_clean = predicted_clean_video(prediction, training_batch)
        correct_delta_error = temporal_delta_error(predicted_clean, training_batch.latents, valid_mask)
        warmup = min(1.0, float(iteration) / max(1, self.temporal_delta_warmup_steps))
        delta_weight = self.temporal_delta_weight * warmup
        total_loss = loss_map["total_loss"] + correct_delta_error * delta_weight
        loss_map["temporal_delta_loss"] = correct_delta_error
        loss_map["temporal_delta_weight"] = correct_delta_error.new_tensor(delta_weight)

        paired_flow_weight = float(getattr(self, "paired_flow_weight", 0.0))
        paired_difference_weight = float(getattr(self, "paired_difference_weight", 0.0))
        if paired_flow_weight > 0.0 or paired_difference_weight > 0.0:
            partner = training_batch.input_kwargs.get("pair_latents")
            partner_camera = training_batch.input_kwargs.get("pair_camera_to_world")
            partner_intrinsics = training_batch.input_kwargs.get("pair_camera_intrinsics")
            partner_mask = training_batch.input_kwargs.get("pair_action_mask")
            if any(value is None for value in (partner, partner_camera, partner_intrinsics, partner_mask)):
                raise RuntimeError("paired camera loss requires a verified real partner video and camera")
            if training_batch.noise is None or training_batch.sigmas is None or training_batch.noisy_model_input is None:
                raise RuntimeError("paired camera loss requires synchronized noise and sigma")
            sigma = training_batch.sigmas
            if self.paired_require_identical_noisy_input and not bool(torch.allclose(sigma, torch.ones_like(sigma))):
                raise ValueError("identical-input paired supervision requires sigma=1")
            partner_bcthw = partner.permute(0, 2, 1, 3, 4)
            partner_noisy = ((1.0 - sigma.float()) * partner_bcthw.float() +
                             sigma.float() * training_batch.noise.float()).to(training_batch.noise.dtype)
            originals = {key: training_batch.input_kwargs[key]
                         for key in ("camera_to_world", "camera_intrinsics", "action_mask")}
            training_batch.input_kwargs.update({
                "camera_to_world": partner_camera,
                "camera_intrinsics": partner_intrinsics,
                "action_mask": partner_mask,
            })
            try:
                partner_prediction = self.student.predict_noise(
                    partner_noisy.permute(0, 2, 1, 3, 4),
                    training_batch.timesteps,
                    training_batch,
                    conditional=True,
                    attn_kind=self._attn_kind,
                )
            finally:
                training_batch.input_kwargs.update(originals)
            if not isinstance(partner_prediction, torch.Tensor):
                raise TypeError("paired world-model prediction must be video-only")
            if self.precondition_outputs:
                partner_clean = predicted_clean_from_noisy(partner_prediction, partner_noisy, sigma)
                pair_flow_loss = video_mse_loss(partner_clean, partner, training_batch)
            else:
                pair_flow_target = training_batch.noise.permute(0, 2, 1, 3, 4) - partner
                pair_flow_loss = video_mse_loss(partner_prediction, pair_flow_target, training_batch)
            pair_difference_loss = paired_clean_difference_loss(
                prediction, partner_prediction, training_batch.noisy_model_input, partner_noisy,
                sigma, training_batch.latents, partner)
            total_loss = (total_loss + paired_flow_weight * pair_flow_loss +
                          paired_difference_weight * pair_difference_loss)
            loss_map["paired_flow_loss"] = pair_flow_loss
            loss_map["paired_difference_loss"] = pair_difference_loss

        latent_motion_weight = float(getattr(self, "latent_motion_weight", 0.0))
        static_latent_motion_weight = float(getattr(self, "static_latent_motion_weight", 1.0))
        direction_weight = float(getattr(self, "latent_motion_direction_weight", 0.0))
        lowpass_kernel = int(getattr(self, "latent_motion_lowpass_kernel", 1))
        if latent_motion_weight > 0.0 or direction_weight > 0.0:
            motion_kwargs = {
                "max_shift": int(getattr(self, "latent_motion_max_shift", 4)),
                "temperature": float(getattr(self, "latent_motion_temperature", 0.02)),
            }
            predicted_motion_input = lowpass_video(predicted_clean, lowpass_kernel)
            target_motion_input = lowpass_video(training_batch.latents, lowpass_kernel)
            predicted_motion = latent_soft_translation(predicted_motion_input, **motion_kwargs)
            with torch.no_grad():
                target_motion = latent_soft_translation(target_motion_input, **motion_kwargs)
            transition_mask = valid_mask[:, 1:] & valid_mask[:, :-1]
            latent_motion_loss = masked_motion_loss(predicted_motion, target_motion, transition_mask)
            latent_direction_loss = masked_motion_direction_loss(predicted_motion, target_motion, transition_mask)
            total_loss = total_loss + latent_motion_loss * latent_motion_weight
            total_loss = total_loss + latent_direction_loss * direction_weight
        else:
            latent_motion_loss = correct_delta_error.new_zeros(())
            latent_direction_loss = correct_delta_error.new_zeros(())
            transition_mask = valid_mask[:, 1:] & valid_mask[:, :-1]
            motion_kwargs = {}
        loss_map["latent_motion_loss"] = latent_motion_loss

        loss_map["latent_motion_direction_loss"] = latent_direction_loss
        self._worldmodel_microbatch += 1
        static_reconstruction_weight = float(getattr(self, "camera_static_reconstruction_weight", 0.0))
        sensitivity_weight = float(getattr(self, "camera_sensitivity_weight", 0.0))
        response_range_weight = float(getattr(self, "camera_response_range_weight", 0.0))
        same_input = bool(getattr(self, "camera_counterfactual_same_input", False))
        response_min_scale = float(getattr(self, "camera_sensitivity_min_scale", 0.0))
        response_max_scale = float(getattr(self, "camera_sensitivity_max_scale", 1.0))
        response_action_epsilon = float(getattr(self, "camera_response_action_epsilon", 1.0e-6))
        paired_static = False
        run_ranking = (self.camera_ranking_weight > 0.0 or latent_motion_weight > 0.0 or direction_weight > 0.0
                       or sensitivity_weight > 0.0 or response_range_weight > 0.0 or static_reconstruction_weight
                       > 0.0) and self._worldmodel_microbatch % self.camera_ranking_every_microbatches == 0
        if run_ranking:
            camera_to_world = training_batch.input_kwargs["camera_to_world"]
            counterfactual_index = (self._worldmodel_microbatch // self.camera_ranking_every_microbatches - 1) % len(
                self.camera_counterfactual_variants)
            counterfactual_variant = self.camera_counterfactual_variants[counterfactual_index]
            counterfactual_camera = make_camera_trajectory_variant(
                camera_to_world,
                counterfactual_variant,
                seed=self._worldmodel_microbatch,
            )
            response_minimum, response_maximum, response_action_magnitude = action_response_bounds(
                camera_to_world,
                counterfactual_camera,
                response_min_scale,
                response_max_scale,
                normalize_translation=bool(getattr(self.student, "normalize_camera_translation", True)),
            )
            response_observable = response_action_magnitude > response_action_epsilon
            counterfactual_target_latents = training_batch.latents
            counterfactual_noisy_model_input = training_batch.noisy_model_input
            paired_static = counterfactual_variant == "static" and bool(
                getattr(self, "camera_static_paired_target", not same_input))
            if paired_static:
                # Pair static camera supervision with a static noisy video at
                # every sigma. Reverse and shuffled counterfactuals retain the
                # moving-video noisy input so their ranking remains causal.
                if training_batch.noise is None or training_batch.sigmas is None:
                    raise RuntimeError("static counterfactual requires noise and sigmas")
                counterfactual_target_latents, counterfactual_noisy_model_input = paired_static_training_inputs(
                    training_batch.latents,
                    training_batch.noise,
                    training_batch.sigmas,
                )
            training_batch.input_kwargs["camera_to_world"] = counterfactual_camera
            try:
                noisy = counterfactual_noisy_model_input.permute(0, 2, 1, 3, 4)
                counterfactual_prediction = self.student.predict_noise(
                    noisy,
                    training_batch.timesteps,
                    training_batch,
                    conditional=True,
                    attn_kind=self._attn_kind,
                )
            finally:
                training_batch.input_kwargs["camera_to_world"] = camera_to_world
            if not isinstance(counterfactual_prediction, torch.Tensor):
                raise TypeError("camera counterfactual requires a video-only prediction")
            counterfactual_clean = predicted_clean_from_noisy(
                counterfactual_prediction,
                counterfactual_noisy_model_input,
                training_batch.sigmas,
            )
            counterfactual_delta_error = temporal_delta_error(
                counterfactual_clean,
                counterfactual_target_latents,
                valid_mask,
            )
            if (latent_motion_weight > 0.0 or direction_weight > 0.0) and counterfactual_variant != "shuffled":
                counterfactual_motion_input = lowpass_video(counterfactual_clean, lowpass_kernel)
                counterfactual_motion = latent_soft_translation(counterfactual_motion_input, **motion_kwargs)
                counterfactual_target = (torch.zeros_like(target_motion)
                                         if counterfactual_variant == "static" else -target_motion)
                counterfactual_latent_motion_loss = masked_motion_loss(counterfactual_motion, counterfactual_target,
                                                                       transition_mask)
                counterfactual_direction_loss = correct_delta_error.new_zeros(())
                if counterfactual_variant == "reverse":
                    counterfactual_direction_loss = masked_motion_direction_loss(counterfactual_motion,
                                                                                 counterfactual_target, transition_mask)
                # Only one counterfactual is evaluated at the configured
                # cadence. Compensate its weight so changing cadence controls
                # compute cost rather than silently weakening supervision.
                cadence = float(self.camera_ranking_every_microbatches)
                motion_multiplier = static_latent_motion_weight if counterfactual_variant == "static" else 1.0
                total_loss = total_loss + (counterfactual_latent_motion_loss * latent_motion_weight *
                                           motion_multiplier * cadence)
                total_loss = total_loss + counterfactual_direction_loss * direction_weight * cadence
            else:
                counterfactual_latent_motion_loss = correct_delta_error.new_zeros(())
                counterfactual_direction_loss = correct_delta_error.new_zeros(())
            rankable_counterfactual = not paired_static and (counterfactual_variant == "static" or same_input)
            if paired_static or rankable_counterfactual:
                if self.precondition_outputs:
                    counterfactual_flow_error = video_mse_loss(
                        counterfactual_clean,
                        counterfactual_target_latents,
                        training_batch,
                    )
                else:
                    if training_batch.noise is None:
                        raise RuntimeError("camera ranking requires training noise")
                    flow_target = training_batch.noise.permute(0, 2, 1, 3, 4) - counterfactual_target_latents
                    counterfactual_flow_error = video_mse_loss(
                        counterfactual_prediction,
                        flow_target,
                        training_batch,
                    )
                if rankable_counterfactual:
                    correct_flow_error = loss_map["finetune_loss"]
                    margin = correct_flow_error.detach() * self.camera_ranking_margin_ratio
                    ranking_loss = F.relu(margin + correct_flow_error - counterfactual_flow_error)
                    ranking_loss = ranking_loss * response_observable.to(ranking_loss.dtype)
                    total_loss = total_loss + ranking_loss * self.camera_ranking_weight
                else:
                    ranking_loss = correct_delta_error.new_zeros(())
            else:
                counterfactual_flow_error = loss_map["finetune_loss"].detach()
                ranking_loss = correct_delta_error.new_zeros(())
            if same_input and not paired_static:
                response_rms = _stable_root_mean_square(counterfactual_prediction.float() - prediction.float())
                response_mask = response_observable.to(response_rms.dtype)
                sensitivity_loss = F.relu(response_minimum - response_rms) * response_mask
                response_range_loss = F.relu(response_rms - response_maximum) * response_mask
                cadence = float(self.camera_ranking_every_microbatches)
                total_loss = total_loss + sensitivity_loss * sensitivity_weight * cadence
                total_loss = total_loss + response_range_loss * response_range_weight * cadence
            else:
                response_rms = correct_delta_error.new_zeros(())
                response_minimum = correct_delta_error.new_zeros(())
                response_maximum = correct_delta_error.new_zeros(())
                sensitivity_loss = correct_delta_error.new_zeros(())
                response_range_loss = correct_delta_error.new_zeros(())
        else:
            counterfactual_variant = "none"
            counterfactual_delta_error = correct_delta_error.detach()
            counterfactual_flow_error = loss_map["finetune_loss"].detach()
            ranking_loss = correct_delta_error.new_zeros(())
            counterfactual_latent_motion_loss = correct_delta_error.new_zeros(())
            counterfactual_direction_loss = correct_delta_error.new_zeros(())
            response_rms = correct_delta_error.new_zeros(())
            response_minimum = correct_delta_error.new_zeros(())
            response_maximum = correct_delta_error.new_zeros(())
            sensitivity_loss = correct_delta_error.new_zeros(())
            response_range_loss = correct_delta_error.new_zeros(())
        is_static = counterfactual_variant == "static"
        is_reverse = counterfactual_variant == "reverse"
        static_reconstruction_loss = correct_delta_error.new_zeros(())
        if is_static and static_reconstruction_weight > 0.0:
            if not paired_static:
                raise RuntimeError("static reconstruction requires a paired static target")
            if training_batch.noise is None or training_batch.sigmas is None:
                raise RuntimeError("static camera reconstruction requires training noise and sigmas")
            # The static counterfactual is now constructed at the sampled
            # sigma above, so no fixed-sigma restriction is needed.
            static_latents = counterfactual_target_latents
            if self.precondition_outputs:
                static_prediction = counterfactual_clean
                static_target = static_latents
            else:
                static_prediction = counterfactual_prediction.float()
                static_target = training_batch.noise.permute(0, 2, 1, 3, 4).float() - static_latents.float()
            static_reconstruction_loss = F.mse_loss(static_prediction.float(), static_target.float())
            total_loss = total_loss + (static_reconstruction_loss * static_reconstruction_weight *
                                       self.camera_ranking_every_microbatches)
        loss_map["camera_counterfactual_latent_motion_loss"] = counterfactual_latent_motion_loss
        loss_map["camera_static_reconstruction_loss"] = static_reconstruction_loss
        loss_map["camera_static_latent_motion_loss"] = (counterfactual_latent_motion_loss
                                                        if is_static else correct_delta_error.new_zeros(()))
        loss_map["camera_reverse_latent_motion_loss"] = (counterfactual_latent_motion_loss
                                                         if is_reverse else correct_delta_error.new_zeros(()))
        loss_map["camera_static_delta_loss"] = counterfactual_delta_error
        loss_map["camera_static_flow_loss"] = counterfactual_flow_error
        loss_map["camera_counterfactual_delta_loss"] = counterfactual_delta_error
        loss_map["camera_ranking_loss"] = ranking_loss
        loss_map["camera_counterfactual_direction_loss"] = counterfactual_direction_loss
        loss_map["camera_sensitivity_loss"] = sensitivity_loss
        loss_map["camera_response_range_loss"] = response_range_loss
        loss_map["camera_response_rms"] = response_rms
        loss_map["camera_response_minimum"] = response_minimum
        loss_map["camera_response_maximum"] = response_maximum
        loss_map["total_loss"] = total_loss
        if bool(getattr(self, "fail_on_nonfinite_loss", True)):
            tensor_metrics = {name: value for name, value in loss_map.items() if isinstance(value, torch.Tensor)}
            finite_metrics = torch.stack([torch.isfinite(value.detach()).all() for value in tensor_metrics.values()])
            if not bool(finite_metrics.all()):
                nonfinite = [
                    name for name, finite in zip(tensor_metrics, finite_metrics.tolist(), strict=True) if not finite
                ]
                raise FloatingPointError("Non-finite H3 world-model loss before backward: "
                                         f"iteration={iteration}, microbatch={self._worldmodel_microbatch}, "
                                         f"counterfactual={counterfactual_variant}, metrics={nonfinite}")
        return loss_map


__all__ = [
    "MiniMaxH3WorldModelFineTuneMethod",
    "paired_static_training_inputs",
    "predicted_clean_video",
    "temporal_delta_error",
]
