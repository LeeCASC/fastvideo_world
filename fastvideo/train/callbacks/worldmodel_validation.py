# SPDX-License-Identifier: Apache-2.0
"""Fixed-parquet loss validation for MiniMax-H3 camera world models."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, TYPE_CHECKING

import torch
import torch.nn.functional as F

from fastvideo.dataset.dataloader.schema import (
    pyarrow_schema_minimax_h3_worldmodel,
    pyarrow_schema_minimax_h3_worldmodel_prope,
)
from fastvideo.distributed import get_world_group
from fastvideo.logger import init_logger
from fastvideo.train.callbacks.callback import Callback
from fastvideo.train.methods.fine_tuning.worldmodel import paired_static_training_inputs
from fastvideo.train.utils.dataloader import build_parquet_t2v_train_dataloader

if TYPE_CHECKING:
    from fastvideo.pipelines import TrainingBatch
    from fastvideo.train.methods.base import TrainingMethod

logger = init_logger(__name__)


def _video_flow_loss(
    prediction: torch.Tensor,
    batch: TrainingBatch,
    *,
    precondition_outputs: bool,
) -> torch.Tensor:
    """Evaluate the same video-only target used by ``FineTuneMethod``."""
    if batch.latents is None or batch.noisy_model_input is None or batch.noise is None or batch.sigmas is None:
        raise RuntimeError("world-model validation received an incomplete TrainingBatch")
    noisy = batch.noisy_model_input.permute(0, 2, 1, 3, 4)
    if precondition_outputs:
        predicted_clean = noisy - prediction * batch.sigmas
        return F.mse_loss(predicted_clean.float(), batch.latents.float())
    target = batch.noise.permute(0, 2, 1, 3, 4) - batch.latents
    return F.mse_loss(prediction.float(), target.float())


class WorldModelLossValidationCallback(Callback):
    """Measure fixed-noise flow loss and dependence on camera actions."""

    def __init__(
        self,
        *,
        data_path: str,
        every_steps: int = 100,
        num_batches: int = 2,
        seed: int = 20260828,
        run_at_start: bool = True,
    ) -> None:
        if not str(data_path).strip():
            raise ValueError("world-model validation data_path must be non-empty")
        if int(every_steps) <= 0 or int(num_batches) <= 0:
            raise ValueError("every_steps and num_batches must be positive")
        self.data_path = str(data_path)
        self.every_steps = int(every_steps)
        self.num_batches = int(num_batches)
        self.seed = int(seed)
        self.run_at_start = bool(run_at_start)
        self.dataloader: Any = None
        self.camera_conditioning = "additive"

    def on_train_start(self, method: TrainingMethod, iteration: int = 0) -> None:
        del iteration
        if method.student._preprocessed_data_type not in {
            "h3_worldmodel", "h3_worldmodel_prope", "h3_worldmodel_prope_pair"
        }:
            raise TypeError("WorldModelLossValidationCallback requires MiniMaxH3WorldModel")
        self.camera_conditioning = str(method.student.camera_conditioning)
        pipeline_config = self.training_config.pipeline_config
        if pipeline_config is None:
            raise ValueError("world-model validation requires a resolved H3 pipeline config")
        validation_data = replace(
            self.training_config.data,
            data_path=self.data_path,
            seed=self.seed,
        )
        text_len = int(pipeline_config.text_encoder_configs[0].arch_config.text_len)
        self.dataloader = build_parquet_t2v_train_dataloader(
            validation_data,
            text_len=text_len,
            parquet_schema=(
                pyarrow_schema_minimax_h3_worldmodel_prope
                if self.camera_conditioning in {"prope", "hybrid"}
                else pyarrow_schema_minimax_h3_worldmodel
            ),
        )

    def on_validation_begin(self, method: TrainingMethod, iteration: int = 0) -> None:
        if iteration == 0 and not self.run_at_start:
            return
        if iteration % self.every_steps:
            return
        if self.dataloader is None:
            raise RuntimeError("world-model validation dataloader was not initialized")
        self._run_validation(method, iteration)

    @torch.no_grad()
    def _run_validation(self, method: TrainingMethod, iteration: int) -> None:
        student = method.student
        transformer = student.transformer
        was_training = transformer.training
        transformer.eval()
        generator = torch.Generator(device=student.device).manual_seed(self.seed)
        flow_losses: list[torch.Tensor] = []
        zero_conditioning_losses: list[torch.Tensor] = []
        shuffled_conditioning_losses: list[torch.Tensor] = []
        conditioning_responses: list[torch.Tensor] = []
        paired_static_losses: list[torch.Tensor] = []
        try:
            for batch_index, raw_batch in enumerate(self.dataloader):
                if batch_index >= self.num_batches:
                    break
                prepared = student.prepare_batch(raw_batch, generator=generator, latents_source="data")
                if prepared.noisy_model_input is None or prepared.timesteps is None or prepared.input_kwargs is None:
                    raise RuntimeError("world-model validation batch is incomplete")
                noisy = prepared.noisy_model_input.permute(0, 2, 1, 3, 4)
                prediction = student.predict_noise(noisy, prepared.timesteps, prepared, conditional=True)
                if not isinstance(prediction, torch.Tensor):
                    raise TypeError("world-model validation requires one video prediction")
                flow_losses.append(
                    _video_flow_loss(
                        prediction,
                        prepared,
                        precondition_outputs=bool(self.training_config.model.precondition_outputs),
                    ))

                if self.camera_conditioning in {"prope", "hybrid"}:
                    conditioning_name = "camera_to_world"
                    conditioning = prepared.input_kwargs[conditioning_name]
                    zero_conditioning = conditioning[:, :1].expand_as(conditioning).clone()
                    mask_name = None
                    conditioning_mask = None
                else:
                    conditioning_name = "actions"
                    conditioning = prepared.input_kwargs[conditioning_name]
                    zero_conditioning = torch.zeros_like(conditioning)
                    mask_name = "action_mask"
                    conditioning_mask = prepared.input_kwargs[mask_name]
                prepared.input_kwargs[conditioning_name] = zero_conditioning
                try:
                    zero_conditioning_prediction = student.predict_noise(
                        noisy,
                        prepared.timesteps,
                        prepared,
                        conditional=True,
                    )
                finally:
                    prepared.input_kwargs[conditioning_name] = conditioning
                if not isinstance(zero_conditioning_prediction, torch.Tensor):
                    raise TypeError("world-model validation requires one zero-conditioning video prediction")
                zero_conditioning_losses.append(
                    _video_flow_loss(
                        zero_conditioning_prediction,
                        prepared,
                        precondition_outputs=bool(self.training_config.model.precondition_outputs),
                    ))
                conditioning_responses.append(
                    (prediction.float() - zero_conditioning_prediction.float()).square().mean().sqrt()
                )

                if bool(getattr(method, "camera_static_paired_target", False)):
                    if prepared.latents is None or prepared.noise is None or prepared.sigmas is None:
                        raise RuntimeError("paired-static validation requires clean latents, noise, and sigmas")
                    static_latents, static_noisy = paired_static_training_inputs(
                        prepared.latents,
                        prepared.noise,
                        prepared.sigmas,
                    )
                    static_prepared = replace(
                        prepared,
                        latents=static_latents,
                        noisy_model_input=static_noisy,
                        input_kwargs={
                            **prepared.input_kwargs,
                            conditioning_name: zero_conditioning,
                        },
                    )
                    paired_static_prediction = student.predict_noise(
                        static_noisy.permute(0, 2, 1, 3, 4),
                        prepared.timesteps,
                        static_prepared,
                        conditional=True,
                    )
                    if not isinstance(paired_static_prediction, torch.Tensor):
                        raise TypeError("world-model validation requires one paired-static video prediction")
                    paired_static_losses.append(
                        _video_flow_loss(
                            paired_static_prediction,
                            static_prepared,
                            precondition_outputs=bool(self.training_config.model.precondition_outputs),
                        )
                    )

                shuffled_conditioning = conditioning.clone()
                shuffled_mask = conditioning_mask.clone() if conditioning_mask is not None else None
                if conditioning.shape[1] > 2:
                    permutation = torch.randperm(
                        conditioning.shape[1] - 1,
                        generator=generator,
                        device=conditioning.device,
                    )
                    shuffled_conditioning[:, 1:] = conditioning[:, 1:][:, permutation]
                    if shuffled_mask is not None and conditioning_mask is not None:
                        shuffled_mask[:, 1:] = conditioning_mask[:, 1:][:, permutation]
                prepared.input_kwargs[conditioning_name] = shuffled_conditioning
                if mask_name is not None and shuffled_mask is not None:
                    prepared.input_kwargs[mask_name] = shuffled_mask
                try:
                    shuffled_conditioning_prediction = student.predict_noise(
                        noisy,
                        prepared.timesteps,
                        prepared,
                        conditional=True,
                    )
                finally:
                    prepared.input_kwargs[conditioning_name] = conditioning
                    if mask_name is not None and conditioning_mask is not None:
                        prepared.input_kwargs[mask_name] = conditioning_mask
                if not isinstance(shuffled_conditioning_prediction, torch.Tensor):
                    raise TypeError("world-model validation requires one shuffled-conditioning video prediction")
                shuffled_conditioning_losses.append(
                    _video_flow_loss(
                        shuffled_conditioning_prediction,
                        prepared,
                        precondition_outputs=bool(self.training_config.model.precondition_outputs),
                    ))
        finally:
            transformer.train(was_training)

        if not flow_losses:
            raise RuntimeError(f"world-model validation dataset is empty: {self.data_path}")
        metric_values = [
            torch.stack(flow_losses).mean(),
            torch.stack(zero_conditioning_losses).mean(),
            torch.stack(shuffled_conditioning_losses).mean(),
            torch.stack(conditioning_responses).mean(),
        ]
        if paired_static_losses:
            metric_values.append(torch.stack(paired_static_losses).mean())
        local_metrics = torch.stack(metric_values)
        world_group = get_world_group()
        metrics = world_group.all_reduce(local_metrics) / world_group.world_size
        if world_group.rank == 0 and method.tracker is not None:
            flow_loss, zero_conditioning_loss, shuffled_conditioning_loss, conditioning_response = (
                float(value) for value in metrics[:4]
            )
            metric_label = "camera_pose" if self.camera_conditioning == "prope" else self.camera_conditioning
            logged_metrics = {
                "validation_worldmodel/flow_loss": flow_loss,
                f"validation_worldmodel/zero_{metric_label}_flow_loss": zero_conditioning_loss,
                f"validation_worldmodel/{metric_label}_loss_gain": zero_conditioning_loss - flow_loss,
                f"validation_worldmodel/shuffled_{metric_label}_flow_loss": shuffled_conditioning_loss,
                f"validation_worldmodel/shuffled_{metric_label}_loss_gain": shuffled_conditioning_loss - flow_loss,
                f"validation_worldmodel/{metric_label}_response_rms": conditioning_response,
            }
            if paired_static_losses:
                logged_metrics[f"validation_worldmodel/paired_static_{metric_label}_flow_loss"] = float(metrics[4])
            method.tracker.log(logged_metrics, iteration)
        logger.info("Finished H3 world-model loss validation at step %d", iteration)


__all__ = ["WorldModelLossValidationCallback"]
