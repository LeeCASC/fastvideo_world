# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from fastvideo.pipelines.basic.minimax_h3.packing import MINIMAX_H3_KEYFRAME_ENCODE_SEED
from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_conditioning import MINIMAX_H3_TEXT_TOKEN_TAGS_KEY
from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_input_preparation import MINIMAX_H3_KEYFRAMES_KEY
from fastvideo.pipelines.preprocess.minimax_h3_worldmodel.encoding import (
    encode_h3_worldmodel_condition,
    encode_h3_worldmodel_video,
)


class _Posterior:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape

    def sample(self, generator: torch.Generator) -> torch.Tensor:
        assert generator.initial_seed() == MINIMAX_H3_KEYFRAME_ENCODE_SEED
        return torch.full(self.shape, 1.0001)


class _VAE:
    @staticmethod
    def normalize_pixels(pixels: torch.Tensor) -> torch.Tensor:
        return pixels * 2.0 - 1.0

    @staticmethod
    def normalize_latents(latents: torch.Tensor) -> torch.Tensor:
        return latents * 3.0

    @staticmethod
    def encode_pixels(pixels: torch.Tensor) -> SimpleNamespace:
        assert pixels.shape == (1, 3, 5, 16, 32)
        assert pixels.device.type == "cpu" and pixels.dtype == torch.uint8
        return SimpleNamespace(latent_dist=_Posterior((1, 24, 2, 1, 2)))

    @staticmethod
    def encode_keyframe(pixels: torch.Tensor) -> SimpleNamespace:
        assert pixels.shape == (1, 3, 1, 16, 32)
        return SimpleNamespace(latent_dist=_Posterior((1, 24, 1, 1, 2)))


def test_encode_h3_worldmodel_video_preserves_native_latent_contract():
    video = torch.zeros((3, 5, 16, 32), dtype=torch.uint8)
    target, first_frame = encode_h3_worldmodel_video(_VAE(), video, device=torch.device("cpu"))

    assert target.shape == (24, 2, 1, 2)
    assert first_frame.shape == (24, 1, 1, 2)
    assert target.dtype == torch.float32
    assert target.is_contiguous() and first_frame.is_contiguous()
    assert torch.allclose(target, torch.full_like(target, 3.0003))
    assert torch.all(first_frame == 3.0)


def test_encode_h3_worldmodel_video_rejects_non_uint8_pixels():
    with pytest.raises(ValueError, match="uint8"):
        encode_h3_worldmodel_video(
            _VAE(),
            torch.zeros((3, 5, 16, 32), dtype=torch.float32),
            device=torch.device("cpu"),
        )


class _ConditioningStage:
    def forward(self, batch, fastvideo_args):
        del fastvideo_args
        image = batch.extra[MINIMAX_H3_KEYFRAMES_KEY][0]
        assert isinstance(image, Image.Image) and image.mode == "RGB"
        batch.prompt_embeds = [torch.zeros((1, 4, 5120), dtype=torch.bfloat16)]
        batch.extra[MINIMAX_H3_TEXT_TOKEN_TAGS_KEY] = torch.tensor([0, 0, 1, 1])
        return batch


def test_encode_h3_worldmodel_condition_keeps_visual_token_tags():
    embedding, tags = encode_h3_worldmodel_condition(
        _ConditioningStage(),
        SimpleNamespace(),
        caption="move forward",
        first_frame=np.zeros((480, 832, 3), dtype=np.uint8),
    )

    assert embedding.shape == (4, 5120)
    assert embedding.dtype == torch.float32
    assert tags.tolist() == [0, 0, 1, 1]
