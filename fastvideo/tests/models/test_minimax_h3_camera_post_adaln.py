import torch
from torch import nn

from fastvideo.configs.models.dits.minimax_h3 import MiniMaxH3ArchConfig
from fastvideo.models.dits.minimax_h3 import MiniMaxH3TransformerBlock, apply_camera_film


class _AdaLN(nn.Module):
    def forward(self, _temb):
        zero = torch.zeros((1, 4))
        one = torch.ones((1, 4))
        return zero, zero, one, zero, zero, zero


class _Attention(nn.Module):
    prope_dim_range = None

    def forward(self, hidden, *_args):
        return hidden


class _ZeroFF(nn.Module):
    def forward(self, hidden):
        return torch.zeros_like(hidden)


def test_post_adaln_camera_film_affects_only_the_bound_video_row():
    block = MiniMaxH3TransformerBlock.__new__(MiniMaxH3TransformerBlock)
    nn.Module.__init__(block)
    block.norm1 = nn.Identity()
    block.norm2 = nn.Identity()
    block.adaln_proj = _AdaLN()
    block.attn = _Attention()
    block.ff = _ZeroFF()
    block.fuse_modulate = False
    hidden = torch.zeros((1, 2, 4))
    action = torch.zeros((1, 2, 8))
    action[0, 1, 4:] = 2.0  # The first row represents text/condition, not future video.
    args = (hidden, torch.zeros((1, 4)), torch.zeros(2, dtype=torch.long),
            (torch.empty(0), torch.empty(0)), 2)

    without = block(*args)
    with_camera = block(*args, action_modulation=action, action_strength=torch.tensor(0.5))

    torch.testing.assert_close(without, hidden)
    torch.testing.assert_close(with_camera[0, 0], hidden[0, 0])
    torch.testing.assert_close(with_camera[0, 1], torch.ones(4))


def test_post_adaln_camera_configuration_requires_film():
    try:
        MiniMaxH3ArchConfig(action_post_adaln=True, action_modulation_mode="additive")
    except ValueError as exc:
        assert "action_post_adaln requires film" in str(exc)
    else:
        raise AssertionError("post-AdaLN without a trainable camera FiLM must fail")


def test_channelwise_camera_film_has_separate_trainable_scale_and_shift():
    hidden = torch.ones((1, 2, 4))
    modulation = torch.cat((torch.ones_like(hidden), 2.0 * torch.ones_like(hidden)), dim=-1)
    strength = torch.zeros((2, 4), requires_grad=True)
    with torch.no_grad():
        strength[0, 0] = 0.5
        strength[1, 1] = 0.25
    output = apply_camera_film(hidden, modulation, strength)
    torch.testing.assert_close(output[0, 0], torch.tensor([1.5, 1.5, 1.0, 1.0]))
    output.sum().backward()
    assert strength.grad is not None
    assert bool((strength.grad != 0).all())


def test_channelwise_camera_gates_require_film():
    try:
        MiniMaxH3ArchConfig(action_channelwise_gates=True, action_modulation_mode="additive")
    except ValueError as exc:
        assert "action_channelwise_gates requires film" in str(exc)
    else:
        raise AssertionError("channelwise camera gates without FiLM must fail")
