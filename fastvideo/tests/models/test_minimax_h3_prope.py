# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from fastvideo.forward_context import set_forward_context
from fastvideo.models.dits.minimax_h3 import MiniMaxH3Attention
from fastvideo.platforms import AttentionBackendEnum

from fastvideo.configs.models.dits.minimax_h3 import MiniMaxH3ArchConfig
from fastvideo.models.dits.minimax_h3_prope import (
    apply_gated_projective_transform,
    apply_projective_transform,
    blend_projective_attention_residual,
    camera_projection_matrices,
    normalize_camera_intrinsics,
    relative_camera_actions,
    relative_world_to_camera,
)


def _camera_trajectory() -> tuple[torch.Tensor, torch.Tensor]:
    c2w = torch.eye(4).repeat(1, 3, 1, 1)
    c2w[0, 1, 0, 3] = 2.0
    c2w[0, 2, 1, 3] = 1.0
    intrinsics = torch.eye(3).repeat(1, 3, 1, 1)
    intrinsics[..., 0, 0] = 0.84
    intrinsics[..., 1, 1] = 1.44
    intrinsics[..., 0, 2] = 0.5
    intrinsics[..., 1, 2] = 0.5
    return c2w, intrinsics


def test_relative_camera_pose_removes_global_frame_and_normalizes_scale():
    c2w, _ = _camera_trajectory()
    global_frame = torch.eye(4)
    global_frame[:3, :3] = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    global_frame[:3, 3] = torch.tensor([7.0, -3.0, 2.0])
    transformed = torch.matmul(global_frame.view(1, 1, 4, 4), c2w)

    relative = relative_world_to_camera(c2w)
    transformed_relative = relative_world_to_camera(transformed)

    torch.testing.assert_close(relative, transformed_relative, atol=1.0e-6, rtol=1.0e-6)
    torch.testing.assert_close(relative[:, 0], torch.eye(4).view(1, 4, 4))
    torch.testing.assert_close(relative[..., :3, 3].norm(dim=-1).amax(dim=1), torch.ones(1))


def test_relative_camera_actions_encode_cumulative_target_pose():
    c2w = torch.eye(4).repeat(1, 2, 1, 1)
    c2w[:, 1, 0, 3] = 3.0
    angle = torch.tensor(torch.pi / 2)
    c2w[:, 1, 0, 0] = angle.cos()
    c2w[:, 1, 0, 2] = angle.sin()
    c2w[:, 1, 2, 0] = -angle.sin()
    c2w[:, 1, 2, 2] = angle.cos()

    actions = relative_camera_actions(c2w)

    torch.testing.assert_close(actions[:, 0], torch.zeros(1, 6), atol=1.0e-6, rtol=0.0)
    torch.testing.assert_close(
        actions[:, 1],
        torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.5, 0.0]]),
        atol=1.0e-6,
        rtol=0.0,
    )


def test_calibrated_translation_scale_survives_action_and_prope_paths():
    c2w, intrinsics = _camera_trajectory()
    faster = c2w.clone()
    faster[..., :3, 3] *= 2.0

    # Legacy behavior stays scale-invariant for old checkpoints.
    torch.testing.assert_close(relative_camera_actions(c2w), relative_camera_actions(faster))
    base = relative_camera_actions(c2w, normalize_translation=False)
    fast = relative_camera_actions(faster, normalize_translation=False)
    torch.testing.assert_close(fast[..., :3], 2.0 * base[..., :3])

    normalized, _ = camera_projection_matrices(
        c2w, intrinsics, image_width=832, image_height=480,
        use_camera_intrinsics=False,
    )
    normalized_fast, _ = camera_projection_matrices(
        faster, intrinsics, image_width=832, image_height=480,
        use_camera_intrinsics=False,
    )
    torch.testing.assert_close(normalized, normalized_fast)
    calibrated, _ = camera_projection_matrices(
        c2w, intrinsics, image_width=832, image_height=480,
        use_camera_intrinsics=False, normalize_translation=False,
    )
    calibrated_fast, _ = camera_projection_matrices(
        faster, intrinsics, image_width=832, image_height=480,
        use_camera_intrinsics=False, normalize_translation=False,
    )
    torch.testing.assert_close(calibrated_fast[..., :3, 3], 2.0 * calibrated[..., :3, 3])


def test_projection_and_inverse_are_exact_pairs():
    c2w, intrinsics = _camera_trajectory()
    projection, inverse = camera_projection_matrices(
        c2w,
        intrinsics,
        image_width=832,
        image_height=480,
    )
    identity = torch.eye(4).expand_as(projection)
    torch.testing.assert_close(projection @ inverse, identity, atol=1.0e-5, rtol=1.0e-5)


def test_disabling_intrinsics_matches_cosmos_extrinsics_only_path():
    c2w, intrinsics = _camera_trajectory()
    projection, _ = camera_projection_matrices(
        c2w[:, :1],
        intrinsics[:, :1] * 100.0,
        image_width=832,
        image_height=480,
        use_camera_intrinsics=False,
    )
    torch.testing.assert_close(projection[0, 0], torch.eye(4))


def test_vwm_intrinsics_are_centered_without_second_resolution_scaling():
    _, intrinsics = _camera_trajectory()
    centered = normalize_camera_intrinsics(intrinsics, image_width=832, image_height=480)
    torch.testing.assert_close(centered[..., 0, 0], intrinsics[..., 0, 0])
    torch.testing.assert_close(centered[..., 1, 1], intrinsics[..., 1, 1])
    torch.testing.assert_close(centered[..., :2, 2], torch.zeros_like(centered[..., :2, 2]))


def test_pixel_intrinsics_are_rejected_before_projective_attention():
    _, intrinsics = _camera_trajectory()
    intrinsics[..., 0, 0] = 930.0
    with pytest.raises(ValueError, match="normalized before PRoPE"):
        normalize_camera_intrinsics(intrinsics, image_width=832, image_height=480)


def test_projective_qk_pair_preserves_static_camera_inner_products():
    features_q = torch.randn(1, 5, 2, 128)
    features_k = torch.randn(1, 5, 2, 128)
    matrix = torch.tensor(
        [[1.2, 0.0, 0.1, 0.2], [0.0, 0.8, 0.0, -0.1], [0.0, 0.0, 1.0, 0.3], [0.0, 0.0, 0.0, 1.0]]
    )
    projection = matrix.view(1, 1, 4, 4).expand(1, 5, 4, 4)
    inverse = torch.linalg.inv(projection)
    query = apply_projective_transform(features_q, projection.transpose(-1, -2), (96, 108))
    key = apply_projective_transform(features_k, inverse, (96, 108))

    original = torch.einsum("blhd,bmhd->bhlm", features_q[..., 96:108], features_k[..., 96:108])
    transformed = torch.einsum("blhd,bmhd->bhlm", query[..., 96:108], key[..., 96:108])
    torch.testing.assert_close(transformed, original, atol=2.0e-5, rtol=2.0e-5)


def test_zero_gate_exactly_preserves_pretrained_features():
    features = torch.randn(1, 4, 3, 128)
    matrices = torch.randn(1, 4, 4, 4)
    output = apply_gated_projective_transform(features, matrices, (96, 108), torch.zeros(3))
    torch.testing.assert_close(output, features, atol=0.0, rtol=0.0)


def test_full_base_strength_matches_ungated_projective_transform_and_keeps_gate_gradient():
    features = torch.randn(1, 2, 3, 128, requires_grad=True)
    matrices = torch.eye(4).repeat(1, 2, 1, 1)
    matrices[:, 1, 0, 3] = 0.5
    gate = torch.zeros(3, requires_grad=True)

    output = apply_gated_projective_transform(
        features,
        matrices,
        (96, 108),
        gate,
        strength_scale=4.0,
        base_strength=1.0,
    )
    expected = apply_projective_transform(features, matrices, (96, 108))

    torch.testing.assert_close(output, expected)
    output.square().mean().backward()
    assert gate.grad is not None and bool((gate.grad != 0).any())


def test_strength_scale_amplifies_small_gate_without_changing_parameters():
    features = torch.randn(1, 4, 3, 128)
    matrices = torch.eye(4).repeat(1, 4, 1, 1)
    matrices[..., 0, 3] = 0.5
    gate = torch.full((3,), 0.01)

    output_1x = apply_gated_projective_transform(features, matrices, (96, 108), gate)
    output_4x = apply_gated_projective_transform(features, matrices, (96, 108), gate, 4.0)

    assert torch.linalg.vector_norm(output_4x - features) > 3.9 * torch.linalg.vector_norm(output_1x - features)
    torch.testing.assert_close(gate, torch.full((3,), 0.01))


def test_h3_prope_range_cannot_overlap_native_rope():
    config = MiniMaxH3ArchConfig(prope_dim_range=(96, 108))
    assert config.prope_dim_range == (96, 108)
    with pytest.raises(ValueError, match="must not overlap native RoPE"):
        MiniMaxH3ArchConfig(prope_dim_range=(84, 96))


def test_exact_prope_mode_is_valid_but_cannot_use_parallel_residual():
    config = MiniMaxH3ArchConfig(prope_dim_range=(96, 108), prope_mode="exact")
    assert config.prope_mode == "exact"
    with pytest.raises(ValueError, match="gated, exact, or residual"):
        MiniMaxH3ArchConfig(prope_mode="unknown")
    with pytest.raises(ValueError, match="incompatible with parallel"):
        MiniMaxH3ArchConfig(prope_dim_range=(96, 108), prope_mode="exact",
                            prope_parallel_residual_interval=1)


def test_parallel_prope_residual_interval_requires_valid_geometry():
    config = MiniMaxH3ArchConfig(prope_dim_range=(96, 108), prope_parallel_residual_interval=6)
    assert config.prope_parallel_residual_interval == 6
    with pytest.raises(ValueError, match="non-negative"):
        MiniMaxH3ArchConfig(prope_parallel_residual_interval=-1)
    with pytest.raises(ValueError, match="require prope_dim_range"):
        MiniMaxH3ArchConfig(prope_parallel_residual_interval=6)


def test_parallel_prope_residual_preserves_static_and_backpropagates_to_gate():
    base = torch.randn(1, 5, 3, 8)
    gate = torch.zeros(3, requires_grad=True)
    static = blend_projective_attention_residual(base, base.clone(), gate, strength_scale=4.0)
    torch.testing.assert_close(static, base, atol=0.0, rtol=0.0)

    camera = base + torch.randn_like(base)
    zero_gate = blend_projective_attention_residual(base, camera, gate, strength_scale=4.0)
    torch.testing.assert_close(zero_gate, base, atol=0.0, rtol=0.0)
    zero_gate.square().mean().backward()
    assert gate.grad is not None and torch.linalg.vector_norm(gate.grad) > 0


def test_video_only_residual_prope_preserves_base_and_learns(distributed_setup):
    """A zero output projection is an exact baseline; camera sees video only."""
    torch.manual_seed(17)
    kwargs = dict(hidden_size=32, num_attention_heads=2, attention_head_dim=16,
                  qk_norm_eps=1e-5, supported_attention_backends=(AttentionBackendEnum.TORCH_SDPA,),
                  quant_config=None, prefix="test.attn")
    base = MiniMaxH3Attention(**kwargs)
    with torch.no_grad():
        for parameter in base.parameters():
            if parameter.ndim == 2:
                parameter.normal_(std=0.1)
    camera = MiniMaxH3Attention(**kwargs, prope_dim_range=(0, 16), prope_mode="residual")
    camera.load_state_dict(base.state_dict(), strict=False)
    camera.prope_o.requires_grad_(True)
    assert torch.count_nonzero(camera.prope_o.weight) == 0
    hidden = torch.randn(1, 5, 32)
    video_indices = torch.tensor([1, 2, 4])
    projection = torch.eye(4).repeat(1, 5, 1, 1)
    projection[0, 2, 0, 3] = 0.4
    inverse = torch.linalg.inv(projection)
    with set_forward_context(current_timestep=torch.ones(1), attn_metadata=None):
        expected = base(hidden, None, 5)
        unchanged = camera(hidden, None, 5, projection, inverse, video_indices)
    torch.testing.assert_close(unchanged, expected, atol=0, rtol=0)
    unchanged.square().mean().backward()
    assert camera.prope_o.weight.grad is not None
    assert torch.linalg.vector_norm(camera.prope_o.weight.grad) > 0
    with torch.no_grad():
        camera.prope_o.weight.normal_(std=0.01)
    altered_non_video = projection.clone()
    altered_non_video[:, (0, 3), 0, 3] = 100.0
    with set_forward_context(current_timestep=torch.ones(1), attn_metadata=None):
        output = camera(hidden, None, 5, projection, inverse, video_indices)
        isolated = camera(hidden, None, 5, altered_non_video, torch.linalg.inv(altered_non_video), video_indices)
    torch.testing.assert_close(output, isolated, atol=0, rtol=0)
    assert torch.linalg.vector_norm(output - expected) > 0


def test_residual_prope_requires_complete_head_geometry():
    assert MiniMaxH3ArchConfig(prope_mode="residual", prope_dim_range=(0, 128)).prope_dim_range == (0, 128)
    with pytest.raises(ValueError, match="every attention-head channel"):
        MiniMaxH3ArchConfig(prope_mode="residual", prope_dim_range=(96, 120))
