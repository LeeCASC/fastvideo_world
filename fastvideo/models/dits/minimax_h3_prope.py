# SPDX-License-Identifier: Apache-2.0
"""Camera projection utilities for MiniMax-H3 PRoPE conditioning."""

from __future__ import annotations

import torch


def invert_se3(transforms: torch.Tensor) -> torch.Tensor:
    """Invert rigid homogeneous transforms without a generic matrix inverse."""
    if transforms.shape[-2:] != (4, 4):
        raise ValueError(f"SE(3) tensors must end in [4,4], got {tuple(transforms.shape)}")
    rotation_inverse = transforms[..., :3, :3].transpose(-1, -2)
    inverse = torch.zeros_like(transforms)
    inverse[..., :3, :3] = rotation_inverse
    inverse[..., :3, 3] = -torch.matmul(
        rotation_inverse,
        transforms[..., :3, 3].unsqueeze(-1),
    ).squeeze(-1)
    inverse[..., 3, 3] = 1.0
    return inverse


def normalize_camera_intrinsics(
    intrinsics: torch.Tensor,
    *,
    image_width: int,
    image_height: int,
) -> torch.Tensor:
    """Convert canonical VWM normalized intrinsics to a centered image plane.

    VWM stores ``[fx/w, fy/h, cx/w, cy/h]`` and its resize/crop transform
    preserves that convention. Cosmos PRoPE additionally centers the principal
    point, so only subtract 0.5 from ``cx`` and ``cy`` here.
    """
    if intrinsics.shape[-2:] != (3, 3):
        raise ValueError(f"camera intrinsics must end in [3,3], got {tuple(intrinsics.shape)}")
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    normalized = torch.zeros_like(intrinsics, dtype=torch.float32)
    source = intrinsics.float()
    if not bool(torch.isfinite(source).all()):
        raise ValueError("camera intrinsics contain non-finite values")
    if bool((source[..., 0, 0] > 32.0).any() or (source[..., 1, 1] > 32.0).any()
            or (source[..., :2, 2].abs() > 2.0).any()):
        raise ValueError("camera intrinsics must be normalized before PRoPE; pixel-valued K was supplied")
    normalized[..., 0, 0] = source[..., 0, 0]
    normalized[..., 1, 1] = source[..., 1, 1]
    normalized[..., 0, 2] = source[..., 0, 2] - 0.5
    normalized[..., 1, 2] = source[..., 1, 2] - 0.5
    normalized[..., 2, 2] = 1.0
    if bool((normalized[..., (0, 1), (0, 1)] <= 0.0).any()):
        raise ValueError("camera intrinsics must have positive focal lengths")
    return normalized


def relative_world_to_camera(
    camera_to_world: torch.Tensor,
    *,
    normalize_translation: bool = True,
) -> torch.Tensor:
    """Express every view relative to frame zero and normalize path scale.

    ``w2c_t @ c2w_0`` makes frame zero identity and removes arbitrary global
    coordinate frames. The legacy default divides by one maximum translation
    norm; disabling it preserves pre-calibrated cross-video speed factors.
    """
    if camera_to_world.ndim != 4 or camera_to_world.shape[-2:] != (4, 4):
        raise ValueError(f"camera_to_world must have shape [B,T,4,4], got {tuple(camera_to_world.shape)}")
    source = camera_to_world.float()
    relative = torch.matmul(invert_se3(source), source[:, :1])
    if not normalize_translation:
        return relative
    translation = relative[..., :3, 3]
    maximum_norm = translation.norm(dim=-1).amax(dim=1, keepdim=True)
    safe_norm = torch.where(maximum_norm > 1.0e-8, maximum_norm, torch.ones_like(maximum_norm))
    normalized = relative.clone()
    normalized[..., :3, 3] = translation / safe_norm.unsqueeze(-1)
    return normalized


def relative_camera_actions(
    camera_to_world: torch.Tensor,
    *,
    normalize_translation: bool = True,
) -> torch.Tensor:
    """Encode each camera relative to frame zero as cumulative 6-DoF action.

    The output order matches the VWM action contract:
    ``[forward, right, up, pitch, yaw, roll]``. Legacy translation is divided
    by the maximum path displacement unless calibrated-scale preservation is
    requested. Rotation is an SO(3) log vector in half-turn units. Unlike tiny
    adjacent-frame deltas, this gives every video token its target camera state.
    Pitch/roll use SO(3) axial signs, not the VWM Euler-angle text-key signs.
    """
    if camera_to_world.ndim != 4 or camera_to_world.shape[-2:] != (4, 4):
        raise ValueError(f"camera_to_world must have shape [B,T,4,4], got {tuple(camera_to_world.shape)}")
    source = camera_to_world.float()
    relative = torch.matmul(invert_se3(source[:, :1]), source)
    translation = relative[..., :3, 3]
    if normalize_translation:
        maximum_norm = translation.norm(dim=-1).amax(dim=1, keepdim=True)
        safe_norm = torch.where(maximum_norm > 1.0e-8, maximum_norm, torch.ones_like(maximum_norm))
        translation = translation / safe_norm.unsqueeze(-1)

    rotation = relative[..., :3, :3]
    trace = rotation.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    cosine = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    angle = torch.acos(cosine)
    vee = torch.stack(
        (
            rotation[..., 2, 1] - rotation[..., 1, 2],
            rotation[..., 0, 2] - rotation[..., 2, 0],
            rotation[..., 1, 0] - rotation[..., 0, 1],
        ),
        dim=-1,
    )
    sine = torch.sin(angle)
    scale = torch.where(
        angle.abs() < 1.0e-4,
        torch.full_like(angle, 0.5),
        angle / (2.0 * sine.clamp_min(1.0e-6)),
    )
    rotation_vector = vee * scale.unsqueeze(-1) / torch.pi
    # Relative c2w axes are [right, up, forward] and SO(3) log axes are
    # [pitch, yaw, roll]. Reorder translation to the canonical action layout.
    return torch.cat(
        (
            translation[..., (2, 0, 1)],
            rotation_vector,
        ),
        dim=-1,
    )


def camera_projection_matrices(
    camera_to_world: torch.Tensor,
    intrinsics: torch.Tensor,
    *,
    image_width: int,
    image_height: int,
    use_camera_intrinsics: bool = True,
    normalize_translation: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build PRoPE ``P=K@w2c`` and its inverse for each latent frame."""
    if intrinsics.shape[:2] != camera_to_world.shape[:2]:
        raise ValueError("camera_to_world and intrinsics must align in batch and time")
    view = relative_world_to_camera(camera_to_world, normalize_translation=normalize_translation)
    if use_camera_intrinsics:
        intrinsic = normalize_camera_intrinsics(
            intrinsics,
            image_width=image_width,
            image_height=image_height,
        )
    else:
        # Match Cosmos ``use_ks=False``: extrinsics only. This also makes a
        # static relative camera path identity for the residual branch.
        intrinsic = torch.eye(3, device=view.device, dtype=torch.float32).expand(
            *view.shape[:-2], 3, 3)
    lifted = torch.zeros_like(view)
    lifted[..., :3, :3] = intrinsic
    lifted[..., 3, 3] = 1.0
    projection = torch.matmul(lifted, view)
    # Intrinsics can contain crop-adjusted principal points/skew, so use the
    # exact inverse rather than the zero-skew shortcut from the fixed-K model.
    inverse_projection = torch.matmul(invert_se3(view), torch.linalg.inv(lifted))
    return projection, inverse_projection


def apply_projective_transform(
    features: torch.Tensor,
    matrices: torch.Tensor,
    dim_range: tuple[int, int],
) -> torch.Tensor:
    """Apply one token-aligned 4x4 transform to a head feature subspace."""
    if features.ndim != 4:
        raise ValueError(f"features must have shape [B,L,H,D], got {tuple(features.shape)}")
    if matrices.shape != (features.shape[0], features.shape[1], 4, 4):
        raise ValueError(
            "PRoPE matrices must have shape [B,L,4,4], "
            f"got {tuple(matrices.shape)} for features {tuple(features.shape)}"
        )
    start, end = dim_range
    width = end - start
    if not 0 <= start < end <= features.shape[-1] or width % 4:
        raise ValueError(f"invalid homogeneous PRoPE dimension range: {dim_range}")
    selected = features[..., start:end].float().reshape(
        features.shape[0],
        features.shape[1],
        features.shape[2],
        width // 4,
        4,
    )
    # Camera geometry must not be silently autocast to BF16 inside the DiT.
    # This matters especially for the full-head residual PRoPE branch.
    with torch.autocast(device_type=features.device.type, enabled=False):
        transformed = torch.einsum("blij,blhgj->blhgi", matrices.float(), selected)
    output = features.clone()
    output[..., start:end] = transformed.flatten(-2).to(features.dtype)
    return output


def make_camera_trajectory_variant(
    camera_to_world: torch.Tensor,
    variant: str,
    *,
    seed: int,
) -> torch.Tensor:
    """Build a counterfactual c2w path while preserving the first camera."""
    if variant not in {"correct", "static", "reverse", "shuffled"}:
        raise ValueError(f"Unknown real trajectory variant {variant!r}")
    if camera_to_world.ndim != 4 or camera_to_world.shape[-2:] != (4, 4):
        raise ValueError("camera_to_world must have shape [B,T,4,4]")
    if variant == "correct":
        return camera_to_world.clone()
    if variant == "static":
        return camera_to_world[:, :1].expand_as(camera_to_world).clone()
    if variant == "reverse":
        reference = camera_to_world[:, :1]
        relative = torch.linalg.solve(reference, camera_to_world)
        return torch.matmul(reference, torch.linalg.inv(relative))
    shuffled = camera_to_world.clone()
    if camera_to_world.shape[1] > 2:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        permutation = torch.randperm(camera_to_world.shape[1] - 1, generator=generator)
        shuffled[:, 1:] = camera_to_world[:, 1:][:, permutation]
    return shuffled


def apply_gated_projective_transform(
    features: torch.Tensor,
    matrices: torch.Tensor,
    dim_range: tuple[int, int],
    gate: torch.Tensor,
    strength_scale: float = 1.0,
    base_strength: float = 0.0,
) -> torch.Tensor:
    """Blend PRoPE per head around an optional fixed-strength baseline."""
    if gate.shape != (features.shape[2],):
        raise ValueError(f"PRoPE gate must have shape [{features.shape[2]}], got {tuple(gate.shape)}")
    if strength_scale <= 0.0:
        raise ValueError(f"PRoPE strength scale must be positive, got {strength_scale}")
    if not 0.0 <= base_strength <= 1.0:
        raise ValueError(f"PRoPE base strength must be in [0,1], got {base_strength}")
    transformed = apply_projective_transform(features, matrices, dim_range)
    strength = (
        float(base_strength) + torch.tanh(gate.float() * float(strength_scale))
    ).view(1, 1, -1, 1)
    return features + (transformed - features) * strength.to(features.dtype)


def blend_projective_attention_residual(
    base_attention: torch.Tensor,
    projective_attention: torch.Tensor,
    gate: torch.Tensor,
    *,
    strength_scale: float = 1.0,
    base_strength: float = 0.0,
) -> torch.Tensor:
    """Add a gated camera-attention delta while retaining base H3 attention."""
    if base_attention.shape != projective_attention.shape or base_attention.ndim != 4:
        raise ValueError("parallel PRoPE attention tensors must share shape [B,L,H,D]")
    if gate.shape != (base_attention.shape[2],):
        raise ValueError(f"PRoPE gate must have shape [{base_attention.shape[2]}], got {tuple(gate.shape)}")
    if strength_scale <= 0.0:
        raise ValueError("parallel PRoPE strength_scale must be positive")
    if not 0.0 <= base_strength <= 1.0:
        raise ValueError("parallel PRoPE base_strength must be in [0,1]")
    strength = (
        float(base_strength) + torch.tanh(gate.float() * float(strength_scale))
    ).view(1, 1, -1, 1)
    return base_attention + (projective_attention - base_attention) * strength.to(base_attention.dtype)


__all__ = [
    "blend_projective_attention_residual",
    "apply_gated_projective_transform",
    "apply_projective_transform",
    "camera_projection_matrices",
    "invert_se3",
    "make_camera_trajectory_variant",
    "normalize_camera_intrinsics",
    "relative_camera_actions",
    "relative_world_to_camera",
]
