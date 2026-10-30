# SPDX-License-Identifier: Apache-2.0

import pytest

from fastvideo.configs.models.dits.minimax_h3 import MiniMaxH3ArchConfig


def test_minimax_h3_action_adapter_is_opt_in():
    assert MiniMaxH3ArchConfig().action_dim == 0
    assert MiniMaxH3ArchConfig(action_dim=6).action_dim == 6
    repeated = MiniMaxH3ArchConfig(
        action_dim=55,
        action_injection_interval=4,
        action_residual_scale=0.5,
        action_residual_clip=0.1,
    )
    assert repeated.action_injection_interval == 4
    assert repeated.action_residual_scale == 0.5
    assert repeated.action_residual_clip == 0.1
    with pytest.raises(ValueError, match="action_dim"):
        MiniMaxH3ArchConfig(action_dim=-1)
    with pytest.raises(ValueError, match="action_injection_interval"):
        MiniMaxH3ArchConfig(action_injection_interval=-1)
    with pytest.raises(ValueError, match="action_residual_scale"):
        MiniMaxH3ArchConfig(action_residual_scale=0.0)
    with pytest.raises(ValueError, match="action_residual_clip"):
        MiniMaxH3ArchConfig(action_residual_clip=-0.1)
