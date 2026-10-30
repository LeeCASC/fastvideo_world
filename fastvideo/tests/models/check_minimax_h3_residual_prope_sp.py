#!/usr/bin/env python3
"""Two-rank SP regression for the MiniMax-H3 video-only PRoPE branch.

Run with ``torchrun --nproc_per_node=2`` on two free or lightly loaded GPUs.
"""

import os

import torch

from fastvideo.configs.models.dits.minimax_h3 import MiniMaxH3ArchConfig
from fastvideo.distributed import cleanup_dist_env_and_memory, maybe_init_distributed_environment_and_model_parallel
from fastvideo.distributed.communication_op import sequence_model_parallel_shard
from fastvideo.forward_context import set_forward_context
from fastvideo.models.dits.minimax_h3 import MiniMaxH3Attention
from fastvideo.platforms import AttentionBackendEnum


def main() -> None:
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    maybe_init_distributed_environment_and_model_parallel(1, 2)
    try:
        torch.manual_seed(17)
        MiniMaxH3ArchConfig(prope_mode="residual", prope_dim_range=(0, 128))
        kwargs = dict(hidden_size=32, num_attention_heads=2, attention_head_dim=16,
                      qk_norm_eps=1e-5, supported_attention_backends=(AttentionBackendEnum.TORCH_SDPA,),
                      quant_config=None, prefix="spcheck.attn")
        base = MiniMaxH3Attention(**kwargs)
        with torch.no_grad():
            for parameter in base.parameters():
                if parameter.ndim == 2:
                    parameter.normal_(std=0.1)
        camera = MiniMaxH3Attention(**kwargs, prope_dim_range=(0, 16), prope_mode="residual")
        camera.load_state_dict(base.state_dict(), strict=False)
        camera.prope_o.requires_grad_(True)
        base.cuda(rank)
        camera.cuda(rank)
        hidden = torch.randn(1, 6, 32, device=f"cuda:{rank}")
        hidden, _ = sequence_model_parallel_shard(hidden, dim=1)
        indices = torch.tensor([1, 2, 4, 5], device=f"cuda:{rank}")
        projection = torch.eye(4, device=f"cuda:{rank}").repeat(1, 6, 1, 1)
        projection[:, 2, 0, 3] = 0.4
        inverse = torch.linalg.inv(projection)
        with set_forward_context(current_timestep=torch.ones(1, device=f"cuda:{rank}"), attn_metadata=None):
            expected = base(hidden, None, 6)
            actual = camera(hidden, None, 6, projection, inverse, indices)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        actual.square().mean().backward()
        if camera.prope_o.weight.grad is None or not bool((camera.prope_o.weight.grad != 0).any()):
            raise AssertionError(f"rank {rank}: camera output projection received zero gradient")
        print(f"rank {rank}: SP=2 baseline parity and camera gradient passed", flush=True)
    finally:
        cleanup_dist_env_and_memory()


if __name__ == "__main__":
    main()
