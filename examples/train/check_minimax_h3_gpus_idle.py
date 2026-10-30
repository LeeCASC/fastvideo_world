#!/usr/bin/env python3
"""Fail closed unless all eight GPUs have no substantial memory owner.

CUDA remains usable on this host when NVML's user/kernel versions disagree,
so do not interpret a failed nvidia-smi command as proof that GPUs are idle.
"""

from __future__ import annotations

import sys

import torch


EXPECTED_GPUS = 8
MAX_IDLE_USED_GIB = 1.25


def main() -> int:
    if not torch.cuda.is_available() or torch.cuda.device_count() != EXPECTED_GPUS:
        print(f"Expected {EXPECTED_GPUS} working CUDA GPUs", file=sys.stderr)
        return 2
    busy = []
    for index in range(EXPECTED_GPUS):
        free, total = torch.cuda.mem_get_info(index)
        used_gib = (total - free) / 2**30
        if used_gib > MAX_IDLE_USED_GIB:
            busy.append((index, round(used_gib, 2)))
    if busy:
        print(f"GPU memory occupied: {busy}", file=sys.stderr)
        return 1
    print(f"All {EXPECTED_GPUS} CUDA GPUs are idle")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
