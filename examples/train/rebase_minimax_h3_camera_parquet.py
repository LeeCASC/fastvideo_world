#!/usr/bin/env python3
"""Rebase derived H3 camera paths without changing source videos or latents."""

from __future__ import annotations

import argparse
from pathlib import Path

from fastvideo.pipelines.preprocess.minimax_h3_worldmodel.preprocess import repair_camera_parquet_shard


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int)
    args = parser.parse_args()
    paths = sorted(args.input_dir.rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no parquet under {args.input_dir}")
    total = 0
    changed = 0
    for index, path in enumerate(paths, 1):
        shard_total, shard_changed = repair_camera_parquet_shard(path)
        total += shard_total
        changed += shard_changed
        if index % 16 == 0 or index == len(paths):
            print(f"shards={index}/{len(paths)} rows={total} rebased={changed}", flush=True)
    if args.expected_rows is not None and total != args.expected_rows:
        raise ValueError(f"expected {args.expected_rows} rows, found {total}")
    print(f"complete: rows={total} rebased={changed}")


if __name__ == "__main__":
    main()
