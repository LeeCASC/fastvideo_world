#!/usr/bin/env python3
"""Stage only frozen JS-easy diagnostic rows from completed H3 VAE parquet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

from examples.train.stage_minimax_h3_action_only_pilot import write_shard
from fastvideo.dataset.dataloader.schema import pyarrow_schema_minimax_h3_worldmodel_prope_vae


def stage_rows(plan: Path, vae_input: Path, output: Path, shard_size: int) -> dict:
    if shard_size < 1:
        raise ValueError("shard-size must be positive")
    wanted = [json.loads(line)["parquet_id"] for line in plan.read_text().splitlines() if line.strip()]
    if not wanted or len(wanted) != len(set(wanted)):
        raise ValueError("frozen plan must have unique, nonempty parquet IDs")
    wanted_set = set(wanted)
    found = {}
    paths = sorted(vae_input.rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no completed H3 VAE parquet under {vae_input}")
    for path in paths:
        identities = pq.read_table(path, columns=["id"])["id"].to_pylist()
        if not wanted_set.intersection(identities):
            continue
        if not pq.read_schema(path).equals(pyarrow_schema_minimax_h3_worldmodel_prope_vae, check_metadata=False):
            raise ValueError(f"VAE parquet schema mismatch: {path}")
        for row in pq.read_table(path, schema=pyarrow_schema_minimax_h3_worldmodel_prope_vae).to_pylist():
            identity = row["id"]
            if identity in wanted_set:
                if identity in found:
                    raise ValueError(f"duplicate H3 VAE row: {identity}")
                found[identity] = row
    missing = sorted(wanted_set - found.keys())
    if missing:
        raise ValueError(f"{len(missing)} frozen JS-easy rows have no H3 VAE cache; first={missing[:3]}")
    target = output / "intermediate" / "train"
    for shard_number, start in enumerate(range(0, len(wanted), shard_size)):
        records = [found[identity] for identity in wanted[start:start + shard_size]]
        write_shard(records, target / f"data_{shard_number:06d}.parquet")
    return {"rows": len(wanted), "shards": (len(wanted) + shard_size - 1) // shard_size, "output": str(target)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--vae-input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--shard-size", type=int, default=16)
    args = parser.parse_args()
    report = stage_rows(args.plan, args.vae_input, args.output, args.shard_size)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
