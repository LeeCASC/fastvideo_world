#!/usr/bin/env python3
"""Select frozen action-only pilot rows from completed H3 VAE parquet shards."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from fastvideo.dataset.dataloader.schema import pyarrow_schema_minimax_h3_worldmodel_prope_vae


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_shard(records: list[dict], output: Path) -> None:
    if output.exists():
        old_ids = pq.read_table(output, columns=["id"])["id"].to_pylist()
        new_ids = [row["id"] for row in records]
        if old_ids != new_ids or not pq.read_schema(output).equals(
            pyarrow_schema_minimax_h3_worldmodel_prope_vae, check_metadata=False
        ):
            raise ValueError(f"existing pilot shard differs from frozen plan: {output}")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f".parquet.tmp.{os.getpid()}")
    try:
        pq.write_table(
            pa.Table.from_pylist(records, schema=pyarrow_schema_minimax_h3_worldmodel_prope_vae),
            temporary,
            compression="zstd",
        )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--source-mapping", type=Path, required=True)
    parser.add_argument("--vae-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-size", type=int, default=16)
    args = parser.parse_args()
    if args.shard_size < 1:
        raise ValueError("shard-size must be positive")
    report = json.loads((args.plan / "report.json").read_text())
    wanted = {}
    for split in ("train", "validation"):
        for row in read_jsonl(args.plan / f"pilot_{split}_ids.jsonl"):
            if row["global_id"] in wanted:
                raise ValueError(f"duplicate pilot ID: {row['global_id']}")
            wanted[row["global_id"]] = split
    if len(wanted) != report["pilot_train"] + report["pilot_validation"]:
        raise ValueError("pilot plan count changed")
    mapped = {}
    for row in read_jsonl(args.source_mapping):
        identity = row["global_id"]
        if identity in wanted:
            mapped[f"{row['source_name']}/{row['sample_id']}"] = (identity, wanted[identity])
    if len(mapped) != len(wanted):
        raise ValueError(f"pilot IDs absent from derived VWM manifest: {len(wanted) - len(mapped)}")

    buffers = {"train": [], "validation": []}
    shard_indices = {"train": 0, "validation": 0}
    seen = set()
    sources = Counter()
    paths = sorted(args.vae_input.rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no VAE parquet under {args.vae_input}")
    for path in paths:
        ids = pq.read_table(path, columns=["id"])["id"].to_pylist()
        if not any(identity in mapped for identity in ids):
            continue
        if not pq.read_schema(path).equals(pyarrow_schema_minimax_h3_worldmodel_prope_vae, check_metadata=False):
            raise ValueError(f"VAE schema mismatch: {path}")
        table = pq.read_table(path, schema=pyarrow_schema_minimax_h3_worldmodel_prope_vae)
        for record in table.to_pylist():
            identity = record["id"]
            if identity not in mapped:
                continue
            if identity in seen:
                raise ValueError(f"duplicate VAE row: {identity}")
            seen.add(identity)
            global_id, split = mapped[identity]
            sources[(split, global_id.split(":", 1)[0])] += 1
            buffers[split].append(record)
            if len(buffers[split]) == args.shard_size:
                target = args.output / "intermediate" / split / f"data_{shard_indices[split]:06d}.parquet"
                write_shard(buffers[split], target)
                shard_indices[split] += 1
                buffers[split] = []
    if seen != set(mapped):
        missing = sorted(set(mapped) - seen)
        raise ValueError(f"pilot VAE rows incomplete: {len(missing)} missing; first={missing[:5]}")
    for split in ("train", "validation"):
        if buffers[split]:
            target = args.output / "intermediate" / split / f"data_{shard_indices[split]:06d}.parquet"
            write_shard(buffers[split], target)
    result = {
        "status": "pilot_vae_rows_staged",
        "rows": len(seen),
        "sources": {f"{split}/{family}": n for (split, family), n in sorted(sources.items())},
        "training_split": report["pilot_train"],
        "validation_split": report["pilot_validation"],
    }
    (args.output / "staging_report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
