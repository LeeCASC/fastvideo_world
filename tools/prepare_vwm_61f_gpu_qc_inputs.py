#!/usr/bin/env python3
"""Prepare deterministic GPU-QC shards for the materialized 61-frame registry."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shards", type=int, default=4)
    args = parser.parse_args()
    if args.shards < 1:
        raise ValueError("shards must be positive")
    rows = read_jsonl(args.registry)
    rows.sort(key=lambda row: str(row["global_id"]))
    selections: list[list[dict]] = [[] for _ in range(args.shards)]
    for index, row in enumerate(rows):
        item = dict(row)
        item.setdefault("audit_bucket", "provisional_pass")
        item.setdefault("audit_reason", [])
        item.setdefault("action", "")
        selections[index % args.shards].append(item)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for shard, selection in enumerate(selections):
        write_jsonl(
            args.output_dir / f"selection_shard_{shard:02d}_of_{args.shards:02d}.jsonl",
            selection,
        )
    summary = {
        "registry": str(args.registry),
        "rows": len(rows),
        "shards": args.shards,
        "counts": [len(selection) for selection in selections],
        "quality_semantics": "CPU provisional_pass; GPU VLM QC remains advisory",
    }
    (args.output_dir / "selection_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
