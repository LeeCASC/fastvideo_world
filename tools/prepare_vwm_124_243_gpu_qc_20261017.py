#!/usr/bin/env python3
"""Prepare GPU-QC shards from the corrected 124f/243f CPU-pass registries."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def prepare(root: Path, name: str, frames: int, shards: int) -> dict:
    base = root / name
    registry = read_jsonl(base / "registry" / f"windows_{frames}_candidates.jsonl")
    quality = read_jsonl(base / "reports" / "cpu_quality.jsonl")
    status_by_id = {str(row["global_id"]): row for row in quality}
    selected = []
    for row in registry:
        audit = status_by_id.get(str(row["global_id"]))
        if audit is None or audit.get("status") != "provisional_pass":
            continue
        item = dict(row)
        item["audit_bucket"] = "cpu_provisional_pass"
        item["audit_reason"] = []
        item["action"] = ""
        selected.append(item)
    selected.sort(key=lambda row: str(row["global_id"]))
    out = base / "gpu_qc" / "selections"
    counts = [[] for _ in range(shards)]
    for index, row in enumerate(selected):
        counts[index % shards].append(row)
    for shard, rows in enumerate(counts):
        write_jsonl(out / f"selection_shard_{shard:02d}_of_{shards:02d}.jsonl", rows)
    summary = {
        "frames": frames,
        "registry_rows": len(registry),
        "cpu_provisional_pass_rows": len(selected),
        "shards": shards,
        "shard_counts": [len(rows) for rows in counts],
        "quality_semantics": "CPU provisional_pass; GPU-VLM QC advisory and separate",
    }
    (out / "selection_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--shards", type=int, default=8)
    args = parser.parse_args()
    if args.shards < 1:
        raise ValueError("--shards must be positive")
    prepare(args.root, "124f", 124, args.shards)
    prepare(args.root, "243f", 243, args.shards)


if __name__ == "__main__":
    main()
