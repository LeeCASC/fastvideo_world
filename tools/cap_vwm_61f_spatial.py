#!/usr/bin/env python3
"""Deterministically cap spatialvid rows in a 61f candidate registry."""
from __future__ import annotations
import argparse, hashlib, json
from collections import Counter
from pathlib import Path

def rank(row: dict) -> str:
    return hashlib.sha256(str(row["window_id"]).encode("utf-8")).hexdigest()

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=50000)
    args = ap.parse_args()
    rows = [json.loads(line) for line in args.input.open(encoding="utf-8") if line.strip()]
    spatial = [r for r in rows if r["canonical_source"] == "spatialvid"]
    other = [r for r in rows if r["canonical_source"] != "spatialvid"]
    if len(spatial) <= args.limit:
        selected = spatial
    else:
        selected = sorted(spatial, key=rank)[:args.limit]
    selected = sorted(selected, key=lambda r: r["window_id"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in sorted(other + selected, key=lambda r: r["window_id"]):
            row = dict(row)
            row["spatial_cap"] = args.limit
            row["spatial_selection"] = "sha256_window_id_v1" if len(spatial) > args.limit else "all_available"
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "input_rows": len(rows), "output_rows": len(other) + len(selected),
        "spatial_input": len(spatial), "spatial_selected": len(selected),
        "spatial_cap": args.limit, "selection": "sha256_window_id_v1",
        "source_counts": dict(Counter(r["canonical_source"] for r in other + selected)),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
