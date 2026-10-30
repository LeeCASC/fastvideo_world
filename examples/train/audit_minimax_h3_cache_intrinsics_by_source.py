#!/usr/bin/env python3
"""Audit cached MiniMax-H3 camera K units by source without reading video latents."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.input_dir.rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError(args.input_dir)
    groups: dict[str, dict] = defaultdict(lambda: {
        "rows": 0, "pixel_or_out_of_range": 0, "malformed": 0,
        "fx_min": float("inf"), "fx_max": 0.0,
        "fy_min": float("inf"), "fy_max": 0.0,
        "cx_min": float("inf"), "cx_max": float("-inf"),
        "cy_min": float("inf"), "cy_max": float("-inf"),
        "invalid_ids": [],
    })
    for shard_index, path in enumerate(paths, 1):
        table = pq.read_table(path, columns=[
            "id", "camera_intrinsics_bytes", "camera_intrinsics_shape", "camera_intrinsics_dtype",
        ])
        for row in table.to_pylist():
            identity = row["id"]
            group = groups[identity.split("/", 1)[0]]
            group["rows"] += 1
            try:
                shape = tuple(row["camera_intrinsics_shape"])
                if shape != (37, 3, 3) or row["camera_intrinsics_dtype"] != "float32":
                    raise ValueError("invalid K shape/dtype")
                k = np.frombuffer(row["camera_intrinsics_bytes"], dtype=np.float32).reshape(shape)
                if not np.isfinite(k).all():
                    raise ValueError("nonfinite K")
            except (TypeError, ValueError) as exc:
                group["malformed"] += 1
                group["invalid_ids"].append({"id": identity, "reason": str(exc)})
                continue
            for field, values in (
                ("fx", k[:, 0, 0]), ("fy", k[:, 1, 1]),
                ("cx", k[:, 0, 2]), ("cy", k[:, 1, 2]),
            ):
                group[f"{field}_min"] = min(group[f"{field}_min"], float(values.min()))
                group[f"{field}_max"] = max(group[f"{field}_max"], float(values.max()))
            if (np.any(k[:, 0, 0] <= 0) or np.any(k[:, 1, 1] <= 0)
                    or np.any(k[:, 0, 0] > 32) or np.any(k[:, 1, 1] > 32)
                    or np.any(np.abs(k[:, :2, 2]) > 2)):
                group["pixel_or_out_of_range"] += 1
                group["invalid_ids"].append({"id": identity, "reason": "pixel-valued or out-of-range K"})
        if shard_index % 256 == 0 or shard_index == len(paths):
            print(f"K units audited {shard_index}/{len(paths)} shards", flush=True)
    report = {"shards": len(paths), "rows": sum(group["rows"] for group in groups.values()),
              "sources": dict(sorted(groups.items()))}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"rows": report["rows"], "sources": {
        source: {"rows": group["rows"], "pixel_or_out_of_range": group["pixel_or_out_of_range"],
                 "malformed": group["malformed"]} for source, group in report["sources"].items()}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
