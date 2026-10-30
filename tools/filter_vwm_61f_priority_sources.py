#!/usr/bin/env python3
"""Select the source families prioritized for the 61f GPU-QC/VLM pass."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def keep_source(source: str) -> bool:
    fixed = {
        "ltx",
        "minmax",
        "ue_easy",
        "game_factory",
        "minimax_h3_records_mi_main",
        "ue_stage1",
        "ue_game_video",
        "ue_stage3",
        "sekai_real_drone",
        "sekai_real_walking_hq",
    }
    return source in fixed or source.startswith("h3_js_easy_") or source.startswith("minimax_h3_records_mi_")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    with args.input.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            source = str(row.get("canonical_source") or row.get("source_name") or "")
            if keep_source(source):
                rows.append(row)
    rows.sort(key=lambda row: str(row["global_id"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + f".tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, args.output)
    print(json.dumps({"input": str(args.input), "output": str(args.output), "rows": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
