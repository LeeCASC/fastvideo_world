#!/usr/bin/env python3
"""Materialize core 61f poses absent from the original transfer tar."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


ROOT = Path("/datacc05/lizihao/worldmodel/vwm_unified_61f_20261015")
REGISTRY = ROOT / "core_priority_sources_20261016/registry/windows_61_priority_ready_for_vlm.jsonl"
OLD_TAR = Path("/datacc05/lizihao/worldmodel/transfer_packages/vwm_61f_core_filtered_20261016.tar")
OUT = Path("/datacc05/lizihao/worldmodel/transfer_packages/vwm_61f_core_additional_poses_20261018")
POSES = OUT / "poses"


def main() -> None:
    old_names = {
        Path(name).name
        for name in subprocess.check_output(["tar", "-tf", str(OLD_TAR)], text=True).splitlines()
        if "/poses/" in name and name.endswith(".npz")
    }
    OUT.mkdir(parents=True, exist_ok=True)
    POSES.mkdir(parents=True, exist_ok=True)
    manifest = OUT / "missing_poses_manifest.jsonl"
    rows = 0
    copied = 0
    with REGISTRY.open(encoding="utf-8") as src, manifest.open("w", encoding="utf-8") as dst:
        for line in src:
            row = json.loads(line)
            source = Path(row["pose_61f"])
            if source.name in old_names:
                continue
            if not source.is_file():
                raise FileNotFoundError(source)
            target = POSES / source.name
            if not target.exists():
                shutil.copy2(source, target)
                copied += 1
            out_row = {
                "global_id": row["global_id"],
                "window_id": row["window_id"],
                "canonical_source": row["canonical_source"],
                "sample_id": row["sample_id"],
                "pose_source": str(source),
                "pose_target": str(target),
            }
            dst.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            rows += 1
    print(f"missing_rows={rows} copied={copied} output={OUT}")


if __name__ == "__main__":
    main()
