#!/usr/bin/env python3
"""Build the exact 124f strict-QC VLM queue and reuse existing captions."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq


ROOT = Path("/datacc05/lizihao/worldmodel/vwm_unified_124_243_20261016")
BASE = ROOT / "124f"
OLD_CACHE = Path("/datacc05/lizihao/worldmodel/fastvideowm_h3_latent_approved_v1_20260925")
OLD_VLM = BASE / "vlm_qwen3vl"
OUT = BASE / "vlm_qwen3vl_strict_50599"
SHARDS = 4


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def key(value: str) -> str:
    value = str(value)
    if ":" in value:
        source, sample = value.split(":", 1)
        return f"{source}/{sample}"
    return value


def old_cache_keys() -> set[str]:
    result: set[str] = set()
    for path in OLD_CACHE.rglob("*.parquet"):
        try:
            table = pq.read_table(path, columns=["id"])
        except Exception:
            continue
        result.update(key(x) for x in table["id"].to_pylist())
    return result


def main() -> None:
    registry = {x["global_id"]: x for x in read_jsonl(BASE / "registry/windows_124_candidates.jsonl")}
    old_keys = old_cache_keys()
    selected = []
    for path in sorted((BASE / "gpu_qc/results").glob("*.jsonl")):
        for result in read_jsonl(path):
            assessment = result.get("assessment") or {}
            if result.get("error") or result.get("parse_error"):
                continue
            if assessment.get("uncertain") is True or assessment.get("visible_camera_motion_match") != "yes":
                continue
            if any(assessment.get(k) is True for k in ("blur", "texture_breakdown", "subject_deformation", "translation_jitter")):
                continue
            row = registry[result["global_id"]]
            if key(f"{row['canonical_source']}/{row['sample_id']}") not in old_keys:
                selected.append(row)
    selected.sort(key=lambda x: str(x["global_id"]))
    if len(selected) != len({x["global_id"] for x in selected}):
        raise RuntimeError("duplicate strict VLM rows")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "registry_strict_50599.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in selected), encoding="utf-8"
    )

    # Reuse already completed captions from the old broad queue, partitioned
    # exactly as the new caption workers will partition the strict registry.
    by_id = {}
    for path in OLD_VLM.glob("captions_shard_*.jsonl"):
        for item in read_jsonl(path):
            if item.get("error") in (None, "") and item.get("scene_description"):
                by_id[str(item.get("global_id"))] = item
    kept = defaultdict(list)
    for position, row in enumerate(selected):
        item = by_id.get(str(row["global_id"]))
        if item is not None:
            kept[position % SHARDS].append(item)
    for shard in range(SHARDS):
        path = OUT / f"captions_shard_{shard:02d}_of_{SHARDS:02d}.jsonl"
        path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in kept[shard]), encoding="utf-8")
    print(f"strict_rows={len(selected)} reused_captions={sum(map(len, kept.values()))}")
    for shard in range(SHARDS):
        print(f"shard={shard} reused={len(kept[shard])}")


if __name__ == "__main__":
    main()
