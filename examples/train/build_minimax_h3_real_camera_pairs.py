#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Add same-first-frame real partner targets without changing source caches."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from fastvideo.dataset.dataloader.schema import pyarrow_schema_minimax_h3_worldmodel_prope_pair

OPPOSITE = {"W": "S", "S": "W", "A": "D", "D": "A", "I": "K", "K": "I", "J": "L", "L": "J"}
PAIRED_FIELDS = ("vae_latent", "camera_to_world", "camera_intrinsics", "action_valid_mask")
SHARED_FIELDS = ("first_frame_latent", "text_embedding", "text_token_tags")


def _tensor(row: dict, key: str) -> np.ndarray:
    return np.frombuffer(row[f"{key}_bytes"], dtype=np.float32).reshape(row[f"{key}_shape"])


def build(source: Path, ids_path: Path, output: Path) -> dict:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite paired training data: {output}")
    manifest = [json.loads(line) for line in ids_path.read_text().splitlines() if line.strip()]
    by_id = {entry["id"]: entry for entry in manifest}
    parquet_paths = sorted(source.rglob("*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No source parquet files in {source}")
    shards = [(path, pq.read_table(path).to_pylist()) for path in parquet_paths]
    rows = [row for _, shard in shards for row in shard]
    if len(rows) != len(manifest) or len(by_id) != len(manifest):
        raise ValueError("Source rows and audited IDs must have the same unique cardinality")
    for index, row in enumerate(rows):
        if row["id"] not in by_id or by_id[row["id"]]["parquet_index"] != index:
            raise ValueError(f"Audited source index mismatch at {index}")
    row_by_id = {row["id"]: row for row in rows}
    groups: dict[str, list[dict]] = defaultdict(list)
    for entry in manifest:
        groups[entry["shared_first_frame_id"]].append(entry)
    partners = {}
    relations = Counter()
    for group in groups.values():
        directions = {entry["direction"] for entry in group}
        if len(group) < 2 or len(directions) < 2:
            raise ValueError("Every real camera pair needs two directions with one shared first frame")
        for entry in group:
            candidates = [other for other in group if other["direction"] != entry["direction"]]
            opposite = [other for other in candidates if other["direction"] == OPPOSITE[entry["direction"]]]
            # Prefer true opposites. Other same-scene directions still provide
            # a valid two-positive target when an opposite is unavailable.
            chosen = min(opposite or candidates, key=lambda item: item["id"])
            partners[entry["id"]] = chosen["id"]
            relations["opposite" if opposite else "cross_axis"] += 1
    output.mkdir(parents=True, exist_ok=False)
    pair_manifest = []
    for source_path, shard in shards:
        paired_rows = []
        for row in shard:
            partner_id = partners[row["id"]]
            partner = row_by_id[partner_id]
            for field in SHARED_FIELDS:
                if (row[f"{field}_bytes"] != partner[f"{field}_bytes"] or
                        row[f"{field}_shape"] != partner[f"{field}_shape"]):
                    raise ValueError(f"Partner does not share {field}: {row['id']} / {partner_id}")
            if _tensor(row, "vae_latent").shape != _tensor(partner, "vae_latent").shape:
                raise ValueError(f"Paired latent shape mismatch: {row['id']} / {partner_id}")
            for sample in (row, partner):
                camera = _tensor(sample, "camera_to_world")
                intrinsics = _tensor(sample, "camera_intrinsics")
                if camera.shape != (37, 4, 4) or not np.allclose(camera[0], np.eye(4), atol=1e-3):
                    raise ValueError(f"Unrebased or malformed camera path: {sample['id']}")
                if intrinsics.shape != (37, 3, 3) or np.max(np.abs(intrinsics[:, :2, 2])) > 2:
                    raise ValueError(f"Unnormalized or malformed intrinsics: {sample['id']}")
            paired = row.copy()
            paired["pair_id"] = partner_id
            paired["pair_direction"] = by_id[partner_id]["direction"]
            for field in PAIRED_FIELDS:
                for suffix in ("bytes", "shape", "dtype"):
                    paired[f"pair_{field}_{suffix}"] = partner[f"{field}_{suffix}"]
            paired_rows.append(paired)
            pair_manifest.append({
                "id": row["id"], "direction": by_id[row["id"]]["direction"],
                "pair_id": partner_id, "pair_direction": by_id[partner_id]["direction"],
                "same_first_frame": True,
            })
        table = pa.Table.from_pylist(paired_rows, schema=pyarrow_schema_minimax_h3_worldmodel_prope_pair)
        destination = output / source_path.name
        temporary = destination.with_suffix(".parquet.tmp")
        pq.write_table(table, temporary, compression="zstd")
        temporary.replace(destination)
    report = {"rows": len(rows), "groups": len(groups), "relations": dict(relations),
              "source": str(source), "data": str(output)}
    (output.parent / "paired_ids.jsonl").write_text(
        "".join(json.dumps(entry) + "\n" for entry in pair_manifest), encoding="utf-8")
    (output.parent / "pair_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--ids", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.ids, args.output), indent=2))


if __name__ == "__main__":
    main()
