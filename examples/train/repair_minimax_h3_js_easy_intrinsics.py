#!/usr/bin/env python3
"""Repair derived JS-easy H3 Parquet K from the original 672x384 VGGT calibration.

Only camera_intrinsics_bytes changes. Existing source videos, source poses,
video/keyframe latents, first-frame tokens, and camera trajectories are kept.
Each replaced shard gets a recoverable .pre_intrinsics_fix.bak hardlink.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from vwm_datasets.transforms import transform_intrinsic_for_resize_center_crop

from fastvideo.dataset.vwm_h3 import canonicalize_source_intrinsics, h3_latent_source_frame_indices


JS_EASY_SOURCES = {
    "h3_js_easy_fasth3_t2v_js_1000_124",
    "h3_js_easy_fasth3_t2v_js_1000_long",
    "h3_js_easy_h3_i2v_js_easy_124",
    "h3_js_easy_h3_i2v_js_easy_long",
}


def _manifest(path: Path) -> dict[str, dict]:
    rows = (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return {f"{row['source_name']}/{row['sample_id']}": row for row in rows
            if row["source_name"] in JS_EASY_SOURCES}


def _expected_intrinsics(row: dict, record: dict, source_root: Path) -> np.ndarray:
    indices = np.asarray(row["source_frame_indices"], dtype=np.int64)
    if indices.shape != (124,):
        raise ValueError(f"{row['id']}: expected 124 source indices")
    with np.load(source_root / record["record_path"] / "camera_c2w.npz", allow_pickle=False) as archive:
        source_k = np.asarray(archive["intrinsics"], dtype=np.float32)
    if source_k.shape == (3, 3):
        chosen_k = np.broadcast_to(source_k, (37, 3, 3))
    else:
        if source_k.shape != (record["num_frames"], 3, 3):
            raise ValueError(f"{row['id']}: unexpected source K shape {source_k.shape}")
        chosen_k = source_k[indices[h3_latent_source_frame_indices(124)]]
    normalized = canonicalize_source_intrinsics(
        chosen_k,
        source_name=record["source_name"],
        image_width=int(record["width"]),
        image_height=int(record["height"]),
    )
    return np.stack([
        transform_intrinsic_for_resize_center_crop(
            value,
            src_size=(int(record["height"]), int(record["width"])),
            dst_size=(int(row["height"]), int(row["width"])),
        ) for value in normalized
    ]).astype(np.float32)


def repair_directory(input_dir: Path, manifest_path: Path, *, selected_ids_path: Path | None,
                     expected_rows: int) -> dict:
    records = _manifest(manifest_path)
    selected = None
    if selected_ids_path is not None:
        ids = [json.loads(line)["parquet_id"] for line in selected_ids_path.read_text(encoding="utf-8").splitlines()
               if line.strip()]
        if len(ids) != len(set(ids)) or not ids:
            raise ValueError("selected Parquet IDs must be nonempty and unique")
        selected = set(ids)
    paths = sorted(input_dir.rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no parquet found under {input_dir}")
    chosen: set[str] = set()
    for path in paths:
        for identity in pq.read_table(path, columns=["id"])["id"].to_pylist():
            if identity in records and (selected is None or identity in selected):
                if identity in chosen:
                    raise ValueError(f"duplicate selected ID: {identity}")
                chosen.add(identity)
    if len(chosen) != expected_rows or (selected is not None and chosen != selected):
        raise ValueError(f"selected {len(chosen)} rows, expected {expected_rows}; missing={sorted((selected or set()) - chosen)[:3]}")

    changed = 0
    already_correct = 0
    source_root = manifest_path.parent.parent
    for path in paths:
        if not any(identity in chosen for identity in pq.read_table(path, columns=["id"])["id"].to_pylist()):
            continue
        table = pq.read_table(path)
        rows = table.to_pylist()
        dirty = False
        for row in rows:
            identity = row["id"]
            if identity not in chosen:
                continue
            expected = _expected_intrinsics(row, records[identity], source_root)
            actual = np.frombuffer(row["camera_intrinsics_bytes"], dtype=np.float32).reshape(row["camera_intrinsics_shape"])
            if actual.shape != expected.shape or row["camera_intrinsics_dtype"] != "float32":
                raise ValueError(f"{identity}: invalid stored K dtype or shape")
            if np.allclose(actual, expected, atol=1.0e-5, rtol=1.0e-5):
                already_correct += 1
                continue
            if float(actual[..., 0, 0].max()) <= 10.0:
                raise ValueError(f"{identity}: normalized K differs from source-derived K; refusing overwrite")
            row["camera_intrinsics_bytes"] = expected.tobytes()
            changed += 1
            dirty = True
        if dirty:
            backup = path.with_suffix(path.suffix + ".pre_intrinsics_fix.bak")
            if not backup.exists():
                os.link(path, backup)
            temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
            try:
                pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), temporary, compression="zstd")
                if not pq.read_schema(temporary).equals(table.schema, check_metadata=False):
                    raise ValueError(f"schema changed during K repair: {path}")
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
    return {"rows": len(chosen), "changed": changed, "already_correct": already_correct,
            "input_dir": str(input_dir), "backup_suffix": ".pre_intrinsics_fix.bak"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--selected-ids-file", type=Path)
    parser.add_argument("--expected-rows", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(repair_directory(args.input_dir, args.manifest,
                                      selected_ids_path=args.selected_ids_file,
                                      expected_rows=args.expected_rows), indent=2), flush=True)


if __name__ == "__main__":
    main()
