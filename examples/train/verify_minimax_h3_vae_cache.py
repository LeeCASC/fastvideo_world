#!/usr/bin/env python3
"""Read every H3 PRoPE VAE row and reject missing, corrupt, or misaligned data.

The first-frame PNG and keyframe latent are encoded from the same video tensor
in preprocess_vae. This audit checks their stored shapes/content and checks
that the first latent camera is the identity; it does not claim to invert a VAE.
"""

from __future__ import annotations

import argparse
from io import BytesIO
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
import pyarrow.parquet as pq

from fastvideo.dataset.dataloader.schema import pyarrow_schema_minimax_h3_worldmodel_prope_vae


def _array(row: dict, name: str, shape: tuple[int, ...]) -> np.ndarray:
    if row[f"{name}_dtype"] != "float32":
        raise ValueError(f"{row['id']}: {name} is not float32")
    actual_shape = tuple(row[f"{name}_shape"])
    if actual_shape != shape:
        raise ValueError(f"{row['id']}: {name} shape {actual_shape} != {shape}")
    raw = row[f"{name}_bytes"]
    if len(raw) != int(np.prod(shape)) * np.dtype(np.float32).itemsize:
        raise ValueError(f"{row['id']}: {name} byte count does not match shape")
    value = np.frombuffer(raw, dtype=np.float32).reshape(shape)
    if not np.isfinite(value).all():
        raise ValueError(f"{row['id']}: {name} contains NaN/Inf")
    return value


def _manifest(path: Path) -> dict[str, int]:
    result: dict[str, int] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            identity = f"{record['source_name']}/{record['sample_id']}"
            if identity in result:
                raise ValueError(f"duplicate manifest ID: {identity}")
            result[identity] = int(record["num_frames"])
    if not result:
        raise ValueError("empty expected manifest")
    return result


def _selected_ids(path: Path) -> set[str]:
    identities = [json.loads(line)["parquet_id"] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not identities or len(identities) != len(set(identities)):
        raise ValueError("selected IDs must be nonempty and unique")
    return set(identities)


def verify_cache(
    input_dir: Path,
    manifest_path: Path,
    *,
    require_complete: bool = True,
    max_shards: int | None = None,
    selected_ids_path: Path | None = None,
) -> dict:
    expected = _manifest(manifest_path)
    if selected_ids_path is not None:
        selected = _selected_ids(selected_ids_path)
        unknown = selected - expected.keys()
        if unknown:
            raise ValueError(f"selected IDs absent from manifest: {sorted(unknown)[:3]}")
        expected = {identity: expected[identity] for identity in selected}
    paths = sorted(input_dir.rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no VAE parquet under {input_dir}")
    if max_shards is not None:
        paths = paths[:max_shards]
    seen: set[str] = set()
    rows = 0
    max_first_camera_error = 0.0
    for index, path in enumerate(paths, 1):
        if not pq.read_schema(path).equals(pyarrow_schema_minimax_h3_worldmodel_prope_vae,
                                           check_metadata=False):
            raise ValueError(f"unexpected VAE schema: {path}")
        for row in pq.read_table(path, schema=pyarrow_schema_minimax_h3_worldmodel_prope_vae).to_pylist():
            identity = row["id"]
            if identity not in expected or identity in seen:
                raise ValueError(f"unknown or duplicate VAE ID: {identity}")
            seen.add(identity)
            rows += 1
            if (row["width"], row["height"], row["num_frames"]) != (832, 480, 124):
                raise ValueError(f"{identity}: unexpected video geometry")
            indices = np.asarray(row["source_frame_indices"], dtype=np.int64)
            if (indices.shape != (124,) or indices[0] < 0 or indices[-1] >= expected[identity]
                    or np.any(np.diff(indices) < 0)):
                raise ValueError(f"{identity}: video/pose source-frame plan is invalid")
            latent = _array(row, "vae_latent", (24, 37, 30, 52))
            keyframe = _array(row, "first_frame_latent", (24, 1, 30, 52))
            if not np.any(latent) or not np.any(keyframe):
                raise ValueError(f"{identity}: all-zero VAE or first-frame latent")
            camera = _array(row, "camera_to_world", (37, 4, 4))
            intrinsics = _array(row, "camera_intrinsics", (37, 3, 3))
            error = float(np.max(np.abs(camera[0] - np.eye(4, dtype=np.float32))))
            max_first_camera_error = max(max_first_camera_error, error)
            if error > 1.0e-4:
                raise ValueError(f"{identity}: first latent camera is not rebased (error={error})")
            if np.max(np.abs(camera[:, 3] - [0.0, 0.0, 0.0, 1.0])) > 1.0e-4:
                raise ValueError(f"{identity}: invalid homogeneous camera row")
            rotation = camera[:, :3, :3]
            if np.max(np.abs(np.swapaxes(rotation, 1, 2) @ rotation - np.eye(3))) > 0.05:
                raise ValueError(f"{identity}: non-rigid camera rotation")
            if np.any(intrinsics[:, 0, 0] <= 0) or np.any(intrinsics[:, 1, 1] <= 0):
                raise ValueError(f"{identity}: nonpositive camera focal length")
            if (np.max(intrinsics[:, 0, 0]) > 32.0 or np.max(intrinsics[:, 1, 1]) > 32.0
                    or np.max(np.abs(intrinsics[:, :2, 2])) > 2.0):
                raise ValueError(f"{identity}: camera intrinsics are pixel-valued, not normalized")
            _array(row, "action_continuous", (37, 6))
            with Image.open(BytesIO(row["first_frame_png"])) as image:
                image.load()
                if image.mode != "RGB" or image.size != (832, 480):
                    raise ValueError(f"{identity}: stored first-frame PNG is not RGB 832x480")
        if index % 64 == 0 or index == len(paths):
            print(f"validated {index}/{len(paths)} shards, {rows} rows", flush=True)
    missing = expected.keys() - seen
    if require_complete and missing:
        raise ValueError(f"VAE cache missing {len(missing)} manifest IDs; first={sorted(missing)[:3]}")
    return {
        "status": "complete" if not missing else "partial",
        "rows": rows,
        "expected_rows": len(expected),
        "shards": len(paths),
        "missing_rows": len(missing),
        "max_first_camera_identity_error": max_first_camera_error,
        "first_frame_contract": "PNG and keyframe latent are derived from the same decoded video[:,0]",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--max-shards", type=int)
    parser.add_argument("--selected-ids-file", type=Path)
    parser.add_argument("--expected-rows", type=int)
    args = parser.parse_args()
    report = verify_cache(args.input_dir, args.manifest,
                          require_complete=not args.allow_partial, max_shards=args.max_shards,
                          selected_ids_path=args.selected_ids_file)
    if args.expected_rows is not None and report["rows"] != args.expected_rows:
        raise ValueError(f"validated {report['rows']} rows, expected {args.expected_rows}")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_name(f"{args.report.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, args.report)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
