#!/usr/bin/env python3
"""Freeze close-first-frame camera pairs without changing source H3 caches.

Only derived rows are written. A scene/speed/prompt group shares one audited
first-frame and text condition; target video and measured rotations stay intact.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from fastvideo.models.dits.minimax_h3_prope import relative_camera_actions


DIRECTIONS = "WASDIJKL"
SOURCE_VIDEO = Path("/home/lizihao/projects/worldmodel/vggt-omega/h3_js_easy/h3_i2v_js_easy_124/videos")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def first_video_frame(global_id: str) -> np.ndarray | None:
    name = global_id.split(":", 1)[1] + ".mp4"
    capture = cv2.VideoCapture(str(SOURCE_VIDEO / name))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        return None
    return cv2.resize(frame, (832, 480), interpolation=cv2.INTER_AREA)


def unpack(row: dict, stem: str) -> np.ndarray:
    return np.frombuffer(row[f"{stem}_bytes"], dtype=row[f"{stem}_dtype"]).reshape(row[f"{stem}_shape"]).copy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-mapping", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-first-frame-mae", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=20260930)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"Refusing to overwrite frozen output: {args.output}")

    manifest = read_jsonl(args.source_root / "ids.jsonl")
    by_id = {entry["id"]: entry for entry in manifest}
    source_map = {entry["global_id"]: entry for entry in read_jsonl(args.source_mapping)}
    rows = {}
    parquet_paths = sorted((args.source_root / "data").glob("*.parquet"))
    for path in parquet_paths:
        table = pq.read_table(path)
        for row in table.to_pylist():
            if row["id"] in by_id:
                rows[row["id"]] = row
    if len(rows) != len(manifest):
        raise SystemExit(f"Missing conditioned rows: {len(rows)}/{len(manifest)}")

    groups = defaultdict(list)
    decoded = {}
    for entry in manifest:
        source = source_map[entry["global_id"]]
        image = first_video_frame(entry["global_id"])
        if image is None:
            continue
        identity = entry["id"]
        decoded[identity] = image
        mode = entry["global_id"].rsplit("_", 2)[-2:]
        groups[(entry["scene"], entry["speed"], tuple(mode))].append(entry)

    selected = {}
    rejected = []
    pair_groups = []
    for key, entries in sorted(groups.items()):
        # Choose the real source frame with the largest close-first-frame set.
        candidates = []
        for anchor in entries:
            first = decoded[anchor["id"]].astype(np.int16)
            accepted = []
            for item in entries:
                mae = float(np.abs(first - decoded[item["id"]].astype(np.int16)).mean())
                if mae <= args.max_first_frame_mae:
                    accepted.append((item, mae))
            candidates.append((len(accepted), -sum(mae for _, mae in accepted), anchor, accepted))
        _, _, anchor, accepted = max(candidates, key=lambda item: (item[0], item[1]))
        if len({item["direction"] for item, _ in accepted}) < 2:
            rejected.extend(item["global_id"] for item in entries)
            continue
        anchor_row = rows[anchor["id"]]
        shared_fields = ("first_frame_latent", "text_embedding", "text_token_tags")
        for entry, mae in accepted:
            row = rows[entry["id"]].copy()
            for field in shared_fields:
                for suffix in ("bytes", "shape", "dtype"):
                    row[f"{field}_{suffix}"] = anchor_row[f"{field}_{suffix}"]
            camera = unpack(row, "camera_to_world")
            if camera.shape != (37, 4, 4) or not np.allclose(camera[0], np.eye(4), atol=1e-3):
                raise SystemExit(f"Bad first pose: {entry['id']}")
            source = source_map[entry["global_id"]]
            # Old preprocessing normalized each translation path to ~1. Restore
            # the measured fast/slow ratio isotropically, never per-axis. For
            # rotations the recorded SO(3) path is already measured and kept.
            factor = float(source["actions"][entry["direction"]].get("speed_scale_factor", 1.0))
            if not np.isfinite(factor) or factor <= 0 or factor > 10:
                raise SystemExit(f"Invalid source speed factor: {entry['id']}={factor}")
            if entry["direction"] in "WASD":
                camera[:, :3, 3] *= factor
            row["camera_to_world_bytes"] = camera.tobytes()
            # Keep the auxiliary incremental action field consistent with the
            # derived C2W; the hybrid adapter itself uses cumulative poses.
            from fastvideo.dataset.vwm_h3 import build_local_camera_actions
            local = build_local_camera_actions(camera).astype(np.float32)
            row["action_continuous_bytes"] = local.tobytes()
            row["action_continuous_shape"] = list(local.shape)
            selected[entry["id"]] = (row, {
                **entry, "shared_first_frame_id": anchor["id"],
                "first_frame_pixel_mae": mae,
                "isotropic_speed_factor": factor if entry["direction"] in "WASD" else 1.0,
            })
        rejected.extend(item["global_id"] for item in entries if item["id"] not in selected)
        pair_groups.append({"key": key, "anchor": anchor["global_id"], "count": len(accepted),
                            "directions": sorted({item["direction"] for item, _ in accepted})})

    if len(selected) < 80:
        raise SystemExit(f"Too few strictly paired rows: {len(selected)}")
    ordered = sorted(selected)
    schema = pq.read_schema(parquet_paths[0])
    args.output.mkdir(parents=True)
    data_dir = args.output / "data"
    data_dir.mkdir()
    for start in range(0, len(ordered), 16):
        chunk = [selected[identity][0] for identity in ordered[start:start + 16]]
        pq.write_table(pa.Table.from_pylist(chunk, schema=schema), data_dir / f"data_{start // 16:04d}.parquet")

    by_direction = defaultdict(list)
    new_manifest = []
    all_actions = []
    for index, identity in enumerate(ordered):
        row, entry = selected[identity]
        entry["parquet_index"] = index
        new_manifest.append(entry)
        by_direction[entry["direction"]].append(index)
        pose = torch.from_numpy(unpack(row, "camera_to_world")).unsqueeze(0)
        all_actions.append(relative_camera_actions(pose, normalize_translation=False)[0].numpy())
    if any(not by_direction[direction] for direction in DIRECTIONS):
        raise SystemExit("All eight directions require at least one sample")
    actions = np.concatenate(all_actions, axis=0)
    # Scale q99 to about 3 of a 5-unit clip, leaving outlier headroom.
    scales = np.maximum(np.quantile(np.abs(actions), 0.99, axis=0) / 3.0, 0.01)
    clipping_rate = float((np.abs(actions / scales) > 5).any(axis=-1).mean())
    if clipping_rate > 0.02:
        raise SystemExit(f"Action clipping too frequent after calibration: {clipping_rate:.2%}")
    rng = random.Random(args.seed)
    target = max(len(by_direction[direction]) for direction in DIRECTIONS)
    balanced = []
    for direction in DIRECTIONS:
        indices = by_direction[direction].copy()
        rng.shuffle(indices)
        balanced.extend(indices[i % len(indices)] for i in range(target))
    rng.shuffle(balanced)
    report = {
        "unique_samples": len(ordered),
        "unique_direction_counts": dict(Counter(item["direction"] for item in new_manifest)),
        "balanced_draws_total": len(balanced),
        "balanced_draws_per_direction": target,
        "paired_group_count": len(pair_groups),
        "decoded_first_frames": len(decoded),
        "first_frame_max_mae": args.max_first_frame_mae,
        "action_channel_scales": scales.tolist(),
        "action_clipping_rate": clipping_rate,
        "source_root": str(args.source_root),
        "derivation": "shared audited first-frame/text per close-scene group; isotropic measured translation speed factor for WASD; original video latents and SO(3) retained",
    }
    (args.output / "ids.jsonl").write_text("".join(json.dumps(item) + "\n" for item in new_manifest))
    (args.output / "rejected_ids.jsonl").write_text("".join(json.dumps(item) + "\n" for item in sorted(set(rejected))))
    (args.output / "pair_groups.json").write_text(json.dumps(pair_groups, indent=2) + "\n")
    (args.output / "train_indices.json").write_text(json.dumps(balanced) + "\n")
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
