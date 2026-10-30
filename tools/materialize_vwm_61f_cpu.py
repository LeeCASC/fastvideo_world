#!/usr/bin/env python3
"""Materialize first-frame PNG and 61-frame rebased pose archives.

Consumes only CPU ``provisional_pass`` rows from the 61f audit. It does not
encode VAE latents or generate VLM text.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np


def safe_id(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def sample_array(array: np.ndarray, source_indices: list[int], source_frames: int) -> np.ndarray:
    if len(array) == source_frames:
        return np.asarray(array)[np.asarray(source_indices, dtype=np.int64)]
    # Some legacy caches store a temporal latent/anchor path rather than one
    # matrix per video frame. Keep deterministic temporal alignment.
    positions = np.rint(np.linspace(0, len(array) - 1, len(source_indices))).astype(np.int64)
    return np.asarray(array)[positions]


def materialize(row: dict, root: Path) -> dict:
    key = safe_id(row["window_id"])
    first_path = root / "first_frames" / f"{key}.png"
    pose_path = root / "poses" / f"{key}.npz"
    first_path.parent.mkdir(parents=True, exist_ok=True)
    pose_path.parent.mkdir(parents=True, exist_ok=True)
    if not first_path.is_file():
        cap = cv2.VideoCapture(row["video"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(row["source_frame_indices"][0]))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise ValueError("first_frame_decode_failed")
        tmp = first_path.with_suffix(f".tmp.{os.getpid()}.png")
        if not cv2.imwrite(str(tmp), frame):
            raise ValueError("first_frame_write_failed")
        os.replace(tmp, first_path)
    with np.load(row["camera"], allow_pickle=False) as payload:
        key_name = "camera_to_world" if "camera_to_world" in payload else "c2w"
        c2w = sample_array(np.asarray(payload[key_name], dtype=np.float32),
                           row["source_frame_indices"], int(row["source_frames"]))
        intrinsics = np.asarray(payload["intrinsics"], dtype=np.float32) if "intrinsics" in payload else None
        if intrinsics is not None:
            intrinsics = sample_array(intrinsics, row["source_frame_indices"], int(row["source_frames"]))
    rebased = np.linalg.inv(c2w[0].astype(np.float64)) @ c2w.astype(np.float64)
    rebased = rebased.astype(np.float32)
    if not np.allclose(rebased[0], np.eye(4, dtype=np.float32), atol=1e-4):
        raise ValueError("first_frame_rebase_failed")
    tmp = pose_path.with_suffix(f".tmp.{os.getpid()}.npz")
    kwargs = {"camera_to_world": rebased, "source_frame_indices": np.asarray(row["source_frame_indices"], dtype=np.int64)}
    if intrinsics is not None:
        kwargs["intrinsics"] = intrinsics
    np.savez_compressed(tmp, **kwargs)
    os.replace(tmp, pose_path)
    output = dict(row)
    output.update({
        "first_frame": str(first_path),
        "pose_61f": str(pose_path),
        "quality_status": "provisional_pass",
        "pose_status": "materialized_rebased",
        "prompt_status": "pending_vlm",
        "vae_status": "not_encoded",
    })
    return output


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", type=Path, required=True)
    ap.add_argument("--registry", type=Path, required=True)
    ap.add_argument("--root", type=Path, required=True)
    args = ap.parse_args()
    rows = {json.loads(line)["window_id"]: json.loads(line) for line in args.registry.open(encoding="utf-8") if line.strip()}
    args.root.mkdir(parents=True, exist_ok=True)
    accepted = []
    rejected = []
    for line in args.audit.open(encoding="utf-8"):
        if not line.strip():
            continue
        audit = json.loads(line)
        row = rows.get(audit["window_id"])
        if row is None:
            continue
        if audit["status"] != "provisional_pass":
            rejected.append({"window_id": audit["window_id"], "status": audit["status"], "reasons": audit.get("hard_rejection_reasons", []) + audit.get("review_reasons", [])})
            continue
        try:
            accepted.append(materialize(row, args.root))
        except Exception as exc:
            rejected.append({"window_id": audit["window_id"], "status": "materialize_failed", "reasons": [str(exc)]})
        if (len(accepted) + len(rejected)) % 1000 == 0:
            print(f"materialized={len(accepted)} rejected={len(rejected)}", flush=True)
    reg = args.root / "registry"
    reg.mkdir(exist_ok=True)
    with (reg / "windows_61_ready_for_vlm.jsonl").open("w", encoding="utf-8") as handle:
        for row in accepted:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (reg / "windows_61_rejected_after_cpu.jsonl").open("w", encoding="utf-8") as handle:
        for row in rejected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.root / "reports" / "materialize_summary.json").parent.mkdir(exist_ok=True)
    (args.root / "reports" / "materialize_summary.json").write_text(json.dumps({
        "ready_for_vlm": len(accepted), "rejected": len(rejected),
        "prompt_status": "pending_vlm", "vae_status": "not_encoded",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"complete ready_for_vlm={len(accepted)} rejected={len(rejected)}", flush=True)


if __name__ == "__main__":
    main()
