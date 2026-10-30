#!/usr/bin/env python3
"""Assemble the audited 61f core subset into a portable transfer directory."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def classify(qc: dict, vlm: dict | None) -> tuple[str, list[str]]:
    assessment = qc.get("assessment") or {}
    reasons: list[str] = []
    if qc.get("error"):
        reasons.append("gpu_qc_error")
    if vlm is None or vlm.get("error"):
        reasons.append("vlm_error_or_missing")
    for key in ("blur", "texture_breakdown", "subject_deformation", "translation_jitter"):
        if assessment.get(key) is True:
            reasons.append(key)
    if assessment.get("visible_camera_motion_match") == "no":
        reasons.append("camera_motion_mismatch")
    if reasons:
        return "filtered_out", reasons
    if assessment.get("uncertain") is True or assessment.get("visible_camera_motion_match") == "uncertain":
        return "review", ["camera_visual_agreement_uncertain"]
    return "pass", []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--qc-dir", type=Path, required=True)
    parser.add_argument("--vlm-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skill-dir", type=Path, required=True)
    args = parser.parse_args()

    registry = {row["global_id"]: row for row in read_jsonl(args.registry)}
    qc = {}
    for path in sorted(args.qc_dir.glob("results_shard_*.jsonl")):
        for row in read_jsonl(path):
            qc[row["global_id"]] = row
    vlm = {}
    for path in sorted(args.vlm_dir.glob("captions_shard_*.jsonl")):
        for row in read_jsonl(path):
            vlm[row["global_id"]] = row
    if set(registry) != set(qc) or set(registry) != set(vlm):
        raise RuntimeError(
            f"ID mismatch registry={len(registry)} qc={len(qc)} vlm={len(vlm)}"
        )

    output = args.output
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing package directory: {output}")
    output.mkdir(parents=True)
    for sub in ("registry", "qc", "vlm", "assets/first_frames", "assets/poses", "skill"):
        (output / sub).mkdir(parents=True, exist_ok=True)

    merged = []
    groups: dict[str, list[dict]] = {"pass": [], "review": [], "filtered_out": []}
    for gid, row in sorted(registry.items()):
        status, reasons = classify(qc[gid], vlm[gid])
        merged_row = dict(row)
        merged_row["final_quality_status"] = status
        merged_row["final_quality_reasons"] = reasons
        merged_row["gpu_qc"] = qc[gid].get("assessment")
        merged_row["vlm_scene_description"] = vlm[gid].get("scene_description")
        merged_row["vlm_confidence"] = vlm[gid].get("confidence")
        merged.append(merged_row)
        groups[status].append(merged_row)

    write_jsonl(output / "registry/merged_qc_vlm.jsonl", merged)
    for name, rows in groups.items():
        write_jsonl(output / f"registry/{name}.jsonl", rows)

    # Preserve the raw audit shards so the receiving server can re-audit.
    for source_dir, target_dir, pattern in (
        (args.qc_dir, output / "qc", "results_shard_*.jsonl"),
        (args.vlm_dir, output / "vlm", "captions_shard_*.jsonl"),
    ):
        for path in source_dir.glob(pattern):
            shutil.copy2(path, target_dir / path.name)

    # Only strict-pass derived media is copied; source videos remain external.
    copied_frames = copied_poses = 0
    source_rows = []
    for row in groups["pass"]:
        frame = Path(row["first_frame"])
        pose = Path(row["pose_61f"])
        target_frame = output / "assets/first_frames" / frame.name
        target_pose = output / "assets/poses" / pose.name
        shutil.copy2(frame, target_frame)
        shutil.copy2(pose, target_pose)
        copied_frames += target_frame.stat().st_size
        copied_poses += target_pose.stat().st_size
        source_rows.append({
            "window_id": row["window_id"],
            "global_id": row["global_id"],
            "video": row["video"],
            "video_bytes": Path(row["video"]).stat().st_size,
            "camera": row["camera"],
            "source_frame_indices": row["source_frame_indices"],
        })
    write_jsonl(output / "source_videos_manifest.jsonl", source_rows)

    # Copy the skill, including its references, into the transfer package.
    shutil.copytree(args.skill_dir, output / "skill/vwm-61f-minimax-h3-vae", dirs_exist_ok=True)
    (output / "README_zh.md").write_text(
        "# FastVideoWM 61 帧核心过滤数据包\n\n"
        "本包包含 GPU 质检与 VLM 描述后的核心子集。原始视频没有复制，\n"
        "`source_videos_manifest.jsonl` 保存了原视频路径、大小和精确帧索引。\n\n"
        "## 目录\n\n"
        "- `registry/pass.jsonl`：严格通过，可直接进入后续 VAE 编码。\n"
        "- `registry/review.jsonl`：未发现明确坏样本，但相机-画面一致性不确定。\n"
        "- `registry/filtered_out.jsonl`：存在模糊、抖动、姿态不匹配或处理错误。\n"
        "- `registry/merged_qc_vlm.jsonl`：每条样本合并后的完整元数据。\n"
        "- `assets/first_frames/`、`assets/poses/`：仅严格通过样本的派生文件。\n"
        "- `skill/vwm-61f-minimax-h3-vae/`：61 帧 MiniMax-H3 VAE 编码 skill。\n\n"
        "GPU-QC 是自动审计，不等同于人工质量批准；review 样本需要额外复核。\n"
        "接收服务器需要按 `source_videos_manifest.jsonl` 映射视频路径，或另行复制原始视频。\n",
        encoding="utf-8",
    )
    summary = {
        "registry_rows": len(registry),
        "pass_rows": len(groups["pass"]),
        "review_rows": len(groups["review"]),
        "filtered_out_rows": len(groups["filtered_out"]),
        "copied_first_frame_bytes": copied_frames,
        "copied_pose_bytes": copied_poses,
        "raw_videos_in_tar": False,
        "source_video_manifest": "source_videos_manifest.jsonl",
        "skill": "skill/vwm-61f-minimax-h3-vae",
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
