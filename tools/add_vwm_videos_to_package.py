#!/usr/bin/env python3
"""Add strict-pass source videos to an existing transfer package."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    video_root = args.package / "assets/videos"
    video_root.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in args.manifest.open(encoding="utf-8") if line.strip()]
    updated = []
    total_bytes = 0
    for row in rows:
        source = Path(row["video"])
        if not source.is_file():
            raise FileNotFoundError(source)
        target = video_root / f"{row['global_id'].replace(':', '__')}.mp4"
        if not target.exists():
            shutil.copy2(source, target)
        total_bytes += target.stat().st_size
        item = dict(row)
        item["package_video"] = str(Path("assets/videos") / target.name)
        updated.append(item)
    with (args.package / "source_videos_manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in updated:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary_path = args.package / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update({
        "raw_videos_in_tar": True,
        "copied_video_rows": len(updated),
        "copied_video_bytes": total_bytes,
        "copied_video_gib": total_bytes / 2**30,
    })
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    readme = args.package / "README_zh.md"
    with readme.open("a", encoding="utf-8") as handle:
        handle.write(
            "\n已包含严格通过样本的原始视频：`assets/videos/`。\n"
            f"视频数量：{len(updated)}，大小：{total_bytes / 2**30:.2f} GiB。\n"
            "review 和 filtered_out 样本的视频未复制。\n"
        )
    print(json.dumps({"videos": len(updated), "bytes": total_bytes, "gib": total_bytes / 2**30}, ensure_ascii=False))


if __name__ == "__main__":
    main()
