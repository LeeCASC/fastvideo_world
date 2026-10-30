# SPDX-License-Identifier: Apache-2.0
"""Build a deterministic VWM manifest with optional per-source limits."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from typing import Sequence

from fastvideo.dataset.vwm_h3 import source_limited_manifest_indices
from vwm_datasets import VWMDataset


def _source_limit(value: str) -> tuple[str, int]:
    source, separator, raw_limit = value.partition("=")
    if not separator or not source or not raw_limit:
        raise argparse.ArgumentTypeError("source limits must use SOURCE=COUNT")
    try:
        limit = int(raw_limit)
    except ValueError as error:
        raise argparse.ArgumentTypeError("source limit COUNT must be an integer") from error
    if limit <= 0:
        raise argparse.ArgumentTypeError("source limit COUNT must be positive")
    return source, limit


def build_source_subset_manifest(
    manifest_path: Path,
    output_path: Path,
    *,
    source_limits: dict[str, int],
    seed: int,
) -> tuple[int, Counter[str], str]:
    """Write selected source rows atomically and return count, composition, and SHA256."""
    lines = manifest_path.read_text(encoding="utf-8").splitlines()
    dataset = VWMDataset(manifest_path)
    if len(lines) != len(dataset.records):
        raise ValueError("manifest line count does not match parsed VWM records")
    selected_indices = source_limited_manifest_indices(dataset.records, source_limits, seed)
    selected_lines = [lines[index] for index in selected_indices]
    composition = Counter(str(dataset.records[index].source_name) for index in selected_indices)

    payload = ("\n".join(selected_lines) + "\n").encode()
    digest = hashlib.sha256(payload).hexdigest()
    if output_path.is_file():
        if output_path.read_bytes() == payload:
            return len(selected_lines), composition, digest
        raise FileExistsError(f"refusing to replace a different existing manifest: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.unlink(missing_ok=True)
    temporary_path.write_bytes(payload)
    os.replace(temporary_path, output_path)
    return len(selected_lines), composition, digest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-limit", action="append", type=_source_limit, default=[])
    parser.add_argument("--seed", type=int, default=20260909)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    limits = dict(args.source_limit)
    if len(limits) != len(args.source_limit):
        raise ValueError("each source may be limited only once")
    count, composition, digest = build_source_subset_manifest(
        args.manifest,
        args.output,
        source_limits=limits,
        seed=args.seed,
    )
    print(json.dumps({"rows": count, "sources": dict(sorted(composition.items())), "sha256": digest}))


if __name__ == "__main__":
    main()
