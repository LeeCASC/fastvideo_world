#!/usr/bin/env python3
"""Freeze a scene-disjoint, multi-action sampling order over the verified H3 cache.

The QC decision is automatic, not a claim of human quality approval. Source
videos, poses, and cached latents are read-only. The output indices refer to
the existing sorted conditioned Parquet shards.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random

import pyarrow.parquet as pq


KEYS = "WASDIJKL"
SPEEDS = ("slow", "medium", "fast")


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditioned-dir", type=Path, required=True)
    parser.add_argument("--source-mapping", type=Path, required=True)
    parser.add_argument("--train-indices", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target", type=int, default=12000)
    parser.add_argument("--multicam-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20261002)
    args = parser.parse_args()
    if not 0 < args.multicam_fraction < 1 or args.target < 1000:
        raise ValueError("invalid target or multicam fraction")
    rng = random.Random(args.seed)
    mapping = {}
    for row in read_jsonl(args.source_mapping):
        source_id = f"{row['source_name']}/{row['sample_id']}"
        if source_id in mapping or row.get("gpu_qc_pass") is not True:
            raise ValueError(f"duplicate or non-QC-passing source: {source_id}")
        mapping[source_id] = row
    train_indices = set(json.loads(args.train_indices.read_text(encoding="utf-8")))
    candidates = []
    row_index = 0
    for shard in sorted(args.conditioned_dir.rglob("*.parquet")):
        for source_id in pq.read_table(shard, columns=["id"]).column("id").to_pylist():
            if row_index in train_indices:
                source = mapping[source_id]
                labels = set()
                for key, item in source["actions"].items():
                    if (key in KEYS and item.get("speed_bin") in SPEEDS
                            and float(item.get("axis_fraction", 0)) >= 0.35
                            and float(item.get("direction_consistency", 0)) >= 0.65):
                        labels.add((key, item["speed_bin"]))
                if labels:
                    candidates.append({
                        "index": row_index,
                        "id": source["global_id"],
                        "source": source["source_name"],
                        "labels": labels,
                        "multicam": source["source_name"] == "train_v2_multicam_video",
                    })
            row_index += 1
    if row_index != len(mapping) or len(train_indices) < args.target:
        raise ValueError(f"unexpected conditioned/train counts: {row_index}, {len(train_indices)}")
    nonmulti = [row for row in candidates if not row["multicam"]]
    multicam = [row for row in candidates if row["multicam"]]
    requested_nonmulti = args.target - int(args.target * args.multicam_fraction)
    if len(nonmulti) < requested_nonmulti:
        raise ValueError(f"only {len(nonmulti)} non-multicam high-confidence rows, need {requested_nonmulti}")
    if len(multicam) < args.target - requested_nonmulti:
        raise ValueError("not enough high-confidence multicam rows")

    required = {(key, speed) for key in KEYS for speed in SPEEDS}
    if {label for row in candidates for label in row["labels"]} != required:
        raise ValueError("the available data do not cover all 24 direction/speed buckets")

    def select(pool: list[dict], target: int, initial_counts: Counter | None = None) -> list[dict]:
        buckets = defaultdict(list)
        for row in pool:
            for label in row["labels"]:
                buckets[label].append(row)
        for values in buckets.values():
            rng.shuffle(values)
        cursor = {label: 0 for label in buckets}
        selected = set()
        result = []
        counts = Counter(initial_counts or {})
        # A video may contribute several labels. Keep it intact rather than
        # pretending its whole 124-frame duration is one pure keypress.
        while len(result) < target:
            best = None
            for label in sorted(buckets, key=lambda value: (counts[value], value)):
                values = buckets[label]
                while cursor[label] < len(values):
                    row = values[cursor[label]]
                    cursor[label] += 1
                    if row["index"] not in selected:
                        best = row
                        break
                if best is not None:
                    break
            if best is None:
                raise ValueError(f"ran out of eligible rows at {len(result)}/{target}")
            selected.add(best["index"])
            result.append(best)
            counts.update(best["labels"])
        return result

    nonmulti_order = select(nonmulti, requested_nonmulti)
    nonmulti_counts = Counter(label for row in nonmulti_order for label in row["labels"])
    # Spend the limited multicam allowance primarily on directions that the
    # other source families cannot supply, especially camera pitch.
    multicam_order = select(multicam, args.target - requested_nonmulti, nonmulti_counts)
    order = []
    npos = mpos = 0
    while len(order) < args.target:
        # Interleave source families from the first training step onward.
        use_multi = (mpos < len(multicam_order) and
                     mpos < int((len(order) + 1) * args.multicam_fraction))
        if use_multi or npos == len(nonmulti_order):
            order.append(multicam_order[mpos])
            mpos += 1
        else:
            order.append(nonmulti_order[npos])
            npos += 1
    counts = Counter(label for row in order for label in row["labels"])
    sources = Counter(row["source"] for row in order)

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    (output / "train_indices.json").write_text(json.dumps([row["index"] for row in order]) + "\n")
    with (output / "train_ids.jsonl").open("w", encoding="utf-8") as handle:
        for row in order:
            handle.write(json.dumps({"index": row["index"], "global_id": row["id"],
                                     "source": row["source"], "labels": sorted(row["labels"])}) + "\n")
    direction = Counter()
    for (key, _speed), count in counts.items():
        direction[key] += count
    report = {
        "quality_gate": "strict automatic GPU-QC pass; human spot-check pending",
        "candidate_rows": len(candidates), "candidate_nonmulticam": len(nonmulti),
        "unique_train_rows": len(order), "multicam_fraction": sources["train_v2_multicam_video"] / len(order),
        "source_counts": dict(sorted(sources.items())),
        "direction_occurrences": {key: direction[key] for key in KEYS},
        "direction_speed_occurrences": {f"{key}/{speed}": counts[key, speed]
                                        for key in KEYS for speed in SPEEDS},
        "seed": args.seed,
        "selection_note": "A video may count in multiple directions; these are audited window labels, not frame counts.",
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
