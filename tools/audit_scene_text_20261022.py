"""Bounded CPU audit of every persisted scene-text training tensor."""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import pyarrow.parquet as pq


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(f'.tmp.{os.getpid()}')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(temporary, path)


def check_shard(path: Path, specs: dict[str, dict[str, Any]]) -> list[tuple[str, str]]:
    """Identical payload checks to the serial audit; no metadata-only shortcut."""
    # Sequential whole-file reads avoid many small NFS seeks. Each file is
    # one bounded row (~10 MB); no cross-file tensor accumulation occurs.
    rows = pq.ParquetFile(BytesIO(path.read_bytes())).read(use_threads=False).to_pylist()
    if len(rows) != 1:
        raise ValueError(f'Expected one persisted row: {path}')
    results = []
    for row in rows:
        spec = specs[row['id']]
        if row['caption'] != spec['caption']:
            raise ValueError(f'Mismatched caption: {row["id"]}')
        tags = np.frombuffer(row['text_token_tags_bytes'], dtype=row['text_token_tags_dtype'])
        camera = np.frombuffer(row['camera_to_world_bytes'], dtype=row['camera_to_world_dtype'])
        camera = camera.reshape(row['camera_to_world_shape'])
        if not np.allclose(camera[0], np.eye(4), atol=1e-4) or not (tags == 1).any() or not (tags == 0).any():
            raise ValueError(f'First-frame or actual text token audit failed: {row["id"]}')
        if len(tags) != row['text_embedding_shape'][0] or len(tags) > 1024:
            raise ValueError(f'Embedding/tag misalignment or truncation: {row["id"]}')
        if path.parents[1].name != spec['split']:
            raise ValueError(f'Train/validation split mismatch: {row["id"]}')
        metadata = json.loads(path.with_suffix('.audit.json').read_text())
        if metadata['id'] != row['id'] or metadata['caption_sha256'] != spec['caption_sha256']:
            raise ValueError(f'Caption provenance mismatch: {row["id"]}')
        if hashlib.sha256(row['caption'].encode()).hexdigest() != spec['caption_sha256']:
            raise ValueError(f'Actual caption hash mismatch: {row["id"]}')
        if spec.get('prompt_composition') == 'background_plus_subject_exact_sentence_dedup_v1':
            from tools.prepare_scene_subject_text_20261024 import compose_caption
            if (row['caption'] != compose_caption(spec['background_description'], spec['subject_description'])
                    or hashlib.sha256(spec['background_description'].encode()).hexdigest()
                    != spec['component_background_sha256'] or hashlib.sha256(
                        spec['subject_description'].encode()).hexdigest() != spec['component_subject_sha256']
                    or any(
                        metadata.get(key) != spec[key]
                        for key in ('prompt_composition', 'component_background_sha256', 'component_subject_sha256'))):
                raise ValueError(f'Background/subject composition provenance mismatch: {row["id"]}')
        if spec.get('caption_policy') == 'exact_window_dual_subject_motion_background_only_v1' and (
                metadata.get('caption_window') != spec['caption_window'] or metadata.get('video') != spec['video']
                or metadata.get('source_frame_indices') != spec['source_frame_indices']
                or metadata.get('registry_sha256') != spec['registry_sha256']
                or Path(metadata['source_parquet']).resolve() != Path(spec['source_parquet']).resolve()
                or metadata.get('source_row') != spec['source_row']
                or row['source_frame_indices'] != spec['source_frame_indices']):
            raise ValueError(f'Exact union video/window/source provenance mismatch: {row["id"]}')
        for stem, expected_shape, audit_key in (
            ('vae_latent', (24, 37, 30, 52), 'target_latent_sha256'),
            ('first_frame_latent', (24, 1, 30, 52), 'first_frame_latent_sha256'),
            ('text_embedding', tuple(row['text_embedding_shape']), 'embedding_sha256'),
            ('camera_to_world', (37, 4, 4), 'camera_sha256'),
            ('camera_intrinsics', tuple(row['camera_intrinsics_shape']), 'intrinsics_sha256'),
        ):
            shape = tuple(row[stem + '_shape'])
            array = np.frombuffer(row[stem + '_bytes'], dtype=row[stem + '_dtype']).reshape(shape)
            if shape != expected_shape or not np.isfinite(array).all():
                raise ValueError(f'Bad native tensor {stem}: {row["id"]}')
            if hashlib.sha256(row[stem + '_bytes']).hexdigest() != metadata[audit_key]:
                raise ValueError(f'Producer/persisted tensor hash mismatch {stem}: {row["id"]}')
        if row['text_embedding_shape'][1] != 5120 or not np.isin(tags, (0, 1)).all():
            raise ValueError(f'Wrong encoder width or semantic tags: {row["id"]}')
        results.append((row['id'], spec['split']))
    return results


def audit_root(root: Path, workers: int = 8) -> None:
    if workers < 1 or workers > 16:
        raise ValueError('Use 1–16 bounded CPU audit readers')
    specs = {}
    with (root / 'text_data_plan.jsonl').open() as handle:
        for line in handle:
            if not line.strip():
                continue
            spec = json.loads(line)
            if spec['id'] in specs:
                raise ValueError(f'Duplicate planned ID: {spec["id"]}')
            specs[spec['id']] = spec
    paths = sorted((root / 'preprocessed').rglob('*.parquet'))
    if len(paths) != len(specs):
        raise ValueError(f'Expected {len(specs)} one-row files, found {len(paths)}')
    seen: set[str] = set()
    counts: Counter[str] = Counter()
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, results in enumerate(pool.map(partial(check_shard, specs=specs), paths), start=1):
            for sample_id, split in results:
                if sample_id in seen:
                    raise ValueError(f'Duplicate persisted ID: {sample_id}')
                seen.add(sample_id)
                counts[split] += 1
            if index % 500 == 0 or index == len(paths):
                elapsed = time.monotonic() - started
                progress = dict(checked=len(seen),
                                expected=len(specs),
                                workers=workers,
                                elapsed_seconds=elapsed,
                                rows_per_second=index / max(elapsed, .001),
                                estimated_remaining_seconds=(len(paths) - index) * elapsed / index,
                                split_counts=dict(counts))
                atomic_json(root / 'text_data_audit_progress.json', progress)
                print('TEXT_AUDIT_PROGRESS', json.dumps(progress), flush=True)
    if seen != set(specs):
        raise ValueError(f'Missing/unexpected rows: expected {len(specs)}, got {len(seen)}')
    atomic_json(
        root / 'text_data_audit.json',
        dict(passed=True,
             rows=len(seen),
             split_counts=dict(counts),
             native_h3_grid=[24, 37, 30, 52],
             text_tokens_present=True,
             camera_first_frame_identity=True,
             caption_sha256_validated=True,
             plan_sha256=hashlib.sha256((root / 'text_data_plan.jsonl').read_bytes()).hexdigest(),
             full_persisted_tensor_finiteness_and_sha256=True,
             sources_read_only=True,
             audit_workers=workers,
             elapsed_seconds=time.monotonic() - started))
    (root / 'TEXT_DATA_VERIFIED').touch()
    print('TEXT_DATA_VERIFIED', dict(counts), flush=True)
