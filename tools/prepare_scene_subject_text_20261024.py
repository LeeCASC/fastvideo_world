"""Freeze explicit background + subject captions without rerunning VLM or VAE."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
from tools.prepare_union_dual_captions_20261023 import (
    POLICY,
    RECOVERY_VERSION,
    atomic_json,
    background_violation,
    camera_text,
    digest,
    file_digest,
    read_jsonl,
    selection_hash,
    verify_captions,
    write_jsonl,
)

COMPOSITION = 'background_plus_subject_exact_sentence_dedup_v1'
SOURCE = Path('/datacc05/lizihao/worldmodel/fastvideowm_union_dual_text_20261023')
OUTPUT = Path('/datacc05/lizihao/worldmodel/fastvideowm_union_scene_subject_text_20261024')


def compose_caption(background: str, subject: str) -> str:
    """Background first; remove only exact sentence duplicates, never infer facts."""
    if camera_text(background) or camera_text(subject) or background_violation(background):
        raise ValueError('Invalid camera/foreground language in caption components')
    sentences, seen = [], set()
    for text in (background, subject):
        for sentence in re.split(r'(?<=[.!?])\s+', text.strip()):
            key = re.sub(r'\s+', ' ', sentence).strip().casefold().rstrip('.!?')
            if key and key not in seen:
                # Preserve original words; normalize whitespace only.
                sentences.append(re.sub(r'\s+', ' ', sentence).strip())
                seen.add(key)
    caption = ' '.join(sentences)
    if not 5 <= len(caption.split()) <= 150:
        raise ValueError('Empty or oversized composed caption')
    return caption


def existing_cache(source: Path, index: int, subject: dict, caption: str) -> tuple[str | None, str]:
    """Reuse is eligible only for an identical caption and exact-window row."""
    partial = source / 'subject_motion/preprocessed' / subject['split'] / (
        f'worker_{index % 8:02d}') / f'data_{index // 8:06d}.parquet'
    candidates = [(subject['caption'], partial)]
    if subject.get('existing_encoded'):
        candidates.append((subject.get('existing_caption', ''), Path(subject['existing_encoded'])))
    for old_caption, path in candidates:
        if caption == old_caption and path.is_file() and path.with_suffix('.audit.json').is_file():
            return str(path), old_caption
    return None, ''


def plan(root: Path, source: Path) -> dict:
    if root.resolve() == source.resolve():
        raise ValueError('Never rewrite the original caption/condition directory')
    verify_captions(argparse.Namespace(root=source))
    proof_path = root / 'SCENE_SUBJECT_COMPOSITION_VERIFIED'
    if proof_path.exists():
        proof = json.loads(proof_path.read_text())
        if (proof.get('source_caption_gate_sha256') != file_digest(source / 'CAPTIONS_VERIFIED')
                or proof.get('composition') != COMPOSITION):
            raise ValueError('Frozen composition source changed')
        verify_captions(argparse.Namespace(root=root))
        return proof
    if any((root / 'subject_motion/preprocessed').rglob('*.parquet')):
        raise ValueError('Refusing to replace an existing conditioned dataset without frozen plans')
    subjects = list(read_jsonl(source / 'subject_motion/text_data_plan.jsonl'))
    backgrounds = list(read_jsonl(source / 'background_only/text_data_plan.jsonl'))
    if len({s['id'] for s in subjects}) != len(subjects):
        raise ValueError('Duplicate source ID')
    hashes = {
        variant: file_digest(source / variant / 'text_data_plan.jsonl')
        for variant in ('subject_motion', 'background_only')
    }
    joined: list[dict] = []
    background_plan: list[dict] = []
    examples: list[dict] = []
    stats: Counter[str] = Counter()
    for index, (subject, background) in enumerate(zip(subjects, backgrounds, strict=True)):
        for key in ('id', 'video', 'window_id', 'source_frame_indices', 'source_parquet', 'source_row', 'split',
                    'registry_sha256'):
            if subject[key] != background[key]:
                raise ValueError(f'Caption components have different source/window: {subject["id"]} {key}')
        caption = compose_caption(background['caption'], subject['caption'])
        cache, old_caption = existing_cache(source, index, subject, caption)
        row = dict(subject,
                   caption=caption,
                   caption_sha256=digest(caption),
                   background_description=background['caption'],
                   subject_description=subject['caption'],
                   component_background_sha256=background['caption_sha256'],
                   component_subject_sha256=subject['caption_sha256'],
                   prompt_composition=COMPOSITION,
                   existing_encoded=cache,
                   existing_caption=old_caption)
        joined.append(row)
        background_plan.append(dict(background, prompt_composition='background_only_deferred'))
        stats['rows'] += 1
        stats['eligible_existing_cache'] += cache is not None
        stats['caption_changed_from_subject_only'] += caption != subject['caption']
        stats['composed_equals_background'] += caption == background['caption']
        if len(examples) < 5:
            examples.append({
                key: row[key]
                for key in ('id', 'video', 'background_description', 'subject_description', 'caption', 'caption_sha256')
            })
    root.mkdir(parents=True, exist_ok=True)
    for name in ('registry.jsonl', 'caption_selection.json', 'inventory_summary.json'):
        src, dst = source / name, root / name
        if not src.exists():
            if name == 'caption_selection.json':
                continue
            raise ValueError(f'Missing original source: {src}')
        if dst.exists():
            if dst.resolve() != src.resolve():
                raise ValueError(f'Unexpected existing source link: {dst}')
        else:
            dst.symlink_to(src)
    plan_hashes = {}
    registry_hash = file_digest(root / 'registry.jsonl')
    for variant, rows in (('subject_motion', joined), ('background_only', background_plan)):
        write_jsonl(root / variant / 'text_data_plan.jsonl', rows)
        plan_hashes[variant] = file_digest(root / variant / 'text_data_plan.jsonl')
        atomic_json(
            root / variant / 'data_plan_summary.json',
            dict(total=len(rows),
                 split_counts=dict(Counter(row['split'] for row in rows)),
                 source_counts=dict(Counter(row['source_name'] for row in rows)),
                 variant=variant,
                 prompt_composition=COMPOSITION if variant == 'subject_motion' else 'background_only_deferred',
                 registry_sha256=registry_hash,
                 selection_sha256=selection_hash(root),
                 plan_sha256=plan_hashes[variant],
                 source_caption_plan_sha256=hashes,
                 vae_reencoding=False,
                 background_encoding_deferred=True))
    proof = dict(composition=COMPOSITION,
                 rows=len(joined),
                 stats=dict(stats),
                 source_root=str(source),
                 source_caption_gate_sha256=file_digest(source / 'CAPTIONS_VERIFIED'),
                 source_caption_plan_sha256=hashes,
                 plan_sha256=plan_hashes,
                 background_encoding_deferred=True,
                 original_data_unchanged=True)
    atomic_json(proof_path, proof)
    atomic_json(root / 'composed_caption_examples_5.json', {'examples': examples})
    atomic_json(
        root / 'CAPTIONS_VERIFIED',
        dict(rows=len(joined),
             policy=POLICY,
             recovery_version=RECOVERY_VERSION,
             registry_sha256=registry_hash,
             selection_sha256=selection_hash(root),
             plan_sha256=plan_hashes,
             prompt_composition=COMPOSITION,
             source_caption_gate_sha256=proof['source_caption_gate_sha256']))
    verify_captions(argparse.Namespace(root=root))
    print('COMPOSED_CAPTION_PLANS_READY', json.dumps(proof), flush=True)
    return proof


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=SOURCE)
    parser.add_argument('--root', type=Path, default=OUTPUT)
    arguments = parser.parse_args()
    plan(arguments.root, arguments.source_root)
