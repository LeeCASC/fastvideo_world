"""Bind audited 124f VLM captions to existing H3 VAE rows, then re-encode text."""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import re

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

BASE = Path('/datacc05/lizihao/worldmodel/vwm_unified_124_243_20261016/124f')
INPUT = BASE / 'vae_latents/minimax_h3_prope_union_input'
CAPTIONS = BASE / 'vlm_qwen3vl_strict_50599'
MODEL = Path('/datacc05/lizihao_wm/worldmodel/ckpts/MiniMax-H3')
POLICY = 'observed_scene_subject_motion_no_camera_v2'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(f'.tmp.{os.getpid()}')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(temp, path)


def read_jsonl(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def clean_caption(text):
    # The producer already applies this policy. Reject residual camera/audio
    # clauses here rather than teaching contradictory numeric/text controls.
    camera = re.compile(r'\b(camera|viewpoint|framing|perspective|pann?ing|tilt(?:s|ing)?|zoom(?:s|ing)?|tracking shot|dolly|synchronized audio|soundtrack)\b', re.I)
    sentences = re.split(r'(?<=[.!?])\s+', str(text).strip())
    return ' '.join(s for s in sentences if not camera.search(s)).strip()


def plan(args):
    registry = {r['global_id']: r for r in read_jsonl(CAPTIONS / 'registry_strict_50599.jsonl')}
    captions = {}
    for path in sorted(CAPTIONS.glob('gpu*/captions_shard_*.jsonl')):
        for row in read_jsonl(path):
            key = row['global_id']
            if key in captions and captions[key] != row:
                raise ValueError(f'Conflicting caption {key}')
            captions[key] = row
    selected, rejected = [], []
    seen = set()
    for path in sorted(INPUT.rglob('*.parquet')):
        rows = pq.read_table(path, columns=['id', 'source_name', 'source_frame_indices', 'num_frames',
            'vae_latent_shape', 'first_frame_latent_shape', 'camera_to_world_bytes', 'camera_to_world_shape',
            'camera_to_world_dtype']).to_pylist()
        for index, row in enumerate(rows):
            key = row['id']
            if key in seen:
                raise ValueError(f'Duplicate VAE ID {key}')
            seen.add(key)
            cap, reg = captions.get(key), registry.get(key)
            reason = None
            if not reg or not cap or cap.get('error') or cap.get('prompt_policy') != POLICY:
                reason = 'missing_valid_exact_window_vlm_caption'
            elif cap.get('window_id') != reg['window_id'] or cap['video'] != reg['video']:
                reason = 'caption_video_window_mismatch'
            elif row['source_frame_indices'] != reg['source_frame_indices']:
                reason = 'vae_caption_source_indices_mismatch'
            elif not set(cap['sampled_source_frame_indices']).issubset(set(row['source_frame_indices'])):
                reason = 'caption_samples_outside_vae_window'
            caption = clean_caption(cap.get('scene_description', '')) if cap else ''
            if not reason and len(caption.split()) < 5:
                reason = 'empty_or_uninformative_scene_caption'
            camera = np.frombuffer(row['camera_to_world_bytes'], dtype=row['camera_to_world_dtype']).reshape(row['camera_to_world_shape'])
            if not np.isfinite(camera).all():
                raise ValueError(f'VAE camera is not finite: {key}')
            # Raw VAE intermediates can retain the original video origin;
            # the existing conditioner rebases its final training output.
            # Apply that same derived-only repair in the new encoder below.
            requires_rebase = not np.allclose(camera[0], np.eye(4), atol=1e-5)
            rebased = np.linalg.inv(camera[0]) @ camera
            if not np.allclose(rebased[0], np.eye(4), atol=1e-4):
                raise ValueError(f'Camera cannot be rebased: {key}')
            if tuple(row['vae_latent_shape']) != (24, 37, 30, 52) or tuple(row['first_frame_latent_shape']) != (24, 1, 30, 52):
                raise ValueError(f'Bad native H3 latent grid: {key}')
            if reason:
                rejected.append({'id': key, 'reason': reason})
                continue
            selected.append(dict(id=key, source_name=row['source_name'], source_parquet=str(path),
                source_row=index, caption=caption, caption_sha256=sha(caption.encode()),
                caption_window=cap['window_id'], video=cap['video'],
                source_frame_indices=row['source_frame_indices'], camera_requires_rebase=requires_rebase, split='train'))
    if len(seen) != 50599:
        raise ValueError(f'Expected 50,599 existing new VAE rows; found {len(seen)}')
    if len(selected) < 1000:
        raise ValueError(f'Unexpectedly few captions matched: {len(selected)}')
    # Reserve complete source videos, not individual windows of a shared video.
    videos = sorted({r['video'] for r in selected}, key=lambda s: sha(s.encode()))
    heldout = set(videos[:128])
    for row in selected:
        if row['video'] in heldout:
            row['split'] = 'validation'
    selected.sort(key=lambda r: (r['source_parquet'], r['source_row']))
    args.root.mkdir(parents=True, exist_ok=True)
    path = args.root / 'text_data_plan.jsonl'
    payload = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in selected)
    if path.exists() and path.read_text() != payload:
        raise ValueError('Frozen training data plan changed')
    if not path.exists():
        path.write_text(payload)
    (args.root / 'excluded_ids.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rejected))
    counts = Counter(r['split'] for r in selected)
    save(args.root / 'data_plan_summary.json', dict(total=len(selected), split_counts=dict(counts),
        source_counts=dict(Counter(r['source_name'] for r in selected)),
        source_by_split={split: dict(Counter(r['source_name'] for r in selected if r['split']==split))
                         for split in ('train','validation')},
        excluded=len(rejected), excluded_reasons=dict(Counter(r['reason'] for r in rejected)),
        camera_rebase_on_derived_output=sum(r['camera_requires_rebase'] for r in selected),
        old_65389_neutral_caption_rows_not_used=True, captions_policy=POLICY,
        vae_reencoding=False, scene_disjoint_claim=False, source_video_disjoint=True,
        plan_sha256=sha(path.read_bytes()),
        initialization='union/checkpoint-800 model weights only; fresh optimizer and step counter'))
    print(json.dumps(dict(selected=len(selected), splits=dict(counts), rejected=len(rejected))), flush=True)


def encode(args):
    import torch
    import torch.distributed as dist
    from fastvideo.pipelines.preprocess.minimax_h3_worldmodel.preprocess import (
        _initialize_worker, _fastvideo_args, _load_component, _load_model_index, _atomic_write_records,
        rebase_prope_vae_record_camera)
    from fastvideo.pipelines.preprocess.minimax_h3_worldmodel.encoding import encode_h3_worldmodel_condition
    from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_conditioning import MiniMaxH3ConditioningStage
    from fastvideo.dataset.dataloader.schema import pyarrow_schema_minimax_h3_worldmodel_prope as schema
    from fastvideo.dataset.dataloader.record_schema import minimax_h3_worldmodel_prope_conditioned_record_creator

    all_rows = read_jsonl(args.root / 'text_data_plan.jsonl')
    rows = all_rows[args.worker::args.workers]
    marker = args.root / f'ENCODER_{args.worker}_COMPLETE'
    expected_marker = dict(plan_sha256=sha((args.root/'text_data_plan.jsonl').read_bytes()),
        worker=args.worker, workers=args.workers, rows=len(rows))
    if marker.exists():
        if json.loads(marker.read_text()) != expected_marker:
            raise ValueError('Stale encoder completion marker')
        return
    _initialize_worker(0, 30450 + args.worker)
    torch.set_num_threads(2)
    fast_args = _fastvideo_args(MODEL)
    index = _load_model_index(MODEL)
    encoder = _load_component('text_encoder', MODEL, index, fast_args)
    stage = MiniMaxH3ConditioningStage(conditioner=encoder,
        tokenizer=_load_component('tokenizer', MODEL, index, fast_args),
        processor=_load_component('processor', MODEL, index, fast_args))
    cached_path, source_rows = None, None
    for position, spec in enumerate(rows):
        output = args.root / 'preprocessed' / spec['split'] / f'worker_{args.worker:02d}' / f'data_{position:06d}.parquet'
        audit_path = output.with_suffix('.audit.json')
        if output.exists() and audit_path.exists():
            audit = json.loads(audit_path.read_text())
            if audit['id'] != spec['id'] or audit['caption_sha256'] != spec['caption_sha256']:
                raise ValueError('Existing text row provenance mismatch')
            continue
        if output.exists():
            raise ValueError(f'Incomplete row requires inspection before recovery: {output}')
        if cached_path != spec['source_parquet']:
            cached_path = spec['source_parquet']
            source_rows = pq.read_table(cached_path).to_pylist()
        source = source_rows[spec['source_row']]
        assert source['id'] == spec['id']
        image = Image.open(BytesIO(source['first_frame_png'])).convert('RGB')
        if image.size != (832,480):
            raise ValueError(f'Wrong cached first-frame geometry: {spec["id"]}')
        with torch.no_grad():
            embedding, tags = encode_h3_worldmodel_condition(stage, fast_args, caption=spec['caption'], first_frame=image)
        if embedding.shape[0] > 1024 or not bool(torch.isfinite(embedding).all()) or not bool((tags==1).any()):
            raise ValueError(f'Invalid or truncated text encoding: {spec["id"]}')
        source = rebase_prope_vae_record_camera(dict(source, caption=spec['caption']))
        record = minimax_h3_worldmodel_prope_conditioned_record_creator(source,
            text_embedding=embedding.numpy(), text_token_tags=tags.numpy())
        _atomic_write_records([record], schema, output)
        save(audit_path, dict(id=spec['id'], caption_sha256=spec['caption_sha256'],
            source_parquet=spec['source_parquet'], source_row=spec['source_row'],
            first_frame_png_sha256=sha(source['first_frame_png']),
            target_latent_sha256=sha(source['vae_latent_bytes']),
            first_frame_latent_sha256=sha(source['first_frame_latent_bytes']),
            camera_sha256=sha(source['camera_to_world_bytes']), intrinsics_sha256=sha(source['camera_intrinsics_bytes']),
            embedding_sha256=sha(record['text_embedding_bytes']),
            text_tokens=int((tags==1).sum()), vision_tokens=int((tags==0).sum()),
            text_rows=int(embedding.shape[0]), caption_window=spec['caption_window']))
        if position % 10 == 0:
            print(f'TEXT_READY worker={args.worker} {position+1}/{len(rows)} {spec["id"]}', flush=True)
        del record, embedding, tags
        if position % 100 == 0:
            gc.collect()
    save(marker, expected_marker)
    dist.destroy_process_group()


def audit(args):
    from tools.audit_scene_text_20261022 import audit_root
    audit_root(args.root, workers=args.audit_workers)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['plan','encode','audit'])
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--worker',type=int,default=0)
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--audit-workers',type=int,default=16)
    args=p.parse_args()
    globals()[args.command](args)


if __name__=='__main__':
    main()
