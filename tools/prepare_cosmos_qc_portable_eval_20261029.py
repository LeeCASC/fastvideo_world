"""Prepare text-enabled native H3 inputs without re-encoding benchmark videos.

The generic portable runner owns generation/decoding. This adapter binds every
unique (group, case) to its actual prompt, while both alpha variants share that
same native condition. It does not start or resume any trainer.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import gc
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml

ROOT = Path('/datacc05/lizihao/worldmodel')
OUTPUT = ROOT / 'fastvideowm_cosmos_poseqc_step1700_portable_eval_20261029'
PREVIOUS = ROOT / 'fastvideowm_union_step800_portable_eval_20261022'
TEXT_PREVIOUS = ROOT / 'fastvideowm_union_step800_subject_text_on_20261022'
BUNDLE = Path('/datacc05/lizihao/js_easy_benchmark_portable_20261021')
MODEL = Path('/datacc05/lizihao_wm/worldmodel/ckpts/MiniMax-H3')


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def text_hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'{path.name}.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    os.replace(temporary, path)


def read_row(path: Path) -> dict:
    table = pq.read_table(path)
    if table.num_rows != 1:
        raise ValueError(f'Expected a single native H3 row: {path}')
    return table.to_pylist()[0]


def tensor(row: dict, name: str) -> np.ndarray:
    result = np.frombuffer(row[f'{name}_bytes'], dtype=np.dtype(row[f'{name}_dtype']))
    result = result.reshape(row[f'{name}_shape'])
    if not np.isfinite(result).all():
        raise ValueError(f'Nonfinite tensor: {name}')
    return result


def condition_plan(cases: list[dict], output: Path) -> tuple[list[dict], list[dict]]:
    """Separate A/B IDs, deduplicate only exact alpha-independent conditions."""
    cases = deepcopy(cases)
    unique: dict[tuple[str, str], dict] = {}
    for case in cases:
        group, case_id = case['group'], case['id']
        if group not in ('trajectory_gt', 'single_action', 'single_action_text'):
            raise ValueError(f'Unexpected group: {group}')
        if Path(case_id).name != case_id or not case_id.startswith('js_'):
            raise ValueError('Unsafe case ID')
        path = output / 'conditions' / group / case_id / 'data.parquet'
        case.update(condition_parquet=str(path), text_condition_used=True)
        entry = {key: case[key] for key in ('group', 'id', 'sample_id', 'prompt', 'condition_parquet')}
        if not entry['prompt'].strip():
            raise ValueError(f'Empty prompt: {group}/{case_id}')
        key = (group, case_id)
        if key in unique and unique[key] != entry:
            raise ValueError(f'Alpha variants disagree about their condition: {key}')
        unique[key] = entry
    return cases, [unique[key] for key in sorted(unique)]


def link_directory(source: Path, target: Path) -> None:
    source = source.resolve(strict=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        target.symlink_to(source, target_is_directory=True)
    if target.resolve(strict=True) != source:
        raise ValueError(f'Input link points to another source: {target}')


def check_condition(case: dict, row: dict, encoding: dict, base: dict, first_frame_hash: str) -> dict:
    """Validate semantics, tensor contents, and image/target bindings on CPU."""
    if row['caption'] != case['prompt'] or encoding['prompt_sha256'] != text_hash(case['prompt']):
        raise ValueError(f'Actual prompt mismatch: {case["group"]}/{case["id"]}')
    if encoding['first_frame_png_sha256'] != first_frame_hash:
        raise ValueError('Condition was encoded with a different first frame')
    embedding, tags = tensor(row, 'text_embedding'), tensor(row, 'text_token_tags')
    if embedding.ndim != 2 or embedding.shape[1] != 5120 or not 1 <= len(embedding) <= 1024:
        raise ValueError('Invalid or truncated native H3 embedding')
    if tags.shape != (len(embedding), ) or not (tags == 0).any() or not (tags == 1).any():
        raise ValueError('Missing real vision/text tokens')
    if not np.isin(tags, (0, 1)).all():
        raise ValueError('Unexpected text/vision token tags')
    actual_hash = hashlib.sha256(row['text_embedding_bytes']).hexdigest()
    if encoding['embedding_sha256'] != actual_hash:
        raise ValueError('Encoded embedding content changed')
    for name in ('first_frame_latent', 'camera_intrinsics', 'camera_to_world', 'vae_latent'):
        tensor(row, name)
        for suffix in ('bytes', 'shape', 'dtype'):
            if row[f'{name}_{suffix}'] != base[f'{name}_{suffix}']:
                raise ValueError(f'Native source tensor changed: {name}')
    latent = tensor(row, 'vae_latent')
    if latent.shape != (24, 37, 30, 52) or tensor(row, 'first_frame_latent').shape != (24, 1, 30, 52):
        raise ValueError('Unexpected H3 latent layout')
    if case['group'] != 'trajectory_gt' and np.count_nonzero(latent):
        raise ValueError('Synthetic case contains future GT instead of a zero shape carrier')
    return dict(group=case['group'],
                id=case['id'],
                prompt_sha256=text_hash(case['prompt']),
                first_frame_png_sha256=first_frame_hash,
                embedding_sha256=actual_hash,
                text_tokens=int((tags == 1).sum()),
                vision_tokens=int((tags == 0).sum()),
                embedding_rows=len(embedding),
                gt_target=case['group'] == 'trajectory_gt',
                passed=True)


def base_row(args, case: dict) -> dict:
    filename = 'gt.parquet' if case['group'] == 'trajectory_gt' else 'simple.parquet'
    return read_row(args.output / 'inputs' / case['sample_id'] / filename)


def idle(args) -> None:
    """Inspect only the four GPUs made visible by this evaluation launcher."""
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 4:
        raise RuntimeError('Expected exactly the four selected CUDA GPUs, physical GPU0–3')
    busy = []
    for index in range(4):
        free, total = torch.cuda.mem_get_info(index)
        used_gib = (total - free) / 2**30
        if used_gib > 1.25:
            busy.append((index, round(used_gib, 2)))
    if busy:
        raise RuntimeError(f'Selected evaluation GPUs are occupied: {busy}')
    print('All four selected GPUs are idle; GPU4–7 are outside evaluation scope', flush=True)


def setup(args) -> None:
    cfg = yaml.safe_load((args.output / 'inference_config.yaml').read_text())
    student = cfg['models']['student']
    if student.get('vision_only_conditioning') is not False or student.get('neutralize_text_conditioning', False):
        raise ValueError('This evaluation must keep real text tokens enabled')
    if not student.get('init_from_training_checkpoint'):
        raise ValueError('Freeze must preserve the checkpoint parent initialization chain')
    distributed = cfg['training']['distributed']
    if (distributed['sp_size'] != 2 or distributed['num_gpus'] != 4 or distributed['hsdp_replicate_dim'] != 1
            or distributed['hsdp_shard_dim'] != 4):
        raise ValueError('Expected one four-rank HSDP1x4 model with two SP2 video lanes')
    original = json.loads((args.output / 'inference_manifest.json').read_text())
    cases, plan = condition_plan(original, args.output)
    if Counter(row['group'] for row in cases) != {'trajectory_gt': 32, 'single_action': 32, 'single_action_text': 64}:
        raise ValueError('Expected complete portable protocol, 128 cases')
    if len(plan) != 64 or {c['alpha'] for c in cases} != {1., .5}:
        raise ValueError('Expected 64 unique conditions and two alpha variants')
    source_audits = []
    for i in range(1, 17):
        sample = f'js_{i:05d}'
        source = args.previous / 'inputs' / sample
        if not (source / 'COMPLETE').is_file():
            raise ValueError(f'Native source cache is not complete: {sample}')
        alignment = json.loads((source / 'alignment.json').read_text())
        for filename, expected in alignment['source_sha256'].items():
            if sha256(args.bundle / 'data' / sample / filename) != expected:
                raise ValueError(f'Portable source no longer matches native cache: {sample}/{filename}')
        link_directory(source, args.output / 'inputs' / sample)
        for filename in ('gt.parquet', 'simple.parquet', 'first_frame.png', 'trajectory_gt.npy', 'single_action.npy',
                         'gt_pixels_61.npy'):
            if not (source / filename).is_file():
                raise FileNotFoundError(source / filename)
        for filename in ('trajectory_gt.npy', 'single_action.npy'):
            pose = np.load(source / filename, allow_pickle=False)
            if pose.shape != (37, 4, 4) or not np.isfinite(pose).all():
                raise ValueError(f'Invalid native camera path: {sample}/{filename}')
            if not np.allclose(pose[0], np.eye(4), atol=1e-4):
                raise ValueError(f'Native camera path is not first-frame relative: {sample}')
        source_audits.append(
            dict(sample_id=sample,
                 source=str(source.resolve()),
                 first_frame_png_sha256=sha256(source / 'first_frame.png'),
                 alignment_sha256=sha256(source / 'alignment.json')))
    atomic_json(args.output / 'inference_manifest.json', cases)
    atomic_json(args.output / 'condition_plan.json', plan)
    atomic_json(args.output / 'native_input_reuse_audit.json', dict(passed=True, cases=source_audits))
    reused = 0
    for case in plan:
        if case['group'] != 'single_action_text':
            continue
        old = args.text_previous / 'conditions' / case['id']
        if not (old / 'COMPLETE').is_file():
            raise ValueError(f'Expected reusable text condition is missing: {case["id"]}')
        row, encoding = read_row(old / 'data.parquet'), json.loads((old / 'encoding.json').read_text())
        first_hash = sha256(args.output / 'inputs' / case['sample_id'] / 'first_frame.png')
        check_condition(case, row, encoding, base_row(args, case), first_hash)
        target = Path(case['condition_parquet'])
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            os.link(old / 'data.parquet', target)
        if sha256(target) != sha256(old / 'data.parquet'):
            raise ValueError('Existing reused text condition differs from the verified source')
        atomic_json(
            target.parent / 'encoding.json',
            dict(encoding, reused_from=str(old.resolve()), group=case['group'], source_parquet_sha256=sha256(target)))
        (target.parent / 'COMPLETE').touch()
        reused += 1
    metadata = json.loads((args.output / 'checkpoints.json').read_text())
    metadata.update(unique_conditions=64,
                    native_input_reuse=str(args.previous),
                    reused_subject_text_conditions=reused,
                    newly_encoded_conditions=32,
                    training_paused=True,
                    automatically_resume_training=False,
                    text_condition_used=True,
                    training_vision_only_conditioning=False,
                    experiment_type='text-trained Cosmos pose-QC model, complete portable A/B/C evaluation')
    metadata.setdefault('protocol_differences', {}).update(
        text='Actual complete per-case prompt and exact first frame encoded with native H3 Qwen3-VL; text enabled',
        C_group='Matched forward/backward subject-text interventions on a text-trained checkpoint; no matched GT')
    atomic_json(args.output / 'checkpoints.json', metadata)
    atomic_json(
        args.output / 'status.json',
        dict(phase='conditions_prepared',
             expected_videos=128,
             reused_conditions=reused,
             pending_conditions=32,
             text_condition_used=True,
             training_paused=True))
    print(f'SETUP_COMPLETE native_inputs=16 reused_text={reused} new_conditions=32', flush=True)


def condition(args) -> None:
    """Only load Qwen when this worker actually has uncached A/B prompts."""
    plan = json.loads((args.output / 'condition_plan.json').read_text())
    pending = [
        case for case in plan[args.worker::args.workers]
        if not (Path(case['condition_parquet']).parent / 'COMPLETE').is_file()
    ]
    if not pending:
        print(f'CONDITION_WORKER_ALREADY_COMPLETE {args.worker}', flush=True)
        return
    from PIL import Image
    import torch
    import torch.distributed as dist
    from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_conditioning import MiniMaxH3ConditioningStage
    from fastvideo.pipelines.preprocess.minimax_h3_worldmodel.encoding import encode_h3_worldmodel_condition
    from fastvideo.pipelines.preprocess.minimax_h3_worldmodel.preprocess import (
        _fastvideo_args,
        _initialize_worker,
        _load_component,
        _load_model_index,
    )
    from scripts.evaluation.portable_js_easy_h3 import write_row
    from scripts.evaluation.prepare_minimax_h3_js_easy_benchmark_20261008 import _set_float32_tensor

    torch.set_num_threads(1)
    _initialize_worker(0, 30830 + args.worker)
    fast_args, index = _fastvideo_args(args.model_path), _load_model_index(args.model_path)
    tokenizer = _load_component('tokenizer', args.model_path, index, fast_args)
    processor = _load_component('processor', args.model_path, index, fast_args)
    encoder = _load_component('text_encoder', args.model_path, index, fast_args)
    stage = MiniMaxH3ConditioningStage(conditioner=encoder, tokenizer=tokenizer, processor=processor)
    for case in pending:
        first = args.output / 'inputs' / case['sample_id'] / 'first_frame.png'
        embedding, tags = encode_h3_worldmodel_condition(stage,
                                                         fast_args,
                                                         caption=case['prompt'],
                                                         first_frame=Image.open(first).convert('RGB'))
        if embedding.shape[0] > 1024 or not torch.isfinite(embedding).all():
            raise ValueError(f'Invalid or truncated embedding: {case["group"]}/{case["id"]}')
        row = base_row(args, case)
        row.update(caption=case['prompt'], id=f'portable_poseqc/{case["group"]}/{case["id"]}')
        _set_float32_tensor(row, 'text_embedding', embedding.cpu().numpy())
        _set_float32_tensor(row, 'text_token_tags', tags.cpu().numpy())
        target = Path(case['condition_parquet'])
        encoding = dict(group=case['group'],
                        case_id=case['id'],
                        sample_id=case['sample_id'],
                        prompt=case['prompt'],
                        prompt_sha256=text_hash(case['prompt']),
                        first_frame_png_sha256=sha256(first),
                        embedding_sha256=hashlib.sha256(row['text_embedding_bytes']).hexdigest(),
                        encoder=str(args.model_path / 'text_encoder'),
                        prompt_truncated=False,
                        text_condition_used=True,
                        reused_from=None)
        check_condition(case, row, encoding, base_row(args, case), sha256(first))
        write_row(target, row)
        atomic_json(target.parent / 'encoding.json', encoding)
        (target.parent / 'COMPLETE').touch()
        print(f'TEXT_ENCODED {case["group"]}/{case["id"]} tokens={embedding.shape[0]}', flush=True)
    del stage, encoder
    gc.collect()
    torch.cuda.empty_cache()
    dist.destroy_process_group()


def audit(args) -> None:
    plan = json.loads((args.output / 'condition_plan.json').read_text())
    checks: list[dict] = []
    pair_embeddings: dict[str, list[str]] = {}
    for case in plan:
        target = Path(case['condition_parquet'])
        if not (target.parent / 'COMPLETE').is_file():
            raise ValueError(f'Condition is incomplete: {target}')
        row, encoding = read_row(target), json.loads((target.parent / 'encoding.json').read_text())
        checked = check_condition(case, row, encoding, base_row(args, case),
                                  sha256(args.output / 'inputs' / case['sample_id'] / 'first_frame.png'))
        checks.append(checked)
        if case['group'] == 'single_action_text':
            pair_embeddings.setdefault(case['sample_id'], []).append(checked['embedding_sha256'])
    if len(checks) != 64 or len(pair_embeddings) != 16:
        raise ValueError('Missing unique conditions or subject pairs')
    if any(len(values) != 2 or values[0] == values[1] for values in pair_embeddings.values()):
        raise ValueError('Missing distinct forward/backward text embeddings')
    atomic_json(
        args.output / 'text_condition_audit.json',
        dict(passed=True, unique_conditions=64, actual_cases=128, checks=checks, future_gt_used_for_generation=False))
    (args.output / 'TEXT_CONDITIONS_COMPLETE').touch()
    print('TEXT_CONDITION_AUDIT_PASSED unique_conditions=64 videos=128', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('idle', 'setup', 'condition', 'audit'))
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--previous', type=Path, default=PREVIOUS)
    parser.add_argument('--text-previous', type=Path, default=TEXT_PREVIOUS)
    parser.add_argument('--bundle', type=Path, default=BUNDLE)
    parser.add_argument('--model-path', type=Path, default=MODEL)
    parser.add_argument('--worker', type=int, default=0)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    if not 0 <= args.worker < args.workers:
        raise ValueError('Invalid condition worker split')
    globals()[args.command](args)


if __name__ == '__main__':
    main()
