"""Bind completed numeric QC to an immutable native-H3 subset and full resume.

Prepare file lengths on CPU BEFORE initializing distributed GPU training. Source
videos, caches and checkpoints are read-only; no architecture/loss changes.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import time
from collections.abc import Iterator

import pyarrow.parquet as pq
import yaml

REPO = Path(__file__).resolve().parents[1]
QC = Path('/datacc05/lizihao/worldmodel/fastvideowm_cosmos_pose_qc_20261027')
SOURCE = Path('/datacc05/lizihao/worldmodel/fastvideowm_union_scene_subject_poseqc_20261027/subject_motion')
OUTPUT = Path('/datacc05/lizihao/worldmodel/fastvideowm_cosmos_poseqc_108439_resume1100_20261027/subject_motion')
CHECKPOINT = Path(
    '/datacc05/lizihao/worldmodel/fastvideowm_union_scene_subject_text_20261024/subject_motion/runs/train/checkpoints/checkpoint-1100'
)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def rows(path: Path) -> Iterator[dict]:
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def dump(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def checkpoint_complete(path: Path) -> None:
    metadata = json.loads((path / 'metadata.json').read_text())
    if metadata['step'] != 1100 or not (path / 'dcp/.metadata').is_file():
        raise ValueError('Expected complete global-step1100 checkpoint')
    shards = list((path / 'dcp').glob('*.distcp'))
    rng = [path / f'rng_state_rank{i}.pt' for i in range(8)]
    if len(shards) != 8 or any(not p.is_file() or p.stat().st_size == 0 for p in shards + rng):
        raise ValueError('Incomplete model/optimizer/RNG checkpoint')


def safe_relative(row: dict) -> Path:
    path = Path(row['conditioned_relative_path'])
    if row['split'] not in {'train', 'validation'} or path.is_absolute() or '..' in path.parts:
        raise ValueError('Unsafe subset path')
    if len(path.parts) != 4 or path.parts[:2] != ('preprocessed', row['split']) or path.suffix != '.parquet':
        raise ValueError('Conditioned path must be within its declared split')
    return path


def link_and_validate(job: tuple[dict, dict, Path, Path]) -> tuple[str, int]:
    row, audit, source, output = job
    rel = safe_relative(row)
    if audit['reasons'] or audit['conditioned_relative_path'] != str(rel):
        raise ValueError('QC did not accept the exact conditioned row')
    path = source / rel
    before = path.stat()
    if (before.st_size, before.st_mtime_ns) != (audit['file_size'], audit['mtime_ns']):
        raise ValueError(f'Input changed after numeric QC: {row["id"]}')
    sidecar = path.with_suffix('.audit.json')
    binding = json.loads(sidecar.read_text())
    for key in ('id', 'caption_sha256', 'video', 'source_frame_indices'):
        if binding.get(key) != row.get(key):
            raise ValueError(f'Full parent audit binding differs: {row["id"]} {key}')
    # Re-read actual Parquet metadata in bounded parallel CPU workers. Never
    # assume one row just because a filename or manifest says so.
    count = pq.ParquetFile(path).metadata.num_rows
    if count != 1:
        raise ValueError('Expected exactly one conditioned example per Parquet')
    for original in (path, sidecar):
        target = output / original.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink():
            raise ValueError('Derived cache must not be a mutable cross-root symlink')
        if not target.exists():
            os.link(original, target)
        if not target.samefile(original):
            raise ValueError('Derived cache is not identical to accepted parent')
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('Input changed during preparation')
    return str((output / rel).resolve(strict=True)), count


def write_cache(root: Path, entries: list[tuple[str, int]]) -> dict:
    """Native loader's exact sorted (absolute_paths, row_counts) pickle format."""
    if not entries or len({p for p, _ in entries}) != len(entries):
        raise ValueError('Empty or duplicate dataset cache')
    for path, count in entries:
        if not Path(path).is_relative_to(root.resolve()) or count != 1:
            raise ValueError('Cache must reference only the new audited one-row subset')
    ordered = sorted(entries)
    paths, counts = zip(*ordered, strict=True)
    target = root / 'map_style_cache/file_info.pkl'
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix('.tmp')
    with tmp.open('wb') as handle:
        pickle.dump((paths, counts), handle)
    tmp.replace(target)
    with target.open('rb') as handle:
        if pickle.load(handle) != (paths, counts):
            raise ValueError('Persisted native dataset index differs from validated metadata')
    return dict(path=str(target), files=len(paths), rows=sum(counts), sha256=sha(target))


def make_config(template: dict, output: Path, checkpoint: Path) -> dict:
    # JSON deep copy; checkpoint metadata is the authoritative previous recipe.
    cfg = json.loads(json.dumps(template))
    student = cfg['models']['student']
    if student['vision_only_conditioning'] or student['normalize_camera_translation']:
        raise ValueError('Expected existing text-on, unnormalized numeric camera recipe')
    if cfg['training']['loop'] != dict(max_train_steps=3000, gradient_accumulation_steps=8):
        raise ValueError('Keep global target3000 and accumulation8 unchanged')
    dist = cfg['training']['distributed']
    if (dist['num_gpus'], dist['sp_size'], dist['tp_size']) != (8, 2, 1):
        raise ValueError('Expected8GPU / SP2 / DP4, totalBS32')
    if cfg['training']['data']['train_batch_size'] != 1:
        raise ValueError('Expected microbatch1')
    # Keep init_from_training_checkpoint=parent100 to reconstruct inherited
    # frozen values; trainer subsequently restores FULL1100 DCP and optimizer.
    cfg['training']['data']['data_path'] = str(output / 'preprocessed/train')
    cfg['callbacks']['worldmodel_validation']['data_path'] = str(output / 'preprocessed/validation')
    cfg['callbacks']['text_input_guard']['output_dir'] = str(output / 'runs/train')
    cfg['training']['checkpoint'].update(output_dir=str(output / 'runs/train/checkpoints'),
                                         resume_from_checkpoint=str(checkpoint),
                                         reset_dataloader_on_data_change=True)
    cfg['training']['tracker']['run_name'] = 'h3_cosmos_poseqc_108439_resume1100_20261027'
    return cfg


def prepare(args: argparse.Namespace) -> None:
    source, output, qc = args.source.resolve(), args.output.resolve(), args.qc.resolve()
    if source == output or source.is_relative_to(output) or output.is_relative_to(source):
        raise ValueError('Output must be independent of all historical cache roots')
    checkpoint_complete(args.checkpoint)
    gate = json.loads((qc / 'NUMERIC_POSE_QC_COMPLETE').read_text())
    for path, expected in [(source / 'text_data_plan.jsonl', gate['input_plan_sha256']),
                           (qc / 'accepted.jsonl', gate['accepted_manifest_sha256']),
                           (qc / 'audit.jsonl', gate['audit_sha256'])]:
        if sha(path) != expected:
            raise ValueError(f'QC gate hash mismatch: {path}')
    inherited = json.loads((source / 'subset_data_audit.json').read_text())
    if not (source / 'SUBSET_DATA_VERIFIED').is_file() or not inherited['passed']:
        raise ValueError('Missing unchanged subset / full parent tensor audit')
    parent_audit = Path(inherited['parent_full_tensor_audit'])
    if sha(parent_audit) != inherited['parent_audit_sha256']:
        raise ValueError('Historical full tensor audit changed')
    previous_audit = json.loads(parent_audit.read_text())
    if not previous_audit['passed']:
        raise ValueError('Historical full tensor audit did not pass')
    specs = list(rows(qc / 'accepted.jsonl'))
    rejected = {r['id'] for r in rows(qc / 'rejected.jsonl')}
    seen = {r['id'] for r in specs}
    if len(specs) != 108439 or len(seen) != len(specs) or seen & rejected:
        raise ValueError('Not the approved108439 unique accepted windows')
    counts = dict(Counter(r['split'] for r in specs))
    if counts != dict(train=105935, validation=2504):
        raise ValueError('Unexpected current accepted split')
    audits = {r['id']: r for r in rows(qc / 'audit.jsonl') if r['id'] in seen}
    if set(audits) != seen or any(r['reasons'] for r in audits.values()):
        raise ValueError('Incomplete per-ID passing audit')
    output.mkdir(parents=True, exist_ok=True)
    if any((output / 'runs/train/checkpoints').glob('checkpoint-*')):
        raise ValueError('Already-trained root must not be rewritten')
    lock = (output / 'prepare.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    entries: dict[str, list] = {'train': [], 'validation': []}
    jobs = ((r, audits[r['id']], source, output) for r in specs)
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        # chunks are bounded rather than allocating108k futures at once.
        for start in range(0, len(specs), 1024):
            batch = [next(jobs) for _ in range(min(1024, len(specs) - start))]
            for row, entry in zip(specs[start:start + 1024], pool.map(link_and_validate, batch), strict=True):
                entries[row['split']].append(entry)
            done = sum(len(v) for v in entries.values())
            progress = dict(checked=done, total=len(specs), elapsed_seconds=time.monotonic() - started)
            dump(output / 'prepare_progress.json', progress)
            print('PREPARE', json.dumps(progress), flush=True)
    # Exact path set, not only file totals, must match the whitelist.
    actual = {str(p.resolve()) for p in (output / 'preprocessed').rglob('*.parquet')}
    wanted = {p for values in entries.values() for p, _ in values}
    if actual != wanted:
        raise ValueError('Unexpected missing/extra Parquet files in training tree')
    caches = {split: write_cache(output / 'preprocessed' / split, values) for split, values in entries.items()}
    with (output / 'text_data_plan.jsonl').open('w') as handle:
        for r in specs:
            r['source_conditioned_parquet'] = str(source / safe_relative(r))
            r['conditioned_data_root'] = str(output)
            r['conditioned_parquet'] = str(output / safe_relative(r))
            handle.write(json.dumps(r, ensure_ascii=False) + '\n')
    template = json.loads((args.checkpoint / 'metadata.json').read_text())['config']
    config = make_config(template, output, args.checkpoint)
    config_path = output / 'training_config.yaml'
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    proof = dict(passed=True,
                 rows=len(specs),
                 split_counts=counts,
                 source_root=str(source),
                 qc_root=str(qc),
                 qc_gate=gate,
                 source_manifest_sha256=gate['accepted_manifest_sha256'],
                 plan_sha256=sha(output / 'text_data_plan.jsonl'),
                 training_config_sha256=sha(config_path),
                 inherited_full_tensor_audit=str(parent_audit),
                 full_tensor_reaudit=False,
                 numeric_pose_qc_all_rows=True,
                 retained_same_inode=True,
                 every_parquet_footer_rechecked=True,
                 caches=caches,
                 checkpoint=str(args.checkpoint),
                 source_files_deleted=False,
                 elapsed_seconds=time.monotonic() - started)
    dump(output / 'resume_data_audit.json', proof)
    dump(output / 'data_plan_summary.json',
         dict(total=len(specs), split_counts=counts, source_counts=dict(Counter(r['source_name'] for r in specs))))
    dump(output / 'RESUME_DATA_READY', dict(proof_sha256=sha(output / 'resume_data_audit.json')))
    print('RESUME_DATA_READY', json.dumps(proof), flush=True)


def launch(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    marker = json.loads((output / 'RESUME_DATA_READY').read_text())
    if sha(output / 'resume_data_audit.json') != marker['proof_sha256']:
        raise ValueError('Preparation proof changed')
    proof = json.loads((output / 'resume_data_audit.json').read_text())
    if sha(output / 'text_data_plan.jsonl') != proof['plan_sha256']:
        raise ValueError('Prepared plan changed')
    if sha(output / 'training_config.yaml') != proof['training_config_sha256']:
        raise ValueError('Prepared training configuration changed')
    for cache in proof['caches'].values():
        if sha(Path(cache['path'])) != cache['sha256']:
            raise ValueError('Prepared file-length cache changed')
    checkpoint_complete(Path(proof['checkpoint']))
    if any((output / 'runs/train/checkpoints').glob('checkpoint-*')):
        raise ValueError('Run already has newer checkpoints; bind its latest complete checkpoint explicitly')
    lock = (output / 'launch.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    env = dict(os.environ,
               CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',
               NUM_GPUS='8',
               MASTER_PORT='30731',
               CONFIG_FILE=str(output / 'training_config.yaml'),
               TRAIN_DATA=str(output / 'preprocessed/train'),
               VALIDATION_DATA=str(output / 'preprocessed/validation'),
               LOG_DIR=str(output / 'runs/train/logs'),
               WANDB_MODE='offline',
               TORCHDYNAMO_DISABLE='1',
               TORCH_COMPILE_DISABLE='1',
               OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1')
    subprocess.run([str(REPO / '.venv/bin/python'), 'examples/train/check_minimax_h3_gpus_idle.py'],
                   cwd=REPO,
                   env=env,
                   check=True)
    if os.statvfs(output).f_bavail * os.statvfs(output).f_frsize < 512 * 1024**3:
        raise RuntimeError('Insufficient512GiB checkpoint disk reserve')
    print('FULL_RESUME_FROM1100 total_bs32 / DP4 / SP2 / micro1 / accumulation8', flush=True)
    result = subprocess.run(['bash', 'examples/train/launch_minimax_h3_worldmodel_stage0.sh'],
                            cwd=REPO,
                            env=env,
                            check=False)
    if result.returncode:
        dump(output / 'NEEDS_ATTENTION.json', dict(stage='training', exit_code=result.returncode))
        raise RuntimeError(f'Training exited with {result.returncode}')
    dump(output / 'TRAINING_COMPLETE', dict(target_step=3000))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'launch'])
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--qc', type=Path, default=QC)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT)
    parser.add_argument('--workers', type=int, default=24)
    args = parser.parse_args()
    if not 1 <= args.workers <= 32:
        raise ValueError('Require1..32 bounded CPU workers')
    globals()[args.command](args)


if __name__ == '__main__':
    main()
