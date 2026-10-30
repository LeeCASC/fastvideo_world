"""Resume the unchanged eight-GPU run only after this evaluation has exited."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import yaml

REPO = Path(__file__).resolve().parents[1]
ROOT = Path('/datacc05/lizihao/worldmodel')
EVALUATION = ROOT / 'fastvideowm_cosmos_poseqc_step1700_portable_eval_20261029'
TRAINING = ROOT / 'fastvideowm_cosmos_poseqc_108439_resume1100_20261027/subject_motion'


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def dump(path: Path, value: dict) -> None:
    temporary = path.with_name(f'{path.name}.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def evaluation_ready(root: Path) -> bool:
    if (root / 'NEEDS_ATTENTION').exists():
        raise RuntimeError('Evaluation failed; do not start training')
    required = ('ALL_LATENTS_COMPLETE', 'VISUALIZATION_COMPLETE', 'INFERENCE_COMPLETE', 'CORE_METRICS_COMPLETE')
    if not all((root / name).is_file() for name in required):
        return False
    cases = json.loads((root / 'inference_manifest.json').read_text())
    if len(cases) != 128 or len({row['output_key'] for row in cases}) != 128:
        raise ValueError('Expected128 distinct evaluation outputs')
    for row in cases:
        relative = Path(row['output_key'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe evaluation output path')
        folder = root / 'videos' / relative
        if not (folder / 'VIDEO_COMPLETE').is_file() or (folder / 'raw.mp4').stat().st_size == 0:
            raise ValueError(f'Missing validated video: {relative}')
    metrics = json.loads((root / 'metrics.json').read_text())
    if len(metrics['gt_cases']) != 32 or len(metrics['no_reference_cases']) != 96:
        raise ValueError('Incomplete metric case coverage')
    status = json.loads((root / 'status.json').read_text())
    if status.get('phase') not in {'complete', 'complete_with_metric_limitations'}:
        # Scoring publishes markers just before its atomic final status.
        return False
    if status.get('videos_complete') != 128:
        raise ValueError('Evaluation completion status does not cover128 videos')
    # Optional VBench dependencies may be explicitly unavailable. They are not
    # fabricated successes and need not block the approved training continuation.
    return True


def validate_data_proof(root: Path) -> dict:
    gate = json.loads((root / 'RESUME_DATA_READY').read_text())
    if sha(root / 'resume_data_audit.json') != gate['proof_sha256']:
        raise ValueError('Dataset proof changed')
    proof = json.loads((root / 'resume_data_audit.json').read_text())
    if not proof['passed'] or proof['split_counts'] != {'train': 105935, 'validation': 2504}:
        raise ValueError('Expected approved108439-row dataset')
    for path, expected in [(root / 'text_data_plan.jsonl', proof['plan_sha256']),
                           (root / 'training_config.yaml', proof['training_config_sha256'])]:
        if sha(path) != expected:
            raise ValueError(f'Prepared data/config changed: {path}')
    for split, cache in proof['caches'].items():
        expected_path = root / 'preprocessed' / split / 'map_style_cache/file_info.pkl'
        if Path(cache['path']) != expected_path or sha(expected_path) != cache['sha256']:
            raise ValueError('Prepared native loader index changed')
    return yaml.safe_load((root / 'training_config.yaml').read_text())


def latest_checkpoint(root: Path, config: dict) -> Path:
    directory = root / 'runs/train/checkpoints'
    complete = []
    for path in directory.glob('checkpoint-*'):
        if not path.name.removeprefix('checkpoint-').isdigit():
            continue
        step = int(path.name.removeprefix('checkpoint-'))
        if step < 1700:
            continue
        try:
            meta = json.loads((path / 'metadata.json').read_text())
            required = [path / 'dcp/.metadata', *[path / f'rng_state_rank{i}.pt' for i in range(8)]]
            shards = list((path / 'dcp').glob('*.distcp'))
            if meta['step'] != step or len(shards) != 8:
                continue
            if any(not p.is_file() or p.stat().st_size == 0 for p in required + shards):
                continue
            if any(time.time() - p.stat().st_mtime < 30 for p in [path / 'metadata.json', *required, *shards]):
                continue
            if meta['config']['training']['data'] != config['training']['data']:
                raise ValueError('Checkpoint dataset/config differs from the approved unchanged run')
            candidate_config = deepcopy(meta['config'])
            original_config = deepcopy(config)
            for cfg in (candidate_config, original_config):
                cfg['training']['checkpoint'].pop('resume_from_checkpoint', None)
            if candidate_config != original_config:
                raise ValueError('Checkpoint recipe differs beyond its resume pointer')
            complete.append((step, path))
        except FileNotFoundError:
            continue
    if not complete:
        raise ValueError('No complete same-run checkpoint at step1700 or later')
    return max(complete)[1]


def make_resume_config(config: dict, checkpoint: Path, root: Path) -> dict:
    result = deepcopy(config)
    training = result['training']
    dist = training['distributed']
    if (dist['num_gpus'], dist['sp_size'], dist['tp_size'], dist['hsdp_replicate_dim'],
            dist['hsdp_shard_dim']) != (8, 2, 1, 2, 4):
        raise ValueError('Expected eightGPU/SP2/HSDP2x4')
    if training['data']['train_batch_size'] != 1 or training['loop'] != {
            'max_train_steps': 3000,
            'gradient_accumulation_steps': 8
    }:
        raise ValueError('Keep totalBS32 and global target3000')
    if training['data']['data_path'] != str(root / 'preprocessed/train'):
        raise ValueError('Wrong training data root')
    if training['checkpoint']['output_dir'] != str(root / 'runs/train/checkpoints'):
        raise ValueError('Wrong checkpoint output root')
    if result['models']['student']['vision_only_conditioning']:
        raise ValueError('Keep background+subject text conditioning')
    if not result['models']['student'].get('init_from_training_checkpoint'):
        raise ValueError('Preserve parent100 model bootstrap before full resume')
    training['checkpoint']['resume_from_checkpoint'] = str(checkpoint)
    return result


def environment(root: Path,
                config_path: Path,
                model_path: str = '/datacc05/lizihao_wm/worldmodel/ckpts/MiniMax-H3') -> dict:
    return dict(os.environ,
                CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',
                NUM_GPUS='8',
                MASTER_PORT='30731',
                MODEL_PATH=model_path,
                CONFIG_FILE=str(config_path),
                TRAIN_DATA=str(root / 'preprocessed/train'),
                VALIDATION_DATA=str(root / 'preprocessed/validation'),
                LOG_DIR=str(root / 'runs/train/logs'),
                WANDB_MODE='offline',
                TORCHDYNAMO_DISABLE='1',
                TORCH_COMPILE_DISABLE='1',
                OMP_NUM_THREADS='1',
                OPENBLAS_NUM_THREADS='1')


def run(args) -> None:
    root, evaluation = args.training.resolve(), args.evaluation.resolve()
    config = validate_data_proof(root)
    checkpoint = latest_checkpoint(root, config)
    make_resume_config(config, checkpoint, root)
    if args.check:
        print(json.dumps(
            dict(checkpoint=str(checkpoint),
                 total_bs=32,
                 target_step=3000,
                 evaluation_ready=evaluation_ready(evaluation),
                 no_gpu_started=True)),
              flush=True)
        return
    lock = (evaluation / 'autoresume.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = evaluation / 'autoresume_state.json'
    deadline = time.monotonic() + args.max_wait_hours * 3600
    previous = None
    evaluation_lock = None
    training_lock = None
    while True:
        phase = 'waiting_for_evaluation'
        if evaluation_ready(evaluation):
            if evaluation_lock is None:
                candidate = (evaluation / 'job.lock').open('a')
                try:
                    fcntl.flock(candidate, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    evaluation_lock = candidate
                except BlockingIOError:
                    candidate.close()
            if evaluation_lock is not None:
                if training_lock is None:
                    training_lock = (root / 'launch.lock').open('a')
                    fcntl.flock(training_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                check = subprocess.run([str(REPO / '.venv/bin/python'), 'examples/train/check_minimax_h3_gpus_idle.py'],
                                       cwd=REPO,
                                       env=environment(root, root / 'training_config.yaml'),
                                       capture_output=True,
                                       text=True,
                                       timeout=120)
                if check.returncode not in (0, 1):
                    raise RuntimeError(f'CUDA/GPU idle check failed: {check.stdout} {check.stderr}')
                if check.returncode == 0:
                    break
                phase = 'waiting_for_eight_idle_gpus'
        state = dict(phase=phase,
                     updated_at=datetime.now().astimezone().isoformat(),
                     poll_seconds=300,
                     checkpoint=str(checkpoint),
                     total_bs=32,
                     target_step=3000,
                     automatic_training_resume_authorized=True,
                     no_other_processes_stopped=True)
        dump(state_path, state)
        if phase != previous:
            print(json.dumps(state), flush=True)
            previous = phase
        if time.monotonic() >= deadline:
            raise TimeoutError('Automatic resume wait exceeded its24h bound; no process was stopped')
        time.sleep(300)
    # Recheck provenance after waiting; write a separate config, never mutate
    # the frozen original or its hash-bound dataset preparation proof.
    config = validate_data_proof(root)
    checkpoint = latest_checkpoint(root, config)
    resumed = make_resume_config(config, checkpoint, root)
    if int(checkpoint.name.removeprefix('checkpoint-')) >= 3000:
        dump(state_path, dict(phase='already_at_target', checkpoint=str(checkpoint)))
        return
    disk = os.statvfs(root)
    if disk.f_bavail * disk.f_frsize < 512 * 1024**3:
        raise RuntimeError('Insufficient512GiB checkpoint disk reserve')
    config_path = root / 'training_resume_after_eval_20261029.yaml'
    config_path.write_text(yaml.safe_dump(resumed, sort_keys=False))
    state = dict(phase='launching_training',
                 checkpoint=str(checkpoint),
                 total_bs=32,
                 target_step=3000,
                 config=str(config_path),
                 config_sha256=sha(config_path),
                 updated_at=datetime.now().astimezone().isoformat(),
                 full_state_resume=True,
                 same_dataset_restore_dataloader=True,
                 automatic_training_resume_authorized=True)
    dump(state_path, state)
    dump(root / 'POST_EVALUATION_RESUME_AUTHORIZED_20261029.json', state)
    log = root / 'resume_after_eval_20261029.log'
    print('FULL_STATE_RESUME', json.dumps(state), flush=True)
    with log.open('a') as stream:
        result = subprocess.run(['bash', 'examples/train/launch_minimax_h3_worldmodel_stage0.sh'],
                                cwd=REPO,
                                env=environment(root, config_path, config['models']['student']['init_from']),
                                stdout=stream,
                                stderr=subprocess.STDOUT,
                                check=False)
    if result.returncode:
        raise RuntimeError(f'Trainer exited with{result.returncode}; see{log}')
    dump(
        state_path,
        dict(phase='training_process_exited_successfully',
             target_step=3000,
             updated_at=datetime.now().astimezone().isoformat()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', type=Path, default=EVALUATION)
    parser.add_argument('--training', type=Path, default=TRAINING)
    parser.add_argument('--check', action='store_true', help='Read-only CPU preflight; never launch or write')
    parser.add_argument('--max-wait-hours', type=float, default=24)
    args = parser.parse_args()
    if not 0 < args.max_wait_hours <= 24:
        raise ValueError('Wait must be bounded to at most24hours')
    try:
        run(args)
    except Exception as error:
        if not args.check:
            dump(args.evaluation / 'AUTORESUME_NEEDS_ATTENTION.json',
                 dict(error=f'{type(error).__name__}: {error}', updated_at=datetime.now().astimezone().isoformat()))
        raise


if __name__ == '__main__':
    main()
