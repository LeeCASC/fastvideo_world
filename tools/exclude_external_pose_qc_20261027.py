"""Exclude user-approved source IDs without deleting live/shared training files.

The attachment is DATA ONLY. Never execute its paths or use them as deletion
targets. Derive exact native IDs, freeze an exclusion manifest, then hardlink
the unchanged approved subset into a new root. Switching training is a separate
explicit command, after a complete checkpoint; original caches remain recoverable.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path, PurePosixPath
import re
import signal
import subprocess
import time

import pyarrow.parquet as pq
import yaml

from tools.visualize_training_sources_20261027 import digest, save_json

REPO = Path(__file__).resolve().parents[1]
SOURCE = Path('/datacc05/lizihao/worldmodel/fastvideowm_union_scene_subject_text_20261024/subject_motion')
OUTPUT = Path('/datacc05/lizihao/worldmodel/fastvideowm_union_scene_subject_poseqc_20261027/subject_motion')
ATTACHMENT = Path('/home/lizihao/.codex/attachments/fbec89e0-3b75-4674-b44d-87e30e34475d/moved_files.jsonl')
PARENT_SHA = '384a081db2fc47b9134c386b98e6ff14728858ceafe34ebe07950078e21065a4'


def alias(value: str) -> str:
    return re.sub(r'[:/\\]', '__', value)


def attachment_key(relative_path: str) -> tuple[str, str, str]:
    path = PurePosixPath(relative_path)
    if path.is_absolute() or len(path.parts) != 3 or '..' in path.parts or path.suffix != '.pt':
        raise ValueError('Attachment must contain relative version/source/sample.pt records')
    version, source, _ = path.parts
    if version not in ('61f', '121f', '241f'):
        raise ValueError('Unknown external frame version')
    stem = re.sub(r'__w\d+__[0-9a-f]{40}$', '', path.stem)
    return version, source, stem


def match_rows(specs: list[dict], external: list[dict]) -> tuple[dict, list[dict], list[dict]]:
    lookup: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    for spec in specs:
        for key in ('id', 'original_global_id'):
            lookup[(spec['source_name'], alias(spec[key]))][spec['id']] = spec
    matches: dict[str, list[dict]] = defaultdict(list)
    unmatched, mapping = [], []
    for number, external_row in enumerate(external, 1):
        version, source, stem = attachment_key(external_row['relative_path'])
        hits = lookup.get((source, stem), {})
        if len(hits) > 1:
            raise ValueError(f'Ambiguous source ID: {external_row["relative_path"]}')
        row = dict(attachment_line=number,
                   external_version=version,
                   relative_path=external_row['relative_path'],
                   reasons=external_row.get('reasons', []))
        mapping.append(dict(row, matched_ids=list(hits)))
        if not hits:
            unmatched.append(row)
        for sample_id in hits:
            matches[sample_id].append(row)
    return dict(matches), unmatched, mapping


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))


def linked_pair(job: tuple[dict, str, str, bool]) -> str:
    spec, source_string, output_string, excluded = job
    source, output = Path(source_string), Path(output_string)
    source_file = source / spec['conditioned_relative_path']
    source_sidecar = source_file.with_suffix('.audit.json')
    sidecar = json.loads(source_sidecar.read_text())
    for key in ('id', 'caption_sha256', 'video', 'source_frame_indices'):
        if sidecar.get(key) != spec.get(key):
            raise ValueError(f'Source sidecar does not match frozen plan: {spec["id"]} {key}')
    if excluded:
        actual = pq.ParquetFile(source_file).read(columns=['id', 'caption', 'source_frame_indices'],
                                                  use_threads=False).to_pylist()
        if len(actual) != 1 or any(actual[0][k] != spec[k] for k in ('id', 'caption', 'source_frame_indices')):
            raise ValueError(f'Exclusion target differs from plan: {spec["id"]}')
        return 'excluded'
    for source_path in (source_file, source_sidecar):
        destination = output / source_path.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            os.link(source_path, destination)
        if not destination.samefile(source_path):
            raise ValueError(f'Subset is not byte-identical to audited parent: {destination}')
    return 'kept'


def prepare(args: argparse.Namespace) -> None:
    source, output = args.source.resolve(), args.output.resolve()
    if output == source or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('Subset output must be separate from the active source root')
    plan = source / 'text_data_plan.jsonl'
    if digest(plan.read_bytes()) != PARENT_SHA:
        raise ValueError('Active parent plan changed; repeat ID matching instead of assuming row positions')
    audit = json.loads((source / 'text_data_audit.json').read_text())
    if not (source / 'TEXT_DATA_VERIFIED').exists() or not audit['passed'] or audit['plan_sha256'] != PARENT_SHA:
        raise ValueError('Parent full tensor audit is missing or stale')
    specs = []
    for index, line in enumerate(plan.open()):
        spec = json.loads(line)
        rel = Path('preprocessed') / spec['split'] / f'worker_{index % 8:02d}' / f'data_{index // 8:06d}.parquet'
        specs.append(dict(spec, conditioned_relative_path=str(rel)))
    external = [json.loads(line) for line in args.attachment.open() if line.strip()]
    matches, unmatched, mapping = match_rows(specs, external)
    excluded = [dict(row, external_pose_qc_matches=matches[row['id']]) for row in specs if row['id'] in matches]
    kept = [row for row in specs if row['id'] not in matches]
    if len(specs) != 115905 or len(excluded) != 881:
        raise ValueError('Not the user-confirmed 115905 -> 115024, excluding881 selection')
    counts = dict(Counter(row['split'] for row in kept))
    if counts != dict(train=112300, validation=2724):
        raise ValueError('Unexpected retained split counts')
    output.mkdir(parents=True, exist_ok=True)
    if any((output / 'runs/train/checkpoints').glob('checkpoint-*')):
        raise ValueError('Subset already trained; never rewrite its frozen plan')
    write_jsonl(output / 'excluded_ids.jsonl', excluded)
    write_jsonl(output / 'external_unmatched.jsonl', unmatched)
    write_jsonl(output / 'external_id_mapping.jsonl', mapping)
    write_jsonl(output / 'text_data_plan.jsonl', kept)
    exclusion_sha = digest((output / 'excluded_ids.jsonl').read_bytes())
    plan_sha = digest((output / 'text_data_plan.jsonl').read_bytes())
    save_json(
        output / 'selection.json',
        dict(authorization='user_20261027_all881_source_ids_including55_validation_and_cross_length',
             attachment=str(args.attachment),
             attachment_sha256=digest(args.attachment.read_bytes()),
             parent_root=str(source),
             parent_plan_sha256=PARENT_SHA,
             parent_rows=len(specs),
             excluded=881,
             excluded_by_split=dict(Counter(row['split'] for row in excluded)),
             kept=len(kept),
             kept_by_split=counts,
             excluded_manifest_sha256=exclusion_sha,
             plan_sha256=plan_sha,
             source_files_deleted=False,
             cross_length_source_id_exclusion=True))
    jobs = ((row, str(source), str(output), row['id'] in matches) for row in specs)
    with ThreadPoolExecutor(max_workers=8) as pool:
        for done, _ in enumerate(pool.map(linked_pair, jobs), 1):
            if done % 5000 == 0 or done == len(specs):
                save_json(output / 'prepare_progress.json', dict(checked=done, total=len(specs)))
                print('SUBSET_PREPARE', done, len(specs), flush=True)
    # No map_style_cache is copied: its indices belong to the previous dataset.
    paths = list((output / 'preprocessed').rglob('*.parquet'))
    if len(paths) != len(kept):
        raise ValueError('Unexpected missing or extra files in retained training tree')
    existing = set(str(path.relative_to(output)) for path in paths)
    if existing != {row['conditioned_relative_path'] for row in kept}:
        raise ValueError('Excluded IDs leaked into derived dataset')
    proof = dict(passed=True,
                 audit_mode='unchanged_hardlink_subset_of_full_tensor_audited_parent',
                 full_tensor_reaudit=False,
                 parent_full_tensor_audit=str(source / 'text_data_audit.json'),
                 parent_audit_sha256=digest((source / 'text_data_audit.json').read_bytes()),
                 parent_plan_sha256=PARENT_SHA,
                 plan_sha256=plan_sha,
                 rows=len(kept),
                 split_counts=counts,
                 retained_parquet_and_sidecars_verified_same_inode=True,
                 all_sidecar_ids_and_windows_checked=True,
                 excluded881_parquet_ids_captions_windows_checked=True,
                 excluded_manifest_sha256=exclusion_sha)
    save_json(output / 'subset_data_audit.json', proof)
    save_json(
        output / 'data_plan_summary.json',
        dict(total=len(kept),
             split_counts=counts,
             source_counts=dict(Counter(row['source_name'] for row in kept)),
             source_by_split={
                 split: dict(Counter(row['source_name'] for row in kept if row['split'] == split))
                 for split in ('train', 'validation')
             },
             excluded_source_counts=dict(Counter(row['source_name'] for row in excluded)),
             matched_external_files=sum(bool(row['matched_ids']) for row in mapping),
             unmatched_external_files=len(unmatched),
             plan_sha256=plan_sha))
    config = yaml.safe_load((source / 'training_config.yaml').read_text())
    # Reproduce the original base + parent initialization, then full-state
    # resume below. This also preserves any inherited frozen base values not
    # present in trainable-only DCP model snapshots.
    config['models']['student']['init_from_training_checkpoint'] = (
        '/datacc05/lizihao/worldmodel/fastvideowm_scene_text_20261022/runs/train/checkpoints/checkpoint-100')
    config['training']['data']['data_path'] = str(output / 'preprocessed/train')
    config['callbacks']['worldmodel_validation']['data_path'] = str(output / 'preprocessed/validation')
    config['callbacks']['text_input_guard']['output_dir'] = str(output / 'runs/train')
    config['training']['checkpoint']['output_dir'] = str(output / 'runs/train/checkpoints')
    config['training']['checkpoint']['reset_dataloader_on_data_change'] = True
    config['training']['checkpoint']['resume_from_checkpoint'] = 'MUST_BE_BOUND_TO_COMPLETE_CHECKPOINT'
    config['training']['tracker']['run_name'] = 'h3_scene_subject_poseqc_115024_resume_20261027'
    if config['training']['loop'] != dict(max_train_steps=3000, gradient_accumulation_steps=8):
        raise ValueError('Training target or accumulation changed unexpectedly')
    (output / 'training_config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    (output / 'SUBSET_DATA_VERIFIED').touch()
    print('SUBSET_READY', json.dumps(proof), flush=True)


def complete_checkpoints(source: Path, minimum: int) -> list[Path]:
    result = []
    for path in (source / 'runs/train/checkpoints').glob('checkpoint-*'):
        step = int(path.name.split('-')[-1])
        if step < minimum or not (path / 'metadata.json').is_file() or not (path / 'dcp/.metadata').is_file():
            continue
        if len(list((path / 'dcp').glob('*.distcp'))) != 8:
            continue
        if not all((path / f'rng_state_rank{rank}.pt').is_file() for rank in range(8)):
            continue
        # RNG files must have settled; no checkpoint save is interrupted.
        files = [p for p in path.rglob('*') if p.is_file()]
        if any(p.stat().st_size == 0 for p in files) or max(p.stat().st_mtime for p in files) > time.time() - 30:
            continue
        result.append(path)
    return sorted(result, key=lambda p: int(p.name.split('-')[-1]))


def switch(args: argparse.Namespace) -> None:
    source, output = args.source.resolve(), args.output.resolve()
    proof = json.loads((output / 'subset_data_audit.json').read_text())
    if not (output / 'SUBSET_DATA_VERIFIED').exists() or not proof['passed']:
        raise ValueError('Derived subset verification missing')
    if digest((output / 'text_data_plan.jsonl').read_bytes()) != proof['plan_sha256']:
        raise ValueError('Subset plan changed after verification')
    if digest((output / 'excluded_ids.jsonl').read_bytes()) != proof['excluded_manifest_sha256']:
        raise ValueError('Authorized exclusions changed')
    lock = (output / 'switch.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    process = Path(f'/proc/{args.old_torchrun_pid}')
    start = time.monotonic()
    while not complete_checkpoints(source, args.minimum_checkpoint):
        if not process.exists() or time.monotonic() - start > 4 * 3600:
            raise RuntimeError('Old job stopped before requested complete checkpoint, or wait exceeded4h')
        print('WAITING_FOR_COMPLETE_CHECKPOINT', args.minimum_checkpoint, flush=True)
        time.sleep(300)
    checkpoint = complete_checkpoints(source, args.minimum_checkpoint)[-1]
    command = (process / 'cmdline').read_bytes().replace(b'\0', b' ').decode()
    if (process.stat().st_uid != os.getuid() or 'torch.distributed.run' not in command
            or str(source / 'training_config.yaml') not in command):
        raise ValueError('Old PID ownership/command changed; refusing to signal')
    # Bind an explicit, complete checkpoint BEFORE stopping the original job.
    config_path = output / 'training_config.yaml'
    config = yaml.safe_load(config_path.read_text())
    config['training']['checkpoint']['resume_from_checkpoint'] = str(checkpoint)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    save_json(
        output / 'switch_state.json',
        dict(stage='stopping_old_train',
             checkpoint=str(checkpoint),
             old_pid=args.old_torchrun_pid,
             target_steps=3000,
             total_bs=32))
    save_json(
        source.parent / 'SUPERSEDED_BY_POSE_QC_20261027.json',
        dict(new_data_root=str(output),
             checkpoint=str(checkpoint),
             reason='user excluded881 source IDs',
             source_files_deleted=False))
    os.kill(args.old_torchrun_pid, signal.SIGINT)
    for _ in range(120):
        if not process.exists():
            break
        time.sleep(5)
    else:
        raise RuntimeError('Old torchrun did not stop gracefully; no force-kill or second trainer')
    for _ in range(288):
        check = subprocess.run([str(REPO / '.venv/bin/python'), 'examples/train/check_minimax_h3_gpus_idle.py'],
                               cwd=REPO,
                               env=dict(os.environ, CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'),
                               check=False)
        if check.returncode == 0:
            break
        time.sleep(300)
    else:
        raise RuntimeError('GPUs not idle; no other-user process was terminated')
    env = dict(os.environ,
               CONFIG_FILE=str(config_path),
               TRAIN_DATA=str(output / 'preprocessed/train'),
               VALIDATION_DATA=str(output / 'preprocessed/validation'),
               LOG_DIR=str(output / 'runs/train/logs'),
               NUM_GPUS='8',
               CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',
               MASTER_PORT='30730',
               WANDB_MODE='offline',
               TORCHDYNAMO_DISABLE='1',
               TORCH_COMPILE_DISABLE='1')
    save_json(
        output / 'switch_state.json',
        dict(stage='launching_filtered_resume',
             checkpoint=str(checkpoint),
             target_steps=3000,
             total_bs=32,
             reset_loader=True))
    print('LAUNCH_FILTERED_RESUME', str(checkpoint), flush=True)
    subprocess.run(['bash', 'examples/train/launch_minimax_h3_worldmodel_stage0.sh'], cwd=REPO, env=env, check=True)
    (output / 'TRAINING_COMPLETE').touch()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'switch'])
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--attachment', type=Path, default=ATTACHMENT)
    parser.add_argument('--old-torchrun-pid', type=int, default=2276631)
    parser.add_argument('--minimum-checkpoint', type=int, default=1100)
    args = parser.parse_args()
    try:
        globals()[args.command](args)
    except Exception as error:
        if args.output.exists():
            save_json(args.output / 'NEEDS_ATTENTION.json', dict(command=args.command, error=repr(error)))
        raise


if __name__ == '__main__':
    main()
