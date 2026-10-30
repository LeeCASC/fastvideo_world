"""CPU-only fail-closed coverage for post-evaluation full-state continuation."""
from argparse import Namespace
from copy import deepcopy
import fcntl
import json
import os
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml

from tools import resume_cosmos_qc_after_eval_20261029 as resume


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    # Synthetic completed metadata must also pass the production settle window.
    settled = time.time() - 60
    os.utime(path, (settled, settled))


def recipe(root):
    return {
        'models': {'student': {
            'init_from': '/frozen/source/config/MiniMax-H3',
            'vision_only_conditioning': False,
            'init_from_training_checkpoint': '/immutable/parent/checkpoint-100',
            'lora_rank': 64,
        }},
        'training': {
            'distributed': {
                'num_gpus': 8, 'sp_size': 2, 'tp_size': 1,
                'hsdp_replicate_dim': 2, 'hsdp_shard_dim': 4,
            },
            'loop': {'max_train_steps': 3000, 'gradient_accumulation_steps': 8},
            'data': {
                'train_batch_size': 1,
                'data_path': str(root / 'preprocessed/train'),
                'validation_data_path': str(root / 'preprocessed/validation'),
                'num_workers': 4,
            },
            'checkpoint': {
                'resume_from_checkpoint': '/immutable/previous/checkpoint-1100',
                'output_dir': str(root / 'runs/train/checkpoints'),
                'checkpoints_total_limit': 35,
                'reset_dataloader_on_data_change': True,
            },
            'optimizer': {'learning_rate': 1e-5, 'betas': [0.9, 0.999]},
        },
        'callbacks': {'text_input_guard': {'enabled': True}},
        'method': {'camera_ranking_weight': 0.0},
    }


def checkpoint(root, config, step=1700):
    folder = root / f'runs/train/checkpoints/checkpoint-{step}'
    write_json(folder / 'metadata.json', {'step': step, 'config': config})
    (folder / 'dcp').mkdir()
    (folder / 'dcp/.metadata').write_bytes(b'metadata')
    for rank in range(8):
        (folder / f'dcp/__{rank}_0.distcp').write_bytes(b'shard')
        (folder / f'rng_state_rank{rank}.pt').write_bytes(b'rng')
    settled = time.time() - 60
    for path in folder.rglob('*'):
        if path.is_file():
            os.utime(path, (settled, settled))
    return folder


@pytest.fixture
def prepared(tmp_path):
    root = tmp_path / 'training'
    root.mkdir()
    config = recipe(root)
    (root / 'training_config.yaml').write_text(yaml.safe_dump(config))
    (root / 'text_data_plan.jsonl').write_text('{"id":"approved"}\n')
    caches = {}
    for split in ('train', 'validation'):
        path = root / f'preprocessed/{split}/map_style_cache/file_info.pkl'
        path.parent.mkdir(parents=True)
        path.write_bytes(b'precomputed-native-loader-index')
        caches[split] = {'path': str(path), 'sha256': resume.sha(path)}
    proof = {
        'passed': True,
        'split_counts': {'train': 105935, 'validation': 2504},
        'plan_sha256': resume.sha(root / 'text_data_plan.jsonl'),
        'training_config_sha256': resume.sha(root / 'training_config.yaml'),
        'caches': caches,
    }
    write_json(root / 'resume_data_audit.json', proof)
    write_json(root / 'RESUME_DATA_READY', {'proof_sha256': resume.sha(root / 'resume_data_audit.json')})
    saved = checkpoint(root, config)
    evaluation = tmp_path / 'evaluation'
    evaluation.mkdir()
    return root, config, saved, evaluation


def complete_evaluation(root):
    for marker in ('ALL_LATENTS_COMPLETE', 'VISUALIZATION_COMPLETE', 'INFERENCE_COMPLETE', 'CORE_METRICS_COMPLETE'):
        (root / marker).touch()
    cases = []
    for index in range(128):
        relative = f'alpha100/group/case_{index:03d}'
        folder = root / 'videos' / relative
        folder.mkdir(parents=True)
        (folder / 'VIDEO_COMPLETE').touch()
        (folder / 'raw.mp4').write_bytes(b'validated-video')
        cases.append({'output_key': relative})
    write_json(root / 'inference_manifest.json', cases)
    write_json(root / 'metrics.json', {
        'gt_cases': [{} for _ in range(32)],
        'no_reference_cases': [{} for _ in range(96)],
        'optional_metrics': {'vbench': {'status': 'unavailable', 'reason': 'Optional dependency not installed'}},
    })
    write_json(root / 'status.json', {'phase': 'complete_with_metric_limitations', 'videos_complete': 128})


def test_evaluation_gate_waits_until_markers_exist(tmp_path):
    assert not resume.evaluation_ready(tmp_path)
    (tmp_path / 'ALL_LATENTS_COMPLETE').touch()
    assert not resume.evaluation_ready(tmp_path)


def test_evaluation_gate_accepts_complete_128_videos_and_optional_unavailable(tmp_path):
    complete_evaluation(tmp_path)
    assert resume.evaluation_ready(tmp_path)


def test_evaluation_gate_refuses_active_failure_even_if_other_outputs_complete(tmp_path):
    complete_evaluation(tmp_path)
    (tmp_path / 'NEEDS_ATTENTION').touch()
    with pytest.raises(RuntimeError, match='Evaluation failed'):
        resume.evaluation_ready(tmp_path)


@pytest.mark.parametrize('corruption', ['missing_video_marker', 'empty_video', 'missing_case', 'duplicate_case',
                                        'gt_metrics', 'no_reference_metrics', 'video_count'])
def test_evaluation_gate_refuses_incomplete_outputs(tmp_path, corruption):
    complete_evaluation(tmp_path)
    folder = tmp_path / 'videos/alpha100/group/case_000'
    if corruption == 'missing_video_marker':
        (folder / 'VIDEO_COMPLETE').unlink()
    elif corruption == 'empty_video':
        (folder / 'raw.mp4').write_bytes(b'')
    elif corruption in {'missing_case', 'duplicate_case'}:
        path = tmp_path / 'inference_manifest.json'
        cases = json.loads(path.read_text())
        if corruption == 'missing_case':
            cases.pop()
        else:
            cases[-1] = cases[0]
        write_json(path, cases)
    elif corruption in {'gt_metrics', 'no_reference_metrics'}:
        path = tmp_path / 'metrics.json'
        metrics = json.loads(path.read_text())
        metrics['gt_cases' if corruption == 'gt_metrics' else 'no_reference_cases'].pop()
        write_json(path, metrics)
    else:
        write_json(tmp_path / 'status.json', {
            'phase': 'complete',
            'videos_complete': 127 if corruption == 'video_count' else 128,
        })
    with pytest.raises(ValueError):
        resume.evaluation_ready(tmp_path)


def test_evaluation_waits_for_atomic_final_status_after_completion_markers(tmp_path):
    complete_evaluation(tmp_path)
    write_json(tmp_path / 'status.json', {'phase': 'metrics', 'videos_complete': 128})
    assert not resume.evaluation_ready(tmp_path)
    write_json(tmp_path / 'status.json', {'phase': 'complete', 'videos_complete': 128})
    assert resume.evaluation_ready(tmp_path)


@pytest.mark.parametrize('unsafe', ['../outside', '/absolute/path'])
def test_evaluation_gate_rejects_path_traversal(tmp_path, unsafe):
    complete_evaluation(tmp_path)
    path = tmp_path / 'inference_manifest.json'
    cases = json.loads(path.read_text())
    cases[0]['output_key'] = unsafe
    write_json(path, cases)
    with pytest.raises(ValueError, match='Unsafe'):
        resume.evaluation_ready(tmp_path)


def test_latest_checkpoint_accepts_complete_1700_and_resume_pointer_difference(prepared):
    root, config, saved, _ = prepared
    metadata = json.loads((saved / 'metadata.json').read_text())
    metadata['config']['training']['checkpoint']['resume_from_checkpoint'] = '/older/checkpoint-1500'
    write_json(saved / 'metadata.json', metadata)
    assert resume.latest_checkpoint(root, config) == saved


@pytest.mark.parametrize('missing', ['dcp/.metadata', 'dcp/__7_0.distcp', 'rng_state_rank7.pt', 'metadata.json'])
def test_latest_checkpoint_ignores_newer_partial_save(prepared, missing):
    root, config, saved, _ = prepared
    newer = checkpoint(root, config, 1800)
    (newer / missing).unlink()
    assert resume.latest_checkpoint(root, config) == saved


def test_latest_checkpoint_ignores_empty_newer_shard_and_old_steps(prepared):
    root, config, saved, _ = prepared
    checkpoint(root, config, 1600)
    newer = checkpoint(root, config, 1800)
    (newer / 'dcp/__7_0.distcp').write_bytes(b'')
    assert resume.latest_checkpoint(root, config) == saved


@pytest.mark.parametrize('changed', ['data', 'optimizer', 'text', 'parent'])
def test_checkpoint_must_match_data_and_full_recipe(prepared, changed):
    root, config, saved, _ = prepared
    metadata = json.loads((saved / 'metadata.json').read_text())
    candidate = metadata['config']
    if changed == 'data':
        candidate['training']['data']['data_path'] = '/unapproved/data'
    elif changed == 'optimizer':
        candidate['training']['optimizer']['learning_rate'] *= 2
    elif changed == 'text':
        candidate['models']['student']['vision_only_conditioning'] = True
    else:
        candidate['models']['student']['init_from_training_checkpoint'] = '/different/parent'
    write_json(saved / 'metadata.json', metadata)
    with pytest.raises(ValueError, match='differs'):
        resume.latest_checkpoint(root, config)


def test_checkpoint_requires_eight_complete_rng_files(prepared):
    root, config, saved, _ = prepared
    (saved / 'rng_state_rank4.pt').write_bytes(b'')
    with pytest.raises(ValueError, match='No complete same-run checkpoint'):
        resume.latest_checkpoint(root, config)


def test_make_resume_config_changes_only_pointer_and_never_mutates_input(prepared):
    root, config, saved, _ = prepared
    before = deepcopy(config)
    expected = deepcopy(config)
    expected['training']['checkpoint']['resume_from_checkpoint'] = str(saved)
    actual = resume.make_resume_config(config, saved, root)
    assert actual == expected
    assert config == before
    # In this SP2 mesh, four data-parallel lanes x microbatch1 x accumulation8.
    dist = actual['training']['distributed']
    assert dist['num_gpus'] // dist['sp_size'] * actual['training']['data']['train_batch_size'] * \
        actual['training']['loop']['gradient_accumulation_steps'] == 32
    assert actual['training']['loop']['max_train_steps'] == 3000
    assert not actual['models']['student']['vision_only_conditioning']
    assert actual['models']['student']['init_from_training_checkpoint'].endswith('checkpoint-100')
    assert actual['training']['checkpoint']['reset_dataloader_on_data_change']
    assert actual['training']['optimizer'] == before['training']['optimizer']


def test_environment_exposes_exact_eight_gpus_and_original_data_roots(prepared, monkeypatch):
    root, _, _, _ = prepared
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0,1,2,3')
    monkeypatch.setenv('CUSTOM_PRESERVED_SETTING', 'yes')
    path = root / 'derived_resume.yaml'
    env = resume.environment(root, path)
    assert env['CUDA_VISIBLE_DEVICES'] == '0,1,2,3,4,5,6,7'
    assert env['NUM_GPUS'] == '8'
    assert env['CONFIG_FILE'] == str(path)
    assert env['TRAIN_DATA'] == str(root / 'preprocessed/train')
    assert env['VALIDATION_DATA'] == str(root / 'preprocessed/validation')
    assert env['LOG_DIR'] == str(root / 'runs/train/logs')
    assert env['WANDB_MODE'] == 'offline'
    assert env['TORCH_COMPILE_DISABLE'] == env['TORCHDYNAMO_DISABLE'] == '1'
    assert env['CUSTOM_PRESERVED_SETTING'] == 'yes'


def test_environment_overrides_wrong_inherited_model_path_and_accepts_frozen_model_path(prepared, monkeypatch):
    root, _, _, _ = prepared
    monkeypatch.setenv('MODEL_PATH', '/wrong/inherited/model')
    path = root / 'derived_resume.yaml'
    assert resume.environment(root, path)['MODEL_PATH'] == '/datacc05/lizihao_wm/worldmodel/ckpts/MiniMax-H3'
    assert resume.environment(root, path, '/frozen/config/model')['MODEL_PATH'] == '/frozen/config/model'


def files_snapshot(root):
    return {str(path.relative_to(root)): path.read_bytes() for path in root.rglob('*') if path.is_file()}


def test_check_mode_is_cpu_readonly_and_never_starts_any_subprocess(prepared, monkeypatch, capsys):
    root, _, saved, evaluation = prepared
    before = files_snapshot(root.parent)
    command = Mock(side_effect=AssertionError('CPU check must not start GPU/launcher subprocesses'))
    monkeypatch.setattr(resume.subprocess, 'run', command)
    resume.run(Namespace(training=root, evaluation=evaluation, check=True, max_wait_hours=24))
    result = json.loads(capsys.readouterr().out)
    assert result == {
        'checkpoint': str(saved), 'total_bs': 32, 'target_step': 3000,
        'evaluation_ready': False, 'no_gpu_started': True,
    }
    command.assert_not_called()
    assert files_snapshot(root.parent) == before


class StopAfterFirstPoll(Exception):
    """End a synthetic wait without sleeping or doing GPU work."""


def assert_first_poll_waits(prepared, monkeypatch):
    root, _, _, evaluation = prepared
    command = Mock(side_effect=AssertionError('No subprocess before all evaluation gates release'))
    monkeypatch.setattr(resume.subprocess, 'run', command)
    sleep = Mock(side_effect=StopAfterFirstPoll)
    monkeypatch.setattr(resume.time, 'sleep', sleep)
    with pytest.raises(StopAfterFirstPoll):
        resume.run(Namespace(training=root, evaluation=evaluation, check=False, max_wait_hours=24))
    command.assert_not_called()
    sleep.assert_called_once_with(300)
    state = json.loads((evaluation / 'autoresume_state.json').read_text())
    assert state['phase'] == 'waiting_for_evaluation'
    assert not (root / 'training_resume_after_eval_20261029.yaml').exists()
    assert not (root / 'POST_EVALUATION_RESUME_AUTHORIZED_20261029.json').exists()


def test_not_ready_wait_does_not_probe_gpus_or_start_training(prepared, monkeypatch):
    assert_first_poll_waits(prepared, monkeypatch)


def test_held_evaluation_lock_blocks_training_even_after_complete_markers(prepared, monkeypatch):
    _, _, _, evaluation = prepared
    complete_evaluation(evaluation)
    with (evaluation / 'job.lock').open('a') as active_evaluation:
        fcntl.flock(active_evaluation, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert_first_poll_waits(prepared, monkeypatch)


@pytest.mark.parametrize('changed_file', ['text_data_plan.jsonl', 'training_config.yaml',
                                         'preprocessed/train/map_style_cache/file_info.pkl'])
def test_data_proof_detects_prepared_inputs_mutation(prepared, changed_file):
    root, _, _, _ = prepared
    path = root / changed_file
    path.write_bytes(path.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='changed'):
        resume.validate_data_proof(root)


@pytest.mark.parametrize('trainer_code', [0, 7])
def test_run_full_resume_launches_only_after_idle_gate_and_preserves_recipe(prepared, monkeypatch, trainer_code):
    root, config, saved, evaluation = prepared
    complete_evaluation(evaluation)
    original = (root / 'training_config.yaml').read_bytes()
    monkeypatch.setenv('MODEL_PATH', '/wrong/inherited/model')
    monkeypatch.setattr(resume.os, 'statvfs', lambda _: SimpleNamespace(f_bavail=1024**3, f_frsize=1024))
    command = Mock(side_effect=[
        SimpleNamespace(returncode=0, stdout='All eight GPUs idle', stderr=''),
        SimpleNamespace(returncode=trainer_code),
    ])
    monkeypatch.setattr(resume.subprocess, 'run', command)
    sleep = Mock(side_effect=AssertionError('Ready evaluation and idle GPUs must launch without sleeping'))
    monkeypatch.setattr(resume.time, 'sleep', sleep)
    args = Namespace(training=root, evaluation=evaluation, check=False, max_wait_hours=24)
    if trainer_code:
        with pytest.raises(RuntimeError, match='Trainer exited with7'):
            resume.run(args)
    else:
        resume.run(args)

    assert command.call_count == 2
    idle_call, trainer_call = command.call_args_list
    assert idle_call.args[0] == [str(resume.REPO / '.venv/bin/python'),
                                 'examples/train/check_minimax_h3_gpus_idle.py']
    assert trainer_call.args[0] == ['bash', 'examples/train/launch_minimax_h3_worldmodel_stage0.sh']
    for call in (idle_call, trainer_call):
        assert call.kwargs['cwd'] == resume.REPO
        assert call.kwargs['env']['CUDA_VISIBLE_DEVICES'] == '0,1,2,3,4,5,6,7'
        assert call.kwargs['env']['NUM_GPUS'] == '8'
        assert call.kwargs['env']['TRAIN_DATA'] == str(root / 'preprocessed/train')
    derived = root / 'training_resume_after_eval_20261029.yaml'
    assert trainer_call.kwargs['env']['CONFIG_FILE'] == str(derived)
    assert trainer_call.kwargs['env']['MODEL_PATH'] == config['models']['student']['init_from']
    assert trainer_call.kwargs['check'] is False
    expected = deepcopy(config)
    expected['training']['checkpoint']['resume_from_checkpoint'] = str(saved)
    assert yaml.safe_load(derived.read_text()) == expected
    assert (root / 'training_config.yaml').read_bytes() == original
    # The provenance proof for the immutable original remains valid after launch.
    assert resume.validate_data_proof(root) == config
    authorization = json.loads((root / 'POST_EVALUATION_RESUME_AUTHORIZED_20261029.json').read_text())
    assert authorization['total_bs'] == 32
    assert authorization['target_step'] == 3000
    assert authorization['full_state_resume'] and authorization['same_dataset_restore_dataloader']
    state = json.loads((evaluation / 'autoresume_state.json').read_text())
    assert state['phase'] == ('launching_training' if trainer_code else 'training_process_exited_successfully')
    sleep.assert_not_called()
