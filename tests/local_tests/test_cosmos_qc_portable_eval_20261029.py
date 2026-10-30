from copy import deepcopy
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from tools.prepare_cosmos_qc_portable_eval_20261029 import check_condition, condition_plan, idle, tensor, text_hash


def put(row, name, value):
    value = np.asarray(value, dtype=np.float32)
    row.update({f'{name}_bytes': value.tobytes(), f'{name}_shape': list(value.shape),
                f'{name}_dtype': str(value.dtype)})


def fixture(group='single_action'):
    case = dict(group=group, id='js_00001', sample_id='js_00001', prompt='A forest road and a moving tank.')
    row = dict(caption=case['prompt'])
    put(row, 'text_embedding', np.ones((4, 5120)))
    put(row, 'text_token_tags', [0, 0, 1, 1])
    put(row, 'first_frame_latent', np.zeros((24, 1, 30, 52)))
    put(row, 'vae_latent', np.ones((24, 37, 30, 52)) if group == 'trajectory_gt' else np.zeros((24, 37, 30, 52)))
    put(row, 'camera_intrinsics', np.broadcast_to(np.eye(3), (37, 3, 3)))
    put(row, 'camera_to_world', np.broadcast_to(np.eye(4), (37, 4, 4)))
    encoding = dict(prompt_sha256=text_hash(case['prompt']), first_frame_png_sha256='image_hash',
                    embedding_sha256=hashlib.sha256(row['text_embedding_bytes']).hexdigest())
    return case, row, encoding, deepcopy(row)


def test_distinct_group_ids_and_shared_alpha_conditions():
    cases = [dict(group=group, id='js_00001', sample_id='js_00001', prompt=f'{group} prompt', alpha=alpha)
             for alpha in (1., .5) for group in ('trajectory_gt', 'single_action')]
    original = deepcopy(cases)
    generated, plan = condition_plan(cases, Path('/new/output'))
    assert cases == original
    assert len(plan) == 2
    assert len({case['condition_parquet'] for case in generated}) == 2
    assert all(case['text_condition_used'] for case in generated)
    assert generated[0]['condition_parquet'] == generated[2]['condition_parquet']
    assert generated[0]['condition_parquet'] != generated[1]['condition_parquet']


def test_full_protocol_has_64_independent_conditions():
    cases = []
    for alpha in (1., .5):
        for i in range(1, 17):
            sample = f'js_{i:05d}'
            for group in ('trajectory_gt', 'single_action'):
                cases.append(dict(group=group, id=sample, sample_id=sample, prompt=f'{group} prompt', alpha=alpha))
            for direction in ('forward', 'backward'):
                cases.append(dict(group='single_action_text', id=f'{sample}_{direction}', sample_id=sample,
                                  prompt=f'A tank moves {direction}.', alpha=alpha))
    generated, plan = condition_plan(cases, Path('/new/output'))
    assert len(generated) == 128 and len(plan) == 64
    assert len({c['condition_parquet'] for c in generated}) == 64


def test_conflicting_alpha_prompts_and_unsafe_ids_are_rejected():
    a = dict(group='single_action', id='js_00001', sample_id='js_00001', prompt='first')
    with pytest.raises(ValueError, match='Alpha variants disagree'):
        condition_plan([a, dict(a, prompt='different')], Path('/output'))
    with pytest.raises(ValueError, match='Unsafe'):
        condition_plan([dict(a, id='../bad')], Path('/output'))


@pytest.mark.parametrize('group', ['trajectory_gt', 'single_action', 'single_action_text'])
def test_actual_native_targets_and_text_pass(group):
    case, row, encoding, base = fixture(group)
    result = check_condition(case, row, encoding, base, 'image_hash')
    assert result['passed'] and result['text_tokens'] == 2 and result['vision_tokens'] == 2
    assert result['gt_target'] == (group == 'trajectory_gt')


def test_gt_must_use_native_gt_not_simple_shape_carrier():
    case, row, encoding, base = fixture('trajectory_gt')
    put(row, 'vae_latent', np.zeros((24, 37, 30, 52)))
    with pytest.raises(ValueError, match='Native source tensor changed: vae_latent'):
        check_condition(case, row, encoding, base, 'image_hash')


def test_synthetic_rows_cannot_leak_future_gt():
    case, row, encoding, base = fixture('single_action')
    put(row, 'vae_latent', np.ones((24, 37, 30, 52)))
    base = deepcopy(row)
    with pytest.raises(ValueError, match='future GT'):
        check_condition(case, row, encoding, base, 'image_hash')


@pytest.mark.parametrize('problem', ['caption', 'image', 'embedding', 'tags'])
def test_semantic_and_actual_embedding_bindings_are_required(problem):
    case, row, encoding, base = fixture()
    if problem == 'caption':
        row['caption'] = 'A video scene observed from a moving camera.'
    elif problem == 'image':
        encoding['first_frame_png_sha256'] = 'another_image'
    elif problem == 'embedding':
        encoding['embedding_sha256'] = 'wrong_hash'
    else:
        put(row, 'text_token_tags', [0, 0, 0, 0])
    with pytest.raises(ValueError):
        check_condition(case, row, encoding, base, 'image_hash')


def test_nonfinite_and_malformed_tensor_are_rejected():
    row = {}
    put(row, 'sample', [np.nan])
    with pytest.raises(ValueError, match='Nonfinite'):
        tensor(row, 'sample')
    put(row, 'sample', [1])
    row['sample_shape'] = [2]
    with pytest.raises(ValueError):
        tensor(row, 'sample')


def test_idle_check_only_queries_four_visible_gpus(monkeypatch):
    queried = []

    def mem_get_info(index):
        queried.append(index)
        return 79 * 2**30, 80 * 2**30

    cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 4, mem_get_info=mem_get_info)
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=cuda))
    idle(None)
    assert queried == [0, 1, 2, 3]


def test_idle_check_rejects_busy_selected_gpu(monkeypatch):
    cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 4,
                           mem_get_info=lambda index: (60 * 2**30, 80 * 2**30))
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=cuda))
    with pytest.raises(RuntimeError, match='occupied'):
        idle(None)


def test_four_gpu_launcher_never_selects_gpu4_through_gpu7():
    script = (Path(__file__).resolve().parents[2] / 'tools/run_cosmos_qc_portable_eval_20261029.sh').read_text()
    assert 'export CUDA_VISIBLE_DEVICES=0,1,2,3' in script
    assert '--nproc_per_node=4' in script and '--inference-num-gpus 4' in script
    assert 'for worker in 0 1; do' in script and '--workers 2' in script
    assert 'for worker in 0 1 2 3; do' in script and '--workers 4' in script
    assert 'check_minimax_h3_gpus_idle.py' not in script
    assert '--nproc_per_node=8' not in script
    assert 'CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7' not in script
