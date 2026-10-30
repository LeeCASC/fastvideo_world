"""Native H3 adapter regressions; CPU only, no FastVideo CUDA imports."""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


@pytest.fixture(scope='module')
def adapter():
    path = Path(__file__).parents[2] / 'tools/filter_current_cosmos_pose_qc_20261027.py'
    spec = importlib.util.spec_from_file_location('native_cosmos_qc_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.load_reference()
    return module


def pose(count):
    return np.broadcast_to(np.eye(4), (count, 4, 4)).copy()


def test_h3_offsets_are_actual_nonuniform_not_wan(adapter):
    offsets = adapter.h3_offsets()
    assert len(offsets) == 37
    assert offsets[:8].tolist() == [0, 4, 8, 12, 16, 17, 21, 25]
    assert offsets[-1] == 123
    with pytest.raises(ValueError):
        adapter.h3_offsets(121)


def test_direct_dense_rebases_window_without_scale_fitting(adapter, tmp_path):
    p = pose(345)
    p[:, 0, 3] = np.arange(345) * .2
    path = tmp_path / 'camera.npz'
    np.savez(path, camera_to_world=p)
    actual = adapter.dense_camera(dict(pose_path=str(path), dense_kind='new_direct'), np.arange(174, 298))
    np.testing.assert_allclose(actual[0], np.eye(4), atol=1e-12)
    assert actual[-1, 0, 3] == pytest.approx(123 * .2)


def test_historical_w2c_basis_and_scale_reproduce_producer(adapter, tmp_path):
    p = pose(175)
    p[:, 1, 3] = np.linspace(0, 8, 175)
    path = tmp_path / 'camera.npz'
    np.savez(path, extrinsics=np.linalg.inv(p)[:, :3])
    actual = adapter.dense_camera(dict(pose_path=str(path), dense_kind='old_calibrated',
                                      source_family='h3_js_easy', translation_scale_xyz=[1, 2, 1]),
                                  np.arange(124))
    assert actual[-1, 1, 3] == pytest.approx(-2 * 123 / 174)


def test_source_dense_cannot_be_replaced_with_interpolation(adapter):
    assert adapter.dense_camera({}, np.arange(124)) is None


def test_reference_keeps_closed_loop_but_rejects_jitter(adapter):
    ref = adapter.load_reference()
    p = pose(124)
    t = np.linspace(0, np.pi * 2, 124)
    p[:, 0, 3] = np.cos(t)
    p[:, 2, 3] = np.sin(t)
    assert not ref.dense_jitter_rejected(ref.motion_stats(p))
    p[:, 0, 3] = np.arange(124) % 2
    p[:, 2, 3] = 0
    assert ref.dense_jitter_rejected(ref.motion_stats(p))


def make_sample(adapter, tmp_path, *, repeat=False, mismatch=False):
    indices = np.rint(np.arange(124) * .625).astype(int) if repeat else np.arange(124)
    p = pose(124)
    p[:, 2, 3] = np.arange(124) * .01
    pose_path = tmp_path / 'source.npz'
    np.savez(pose_path, camera_to_world=p)
    anchors = adapter.relative(p[indices])[adapter.h3_offsets()].astype(np.float32)
    if mismatch:
        anchors[-1, 0, 3] += .1
    latent = np.zeros((24, 37, 30, 52), np.float32)
    row = dict(id='test/001', source_name='test', caption='An open plain.', num_frames=124,
               fps=24., source_fps=15. if repeat else 24., source_frame_indices=indices.tolist())
    for name, value in [('vae_latent', latent), ('camera_to_world', anchors)]:
        row.update({name + '_bytes': value.tobytes(), name + '_shape': list(value.shape),
                    name + '_dtype': str(value.dtype)})
    path = tmp_path / 'preprocessed/train/worker_00/data_000000.parquet'
    path.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([row]), path)
    path.with_suffix('.audit.json').write_text(json.dumps({
        'target_latent_sha256': hashlib.sha256(latent.tobytes()).hexdigest(),
        'camera_sha256': hashlib.sha256(anchors.tobytes()).hexdigest()}))
    descriptor = dict(id=row['id'], source_name='test', split='train',
                      conditioned_relative_path=str(path.relative_to(tmp_path)),
                      source_frame_indices=indices.tolist(),
                      caption_sha256=hashlib.sha256(row['caption'].encode()).hexdigest(),
                      pose_path=str(pose_path), dense_kind='new_direct')
    adapter.INPUT_ROOT = tmp_path
    return descriptor


def test_real_native_parquet_passes(adapter, tmp_path):
    descriptor = make_sample(adapter, tmp_path)
    result = adapter.inspect(descriptor)
    assert result['reasons'] == []
    assert result['dense_pose_available']
    assert result['dense_cache_max_abs_error'] < 1e-6


def test_repeated_source_indices_are_valid_resampling_not_corruption(adapter, tmp_path):
    result = adapter.inspect(make_sample(adapter, tmp_path, repeat=True))
    assert result['reasons'] == []
    assert result['unique_source_frames'] == 78


def test_dense_vs_actual_anchor_mismatch_is_not_hidden_by_scale_fit(adapter, tmp_path):
    result = adapter.inspect(make_sample(adapter, tmp_path, mismatch=True))
    assert 'dense_training_pose_mismatch' in result['reasons']


def test_damaged_latent_is_rejected(adapter, tmp_path):
    descriptor = make_sample(adapter, tmp_path)
    path = tmp_path / descriptor['conditioned_relative_path']
    row = pq.read_table(path).to_pylist()[0]
    latent = np.zeros((24, 37, 30, 52), np.float32)
    latent[0, 0, 0, 0] = np.nan
    row['vae_latent_bytes'] = latent.tobytes()
    pq.write_table(pa.Table.from_pylist([row]), path)
    result = adapter.inspect(descriptor)
    assert 'inspection_error' in result['reasons']
    assert 'finite' in result['error']
