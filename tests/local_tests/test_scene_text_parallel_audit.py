import hashlib
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tools.audit_scene_text_20261022 import audit_root


def create_shard(root, sample_id, split='train', filename='data_000000.parquet', corruption=None):
    folder = root / 'preprocessed' / split / 'worker_00'
    folder.mkdir(parents=True, exist_ok=True)
    caption = 'A vehicle moves across a road.'
    spec = dict(id=sample_id, split=split, caption=caption,
                caption_sha256=hashlib.sha256(caption.encode()).hexdigest())
    row = dict(id=sample_id, caption=caption)
    metadata = dict(id=sample_id, caption_sha256=spec['caption_sha256'])
    arrays = {
        'vae_latent': (np.zeros((24, 37, 30, 52), np.float16), 'target_latent_sha256'),
        'first_frame_latent': (np.zeros((24, 1, 30, 52), np.float16), 'first_frame_latent_sha256'),
        'text_embedding': (np.zeros((2, 5120), np.float16), 'embedding_sha256'),
        'text_token_tags': (np.array([0, 1], np.int32), None),
        'camera_to_world': (np.repeat(np.eye(4, dtype=np.float32)[None], 37, axis=0), 'camera_sha256'),
        'camera_intrinsics': (np.eye(3, dtype=np.float32), 'intrinsics_sha256'),
    }
    for stem, (array, key) in arrays.items():
        payload = array.tobytes()
        row.update({stem + '_bytes': payload, stem + '_shape': list(array.shape), stem + '_dtype': str(array.dtype)})
        if key:
            metadata[key] = hashlib.sha256(payload).hexdigest()
    if corruption == 'hash':
        row['text_embedding_bytes'] = b'\x01' + row['text_embedding_bytes'][1:]
    if corruption == 'origin':
        camera = arrays['camera_to_world'][0].copy()
        camera[0, 0, 3] = 1
        row['camera_to_world_bytes'] = camera.tobytes()
    if corruption == 'text':
        row['text_token_tags_bytes'] = np.array([0, 0], np.int32).tobytes()
    path = folder / filename
    pq.write_table(pa.Table.from_pylist([row]), path)
    path.with_suffix('.audit.json').write_text(json.dumps(metadata))
    return spec


def write_plan(root, specs):
    (root / 'text_data_plan.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in specs))


def test_parallel_audit_reads_all_payloads_and_reduces_both_splits(tmp_path):
    specs = [create_shard(tmp_path, 'train-row'), create_shard(tmp_path, 'validation-row', 'validation')]
    write_plan(tmp_path, specs)
    audit_root(tmp_path, workers=2)
    report = json.loads((tmp_path / 'text_data_audit.json').read_text())
    assert report['rows'] == 2
    assert report['split_counts'] == {'train': 1, 'validation': 1}
    assert (tmp_path / 'TEXT_DATA_VERIFIED').is_file()


@pytest.mark.parametrize('corruption', ['hash', 'origin', 'text'])
def test_parallel_audit_rejects_corrupted_input_without_pass_marker(tmp_path, corruption):
    write_plan(tmp_path, [create_shard(tmp_path, 'bad-row', corruption=corruption)])
    with pytest.raises(ValueError):
        audit_root(tmp_path, workers=2)
    assert not (tmp_path / 'TEXT_DATA_VERIFIED').exists()


def test_duplicate_persisted_id_is_not_accepted_as_complete_coverage(tmp_path):
    first = create_shard(tmp_path, 'first')
    second = create_shard(tmp_path, 'second', filename='data_000001.parquet')
    create_shard(tmp_path, 'first', filename='data_000001.parquet')
    write_plan(tmp_path, [first, second])
    with pytest.raises(ValueError, match='Duplicate persisted ID'):
        audit_root(tmp_path, workers=2)
    assert not (tmp_path / 'TEXT_DATA_VERIFIED').exists()


@pytest.mark.parametrize('corrupt_window', [False, True])
def test_dual_union_audit_checks_window_and_source_indices(tmp_path, corrupt_window):
    spec = create_shard(tmp_path, 'one')
    spec.update(caption_policy='exact_window_dual_subject_motion_background_only_v1',
                caption_window='new-window', video='/source/video.mp4',
                source_frame_indices=list(range(124)), registry_sha256='registry-sha',
                source_parquet='/source/raw.parquet', source_row=0)
    path = tmp_path / 'preprocessed/train/worker_00/data_000000.parquet'
    row = pq.read_table(path).to_pylist()[0]
    row['source_frame_indices'] = spec['source_frame_indices']
    pq.write_table(pa.Table.from_pylist([row]), path)
    audit_path = path.with_suffix('.audit.json')
    metadata = json.loads(audit_path.read_text())
    metadata.update({key: spec[key] for key in ('caption_window', 'video', 'source_frame_indices',
                     'registry_sha256', 'source_parquet', 'source_row')})
    if corrupt_window:
        metadata['caption_window'] = 'stale-window'
    audit_path.write_text(json.dumps(metadata))
    write_plan(tmp_path, [spec])
    if corrupt_window:
        with pytest.raises(ValueError, match='provenance mismatch'):
            audit_root(tmp_path, workers=1)
        assert not (tmp_path / 'TEXT_DATA_VERIFIED').exists()
    else:
        audit_root(tmp_path, workers=1)
        report = json.loads((tmp_path / 'text_data_audit.json').read_text())
        assert report['plan_sha256'] == hashlib.sha256((tmp_path / 'text_data_plan.jsonl').read_bytes()).hexdigest()


@pytest.mark.parametrize('corrupt_component', [False, True])
def test_composed_caption_audit_requires_actual_background_and_component_hashes(tmp_path, corrupt_component):
    from tools.prepare_scene_subject_text_20261024 import compose_caption, COMPOSITION
    spec = create_shard(tmp_path, 'composed')
    background = 'A forest surrounds a sunlit dirt road.'
    subject = 'A red truck travels along the road.'
    spec.update(background_description=background, subject_description=subject,
                prompt_composition=COMPOSITION,
                component_background_sha256=hashlib.sha256(background.encode()).hexdigest(),
                component_subject_sha256=hashlib.sha256(subject.encode()).hexdigest(),
                caption=compose_caption(background, subject))
    spec['caption_sha256'] = hashlib.sha256(spec['caption'].encode()).hexdigest()
    path = tmp_path / 'preprocessed/train/worker_00/data_000000.parquet'
    row = pq.read_table(path).to_pylist()[0]
    row['caption'] = spec['caption']
    pq.write_table(pa.Table.from_pylist([row]), path)
    metadata_path = path.with_suffix('.audit.json')
    metadata = json.loads(metadata_path.read_text())
    metadata.update({key: spec[key] for key in ('caption_sha256', 'prompt_composition',
                     'component_background_sha256', 'component_subject_sha256')})
    if corrupt_component:
        metadata['component_background_sha256'] = 'stale-background'
    metadata_path.write_text(json.dumps(metadata))
    write_plan(tmp_path, [spec])
    if corrupt_component:
        with pytest.raises(ValueError, match='composition provenance'):
            audit_root(tmp_path, workers=1)
    else:
        audit_root(tmp_path, workers=1)
