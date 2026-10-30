"""Immutable subset, exact native loader cache, and full-resume recipe tests."""
import ast
from collections.abc import Sequence
import json
import logging
import os
from pathlib import Path
import pickle

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tools.prepare_cosmos_qc_resume_20261027 import link_and_validate, make_config, safe_relative, write_cache


def sample(source, *, count=1):
    rel = Path('preprocessed/train/worker_00/data_000003.parquet')
    path = source / rel
    path.parent.mkdir(parents=True)
    pq.write_table(pa.table({'id': ['test'] * count}), path)
    row = dict(id='test', split='train', caption_sha256='abc', video='/readonly/video.mp4',
               source_frame_indices=list(range(124)), conditioned_relative_path=str(rel))
    path.with_suffix('.audit.json').write_text(json.dumps(row))
    info = path.stat()
    audit = dict(id='test', reasons=[], conditioned_relative_path=str(rel),
                 file_size=info.st_size, mtime_ns=info.st_mtime_ns)
    return row, audit


def test_subset_hardlinks_preserve_parent_and_recheck_actual_rows(tmp_path):
    source, output = tmp_path / 'source', tmp_path / 'new'
    row, audit = sample(source)
    path, count = link_and_validate((row, audit, source, output))
    assert count == 1
    assert Path(path).samefile(source / row['conditioned_relative_path'])
    assert not Path(path).is_symlink()
    assert pq.ParquetFile(path).metadata.num_rows == 1


def test_source_mutation_since_qc_is_refused(tmp_path):
    source = tmp_path / 'source'
    row, audit = sample(source)
    audit['mtime_ns'] -= 1
    with pytest.raises(ValueError, match='changed after numeric QC'):
        link_and_validate((row, audit, source, tmp_path / 'new'))


def test_multisample_shards_cannot_be_mistaken_for_one_row(tmp_path):
    source = tmp_path / 'source'
    row, audit = sample(source, count=2)
    with pytest.raises(ValueError, match='exactly one'):
        link_and_validate((row, audit, source, tmp_path / 'new'))


@pytest.mark.parametrize('value', ['/tmp/video.parquet', '../train/x.parquet',
                                   'preprocessed/validation/worker_00/data_000000.parquet'])
def test_path_traversal_and_wrong_split_refused(value):
    with pytest.raises(ValueError):
        safe_relative(dict(split='train', conditioned_relative_path=value))


def test_cache_binds_new_paths_not_parent_order_or_paths(tmp_path):
    root = tmp_path / 'preprocessed/train'
    root.mkdir(parents=True)
    p1, p2 = root / 'b.parquet', root / 'a.parquet'
    p1.touch()
    p2.touch()
    proof = write_cache(root, [(str(p1), 1), (str(p2), 1)])
    with Path(proof['path']).open('rb') as handle:
        paths, counts = pickle.load(handle)
    assert paths == (str(p2), str(p1))
    assert counts == (1, 1)
    assert proof['rows'] == proof['files'] == 2
    with pytest.raises(ValueError):
        write_cache(root, [(str(tmp_path / 'parent.parquet'), 1)])


@pytest.mark.parametrize('rank', [0, 7])
def test_actual_native_loader_accepts_offline_cache_without_metadata_scan(tmp_path, rank):
    # Exercise the actual trusted loader functions, not a reimplementation;
    # importing the full fastvideo package would require CUDA attention kernels.
    path = Path(__file__).parents[2] / 'fastvideo/dataset/parquet_dataset_map_style.py'
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and
             n.name in {'_parse_data_path_specs', 'get_parquet_files_and_length'}]
    class Group:
        def __init__(self):
            self.calls = 0

        def barrier(self):
            self.calls += 1

    group = Group()
    namespace = dict(os=os, pickle=pickle, Sequence=Sequence, get_world_rank=lambda: rank,
                     get_world_group=lambda: group, logger=logging.getLogger('cache_test'))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    root = tmp_path / 'new/preprocessed/train'
    root.mkdir(parents=True)
    p = root / 'x.parquet'
    p.touch()
    # No pq/tqdm injected: test fails if the actual loader tries a footer scan.
    write_cache(root, [(str(p), 1)])
    paths, counts = namespace['get_parquet_files_and_length'](str(root))
    assert tuple(paths) == (str(p),) and tuple(counts) == (1,)
    assert group.calls == 1


def test_recipe_keeps_parent_frozen_bootstrap_full1100_and_total_bs32(tmp_path):
    cfg = dict(models={'student': dict(vision_only_conditioning=False, normalize_camera_translation=False,
                                      init_from_training_checkpoint='/parent/checkpoint-100')},
               training=dict(loop=dict(max_train_steps=3000, gradient_accumulation_steps=8),
                             distributed=dict(num_gpus=8, sp_size=2, tp_size=1),
                             data=dict(train_batch_size=1, data_path='/old'),
                             checkpoint=dict(checkpoints_total_limit=35), tracker={}),
               callbacks=dict(worldmodel_validation={}, text_input_guard={}),
               method=dict(camera_ranking_weight=0.))
    checkpoint = tmp_path / 'checkpoint-1100'
    result = make_config(cfg, tmp_path / 'new', checkpoint)
    assert result['models'] == cfg['models']
    assert result['method'] == cfg['method']
    assert result['training']['loop'] == cfg['training']['loop']
    assert result['training']['checkpoint']['resume_from_checkpoint'] == str(checkpoint)
    assert result['training']['checkpoint']['reset_dataloader_on_data_change']
    assert result['training']['checkpoint']['checkpoints_total_limit'] == 35
    assert cfg['training']['data']['data_path'] == '/old'
