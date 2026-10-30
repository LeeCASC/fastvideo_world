import json
import importlib.util
import logging
from pathlib import Path
import sys
import types

import pytest
import torch
import torch.distributed.checkpoint as dcp

from tools.exclude_external_pose_qc_20261027 import attachment_key, match_rows


@pytest.fixture
def ck(monkeypatch):
    # The public fastvideo package imports CUDA attention kernels eagerly.
    # Load the actual checkpoint implementation alone for CPU DCP testing;
    # only logging is stubbed, no checkpoint/load/state behavior is mocked.
    logger = types.ModuleType('fastvideo.logger')
    logger.init_logger = logging.getLogger
    monkeypatch.setitem(sys.modules, 'fastvideo.logger', logger)
    path = Path(__file__).resolve().parents[2] / 'fastvideo/train/utils/checkpoint.py'
    spec = importlib.util.spec_from_file_location('_cpu_checkpoint_policy', path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_aliases_match_old_and_new_native_ids_but_not_other_sources():
    rows = [dict(id='h3_source/000001_abcdef', original_global_id='h3_js_easy:video_D', source_name='h3_source')]
    external = [dict(relative_path='121f/h3_source/h3_js_easy__video_D.pt'),
                dict(relative_path='61f/h3_source/h3_source__000001_abcdef__w000__' + 'a' * 40 + '.pt'),
                dict(relative_path='61f/other_source/h3_js_easy__video_D.pt')]
    matches, unmatched, mapping = match_rows(rows, external)
    assert len(matches) == 1 and len(matches[rows[0]['id']]) == 2
    assert len(unmatched) == 1 and len(mapping) == 3


@pytest.mark.parametrize('value', ['/tmp/source/item.pt', '61f/source/../../../item.pt',
                                   '61f/source/item.sh', '999f/source/item.pt'])
def test_attachment_cannot_supply_execution_or_deletion_targets(value):
    with pytest.raises(ValueError):
        attachment_key(value)


def test_ambiguous_alias_rejected():
    rows = [dict(id=f'x/{i}', original_global_id='same:video', source_name='x') for i in range(2)]
    with pytest.raises(ValueError, match='Ambiguous'):
        match_rows(rows, [dict(relative_path='121f/x/same__video.pt')])


class Loader:
    def __init__(self):
        self.cursor = torch.tensor(0)

    def state_dict(self):
        return {'cursor': self.cursor}

    def load_state_dict(self, value):
        self.cursor = value['cursor']


class Method:
    def __init__(self):
        self.weight = torch.tensor(0.)
        self.optimizer_momentum = torch.tensor(0.)

    def checkpoint_state(self):
        return {'weight': self.weight, 'optimizer_momentum': self.optimizer_momentum}


@pytest.mark.parametrize('reset,new_path,expected_cursor', [(False, '/filtered', 123),
                                                          (True, '/filtered', 0), (True, '/old', 123)])
def test_real_cpu_dcp_keeps_model_optimizer_step_and_resets_only_changed_loader(tmp_path, ck, reset, new_path,
                                                                            expected_cursor):
    parent = tmp_path / 'checkpoint-1100'
    parent.mkdir()
    loader = Loader()
    loader.cursor = torch.tensor(123)
    old_method = Method()
    old_method.weight.fill_(7.)
    old_method.optimizer_momentum.fill_(9.)
    dcp.save(dict(old_method.checkpoint_state(), dataloader=loader), checkpoint_id=str(parent / 'dcp'))
    (parent / 'metadata.json').write_text(json.dumps(dict(config={'training': {'data': {'data_path': '/old'}}})))
    new_loader, method = Loader(), Method()
    manager = ck.CheckpointManager(method=method, dataloader=new_loader, output_dir=str(tmp_path),
                                   config=ck.CheckpointConfig(save_steps=100, keep_last=35,
                                                              reset_dataloader_on_data_change=reset),
                                   raw_config={'training': {'data': {'data_path': new_path}}})
    assert manager.maybe_resume(resume_from_checkpoint=str(parent)) == 1100
    assert method.weight.item() == 7. and method.optimizer_momentum.item() == 9.
    assert new_loader.cursor.item() == expected_cursor


def test_changed_data_reset_requires_provenance(tmp_path, ck):
    parent = tmp_path / 'checkpoint-1100'
    (parent / 'dcp').mkdir(parents=True)
    (parent / 'metadata.json').write_text('{"config": {}}')
    manager = ck.CheckpointManager(method=Method(), dataloader=Loader(), output_dir=str(tmp_path),
                                   config=ck.CheckpointConfig(save_steps=100, keep_last=35,
                                                              reset_dataloader_on_data_change=True), raw_config={})
    with pytest.raises(ValueError, match='provenance'):
        manager.maybe_resume(resume_from_checkpoint=str(parent))
