"""CPU regressions for the opt-in scene-text continuation input contract."""
from types import SimpleNamespace

import pytest
import torch

from tools.prepare_text_training_20261022 import clean_caption
import tools.text_condition_training_guard_20261022 as guard_module


def test_camera_language_does_not_enter_scene_caption():
    assert clean_caption('A tank rolls on a road. The camera pans left. Trees are green.') == (
        'A tank rolls on a road. Trees are green.')


@pytest.mark.parametrize('vision_only,neutralized', [(True, False), (False, True)])
def test_training_guard_rejects_text_disabled(tmp_path, vision_only, neutralized):
    guard = guard_module.TextConditionTrainingGuard(str(tmp_path))
    student = SimpleNamespace(vision_only_conditioning=vision_only, neutralize_text_conditioning=neutralized)
    with pytest.raises(ValueError, match='actual, non-neutralized'):
        guard.on_train_start(SimpleNamespace(student=student))


@pytest.mark.parametrize('tags,extra_source_tags,should_pass', [
    ([0, 0, 1], 0, True), ([0, 0, 0], 0, False), ([0, 0, 1], 1, False),
])
def test_training_guard_checks_actual_packed_tags(tmp_path, monkeypatch, tags, extra_source_tags, should_pass):
    monkeypatch.setattr(guard_module, 'get_world_rank', lambda: 0)
    layout = SimpleNamespace(token_tags=torch.tensor(tags), text_indices=torch.arange(3))
    prepared = SimpleNamespace(minimax_h3_layout=layout)
    student = SimpleNamespace(vision_only_conditioning=False, neutralize_text_conditioning=False,
        prepare_batch=lambda raw, **kwargs: prepared)
    guard = guard_module.TextConditionTrainingGuard(str(tmp_path))
    guard.on_train_start(SimpleNamespace(student=student))
    raw = {'text_attention_mask': torch.ones(1, 3),
           'text_token_tags': torch.tensor([tags + [1]*extra_source_tags])}
    if should_pass:
        assert student.prepare_batch(raw) is prepared
        assert guard.text_tokens == 1
        assert (tmp_path / 'training_text_input_audit.jsonl').is_file()
    else:
        with pytest.raises(ValueError, match='Text missing'):
            student.prepare_batch(raw)
