import hashlib
import json

import pytest
import yaml

from tools.configure_union_dual_training_20261023 import configure


def prepare(root):
    variant = root / 'subject_motion'
    variant.mkdir()
    plan = variant / 'text_data_plan.jsonl'
    plan.write_text('{"fixture":"synthetic audited plan"}\n')
    registry = root / 'registry.jsonl'
    registry.write_text('{"fixture":"synthetic audited registry"}\n')
    plan_hash = hashlib.sha256(plan.read_bytes()).hexdigest()
    registry_hash = hashlib.sha256(registry.read_bytes()).hexdigest()
    split = {'train': 113207, 'validation': 2781}
    (variant / 'data_plan_summary.json').write_text(json.dumps(dict(total=115988,
        split_counts=split, plan_sha256=plan_hash, registry_sha256=registry_hash)))
    (variant / 'text_data_audit.json').write_text(json.dumps(dict(passed=True, rows=115988,
        split_counts=split, plan_sha256=plan_hash)))
    (root / 'CAPTIONS_VERIFIED').write_text(json.dumps(dict(rows=115988,
        plan_sha256={'subject_motion': plan_hash}, registry_sha256=registry_hash)))
    template = root / 'template.yaml'
    template.write_text(yaml.safe_dump(dict(models={'student': {'vision_only_conditioning': False}}, training={
        'distributed': {'num_gpus': 8, 'sp_size': 2}, 'data': {'train_batch_size': 1},
        'loop': {'gradient_accumulation_steps': 8, 'max_train_steps': 3000}, 'checkpoint': {}, 'tracker': {}},
        callbacks={'worldmodel_validation': {}, 'text_input_guard': {}})))
    return template


def test_approved_recipe_keeps_bs32_and_new_output_paths(tmp_path):
    template = prepare(tmp_path)
    configure(tmp_path, template)
    config = yaml.safe_load((tmp_path / 'subject_motion/training_config.yaml').read_text())
    assert config['training']['loop'] == {'gradient_accumulation_steps': 8, 'max_train_steps': 3000}
    assert config['training']['data']['data_path'] == str(tmp_path / 'subject_motion/preprocessed/train')
    assert config['models']['student']['vision_only_conditioning'] is False


def test_changed_plan_cannot_reuse_old_pass_marker(tmp_path):
    template = prepare(tmp_path)
    (tmp_path / 'subject_motion/text_data_plan.jsonl').write_text('changed\n')
    with pytest.raises(ValueError, match='Frozen plan'):
        configure(tmp_path, template)


def test_new_run_checkpoint_prevents_accidental_fresh_restart(tmp_path):
    template = prepare(tmp_path)
    (tmp_path / 'subject_motion/runs/train/checkpoints/checkpoint-100').mkdir(parents=True)
    with pytest.raises(ValueError, match='resume explicitly'):
        configure(tmp_path, template)


def test_effective_batch_change_is_rejected(tmp_path):
    template = prepare(tmp_path)
    config = yaml.safe_load(template.read_text())
    config['training']['loop']['gradient_accumulation_steps'] = 4
    template.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match='BS32'):
        configure(tmp_path, template)


def test_partial_count_without_explicit_selection_is_rejected(tmp_path):
    template = prepare(tmp_path)
    for name in ('data_plan_summary.json', 'text_data_audit.json'):
        path = tmp_path / 'subject_motion' / name
        value = json.loads(path.read_text())
        value['total' if name.startswith('data_plan') else 'rows'] = 115905
        path.write_text(json.dumps(value))
    path = tmp_path / 'CAPTIONS_VERIFIED'
    value = json.loads(path.read_text())
    value['rows'] = 115905
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='partial union'):
        configure(tmp_path, template)


def test_explicit_hash_bound_selection_allows_audited_subset_and_rejects_tampering(tmp_path):
    template = prepare(tmp_path)
    excluded = dict(id='excluded', source_name='source', video='/video.mp4', window_id='window',
                    source_frame_indices=list(range(124)), reason='explicit user-selected caption tail')
    included = [{'id': str(index)} for index in range(115987)]
    registry = tmp_path / 'registry.jsonl'
    registry.write_text(''.join(json.dumps(r) + '\n' for r in [*included, excluded]))
    registry_hash = hashlib.sha256(registry.read_bytes()).hexdigest()
    selection = tmp_path / 'caption_selection.json'
    selection.write_text(json.dumps(dict(authorization='user_20261024_discard_persistent_caption_failures',
                                         registry_sha256=registry_hash, original_rows=115988,
                                         included_rows=115987, excluded=[excluded])))
    selection_hash = hashlib.sha256(selection.read_bytes()).hexdigest()
    variant = tmp_path / 'subject_motion'
    plan = variant / 'text_data_plan.jsonl'
    plan.write_text(''.join(json.dumps(r) + '\n' for r in included))
    plan_hash = hashlib.sha256(plan.read_bytes()).hexdigest()
    split = {'train': 113206, 'validation': 2781}
    (variant / 'data_plan_summary.json').write_text(json.dumps(dict(total=115987,
        split_counts=split, plan_sha256=plan_hash, registry_sha256=registry_hash,
        selection_sha256=selection_hash)))
    (variant / 'text_data_audit.json').write_text(json.dumps(dict(passed=True, rows=115987,
        split_counts=split, plan_sha256=plan_hash)))
    (tmp_path / 'CAPTIONS_VERIFIED').write_text(json.dumps(dict(rows=115987,
        plan_sha256={'subject_motion': plan_hash}, registry_sha256=registry_hash,
        selection_sha256=selection_hash)))
    configure(tmp_path, template)
    config = yaml.safe_load((variant / 'training_config.yaml').read_text())
    assert '115987' in config['training']['tracker']['run_name']
    selection.write_text(selection.read_text() + '\n')
    with pytest.raises(ValueError, match='partial union'):
        configure(tmp_path, template)


def test_composed_run_keeps_checkpoint_space_reserve_and_separate_name(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import tools.configure_union_dual_training_20261023 as gate
    template = prepare(tmp_path)
    for path in (tmp_path / 'subject_motion/data_plan_summary.json', tmp_path / 'CAPTIONS_VERIFIED'):
        value = json.loads(path.read_text())
        value['prompt_composition'] = 'background_plus_subject_exact_sentence_dedup_v1'
        path.write_text(json.dumps(value))
    monkeypatch.setattr(gate.shutil, 'disk_usage', lambda _: SimpleNamespace(free=100 * 2**30))
    with pytest.raises(ValueError, match='checkpoint reserve'):
        configure(tmp_path, template)
    monkeypatch.setattr(gate.shutil, 'disk_usage', lambda _: SimpleNamespace(free=1024 * 2**30))
    configure(tmp_path, template)
    config = yaml.safe_load((tmp_path / 'subject_motion/training_config.yaml').read_text())
    assert 'scene_subject_motion' in config['training']['tracker']['run_name']
