import importlib.util
import json
from pathlib import Path

import pytest


PATH = Path(__file__).resolve().parents[2] / 'tools/prepare_union_dual_captions_20261023.py'
SPEC = importlib.util.spec_from_file_location('dual_captions', PATH)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def example():
    return dict(subject_motion_description='A tan tank reverses along a dirt road.',
                background_description='A dirt road crosses a forest under a blue sky.',
                subject_motion_status='observed', confidence=.8, quality_flags=[])


def test_two_descriptions_keep_actor_only_in_subject_variant():
    assert module.validate_pair(example())['subject_motion_description'].startswith('A tan tank')


@pytest.mark.parametrize('background', [
    'A tank is parked on a forest road.',
    'A camera moves along a forest road.',
    'A vehicle moves forward across the open desert.',
    'People are walking along the forest road.',
    'A view of a road with audio narration.',
    "The player's view is fixed on the forest road.",
])
def test_background_rejects_actors_motion_and_camera_audio(background):
    pair = example()
    pair['background_description'] = background
    with pytest.raises(ValueError):
        module.validate_pair(pair)


def test_unknown_motion_is_not_forced_to_forward():
    pair = example()
    pair.update(subject_motion_description='A tan tank is visible on a forest road.', subject_motion_status='uncertain')
    assert module.validate_pair(pair)['subject_motion_status'] == 'uncertain'


def test_static_road_runs_is_not_actor_motion():
    pair = example()
    pair['background_description'] = 'A dirt road runs along the top of the dunes.'
    assert module.validate_pair(pair)['background_description']


def test_reuse_requires_whole_window_not_only_video(tmp_path):
    video = tmp_path / 'video.mp4'
    assert module.window_key(video, list(range(124))) != module.window_key(video, list(range(1, 125)))


def test_invalid_status_and_confidence_fail():
    for update in ({'subject_motion_status': 'forward'}, {'confidence': 2.0}, {'background_description': ''}):
        with pytest.raises(ValueError):
            module.validate_pair(dict(example(), **update))
    pair = example()
    pair.pop('confidence')
    with pytest.raises(ValueError, match='missing confidence'):
        module.validate_pair(pair)


def test_sentence_cleanup_keeps_background_evidence_without_inventing_motion():
    pair = example()
    pair['background_description'] += ' A tank is parked beside the road.'
    cleaned = module.clean_pair_sentences(pair)
    assert cleaned['background_description'] == example()['background_description']
    assert cleaned['subject_motion_description'] == pair['subject_motion_description']
    assert 'removed_forbidden_sentences:background_description' in cleaned['quality_flags']


def test_removed_camera_claim_downgrades_motion_certainty():
    pair = example()
    pair['subject_motion_description'] += ' The camera follows the tank.'
    cleaned = module.clean_pair_sentences(pair)
    assert cleaned['subject_motion_description'] == example()['subject_motion_description']
    assert cleaned['subject_motion_status'] == 'uncertain'


def test_cleanup_never_uses_generic_fallback_for_empty_background():
    pair = example()
    pair['background_description'] = 'A tank is parked on the road.'
    with pytest.raises(ValueError):
        module.clean_pair_sentences(pair)


def test_finalize_repairs_append_only_response_and_binds_exact_window(tmp_path):
    from argparse import Namespace
    indices = list(range(124))
    spec = dict(id='one', window_id='one-window', video='/source/video.mp4',
                source_frame_indices=indices, split='train', source_name='one-source')
    module.write_jsonl(tmp_path / 'registry.jsonl', [spec])
    pair = example()
    pair['background_description'] += ' A tank is parked beside the road.'
    result = dict(spec, policy=module.POLICY, sampled_source_frame_indices=[0, 25, 49, 74, 98, 123], sampled_rgb_sha256=['frame-sha']*6,
                  raw_response=json.dumps(pair), error='foreground in background')
    path = tmp_path / 'captions/worker_00.jsonl'
    module.write_jsonl(path, [result])
    original = path.read_bytes()
    module.finalize(Namespace(root=tmp_path))
    assert path.read_bytes() == original
    report = json.loads((tmp_path / 'CAPTIONS_VERIFIED').read_text())
    assert report['rows'] == 1 and report['repaired_caption_pairs'] == 1
    planned = next(module.read_jsonl(tmp_path / 'subject_motion/text_data_plan.jsonl'))
    assert planned['caption'] == pair['subject_motion_description']
    assert planned['source_frame_indices'] == indices
    assert planned['subject_motion_status'] == 'observed'
    assert planned['registry_sha256'] == module.file_digest(tmp_path / 'registry.jsonl')


@pytest.mark.parametrize('text', [
    'An industrial courtyard contains cylindrical storage tanks and concrete walls.',
    'A corridor has wooden framing, stone arches and pale tiled floors.',
    'A wooden boardwalk with tables and benches runs through a tropical forest.',
    'A train station platform has tiled flooring and yellow tactile paving.',
    'Steel train tracks cross a gravel yard beneath a cloudy sky.',
    'A hot dog cart stands beside snowy trees under glowing lanterns.',
    "A men's clothing store has a dark facade and illuminated green signage.",
])
def test_literal_infrastructure_is_not_camera_or_foreground(text):
    assert module.validate_pair(dict(example(), background_description=text))


@pytest.mark.parametrize('text', [
    'A battle tank is beside a storage tank on concrete pavement.',
    'A tank stands in an industrial complex under a blue sky.',
    'A train engine sits beside the concrete industrial building.',
    'The camera pans left along the industrial yard.',
])
def test_disambiguation_does_not_allow_combat_tanks_or_controls(text):
    with pytest.raises(ValueError):
        module.validate_pair(dict(example(), background_description=text))


def test_no_subject_caption_must_still_describe_real_scene():
    scene = 'An ornate stone corridor contains arched doorways and golden wall carvings.'
    pair = dict(example(), subject_presence='none', subject_motion_status='not_observed',
                subject_motion_description=scene, background_description=scene)
    assert module.validate_pair(pair)['subject_motion_description'] == scene
    for text in ('', 'No subjects are visible to observe motion.', 'No physical movement is evident.'):
        with pytest.raises(ValueError):
            module.validate_pair(dict(pair, subject_motion_description=text))
    with pytest.raises(ValueError, match='not_observed'):
        module.validate_pair(dict(pair, subject_motion_status='observed'))


def _fixture_spec():
    return dict(id='one', window_id='one-window', video='/source/video.mp4', source_frame_indices=list(range(124)),
                split='train', source_name='source', existing_caption='')


def test_only_explicit_no_subject_allows_paired_visible_environment_reuse():
    pair = dict(example(), subject_presence='none', subject_motion_status='not_observed',
                subject_motion_description='No subjects are visible. The camera pans right.')
    clean = module.clean_pair_sentences(pair)
    assert clean['subject_motion_description'] == pair['background_description']
    assert clean['caption_mode'] == 'environment_only_explicit_no_subject'
    assert clean['confidence'] == pair['confidence']
    pair.pop('subject_presence')
    with pytest.raises(ValueError):
        module.clean_pair_sentences(pair)
    pair.update(subject_presence='none', subject_motion_description='A tan tank reverses along the dirt road.')
    with pytest.raises(ValueError, match='contradicts'):
        module.clean_pair_sentences(pair)


@pytest.mark.parametrize('absence', [
    'No foreground subjects are visible.',
    'No foreground subject is visible to track motion.',
    'There are no visible foreground subjects in these images.',
    'No main actors are visible.',
])
def test_foreground_absence_sentence_does_not_block_paired_environment(absence):
    pair = dict(example(), subject_presence='none', subject_motion_status='not_observed',
                subject_motion_description=absence + ' The camera moves forward along the road.')
    cleaned = module.clean_pair_sentences(pair)
    assert cleaned['subject_motion_description'] == pair['background_description']
    assert cleaned['caption_mode'] == 'environment_only_explicit_no_subject'
    assert cleaned['confidence'] == pair['confidence']
    assert 'removed_forbidden_sentences:subject_motion_description' in cleaned['quality_flags']


def test_foreground_absence_cleanup_cannot_hide_real_visible_actor_contradiction():
    pair = dict(example(), subject_presence='none', subject_motion_status='not_observed',
                subject_motion_description='No foreground subjects are visible. Several parked vehicles line the road.')
    with pytest.raises(ValueError, match='contradicts'):
        module.clean_pair_sentences(pair)


def test_no_actor_static_motion_wording_is_removed_not_learned_as_actor_motion():
    pair = dict(example(), subject_presence='none', subject_motion_status='not_observed',
                subject_motion_description='No foreground subjects are visible. The distant hills remain stationary.')
    cleaned = module.clean_pair_sentences(pair)
    assert cleaned['subject_motion_description'] == pair['background_description']
    assert cleaned['subject_motion_status'] == 'not_observed'
    assert cleaned['confidence'] == pair['confidence']


def _fixture_result(spec, **overrides):
    return dict(spec, **example(), policy=module.POLICY,
                sampled_source_frame_indices=[0, 25, 49, 74, 98, 123], sampled_rgb_sha256=['frame-sha'] * 6,
                reused_subject_caption=False, error=None, **overrides)


def test_plan_reuses_global_repairs_after_resharding_and_does_not_load_gpu(tmp_path):
    from argparse import Namespace
    spec = _fixture_spec()
    result = _fixture_result(spec)
    result['background_description'] += ' A tank is parked nearby.'
    result['error'] = 'foreground in background'
    result['raw_response'] = json.dumps({k: result[k] for k in example()})
    module.write_jsonl(tmp_path / 'registry.jsonl', [spec])
    module.write_jsonl(tmp_path / 'captions/worker_07.jsonl', [result])
    module.plan(Namespace(root=tmp_path, workers=1))
    assert list(module.read_jsonl(tmp_path / 'caption_pending.jsonl')) == []
    module.caption(Namespace(root=tmp_path, workers=1, worker=0, batch_size=4, limit=0, attempt=1))
    assert len(list((tmp_path / 'captions').glob('worker_*.jsonl'))) == 1
    # A valid repaired pair survives even if only the separate repair evidence remains.
    (tmp_path / 'captions/worker_07.jsonl').rename(tmp_path / 'raw_preserved.jsonl')
    valid, _, _ = module.collect_captions(tmp_path, [spec])
    assert valid['one']['background_description'] == example()['background_description']


def test_wrong_window_or_policy_is_never_reused(tmp_path):
    spec = _fixture_spec()
    result = _fixture_result(spec)
    result['video'] = '/other/video.mp4'
    module.write_jsonl(tmp_path / 'captions/worker_00.jsonl', [result])
    assert module.collect_captions(tmp_path, [spec])[0] == {}
    result['video'] = spec['video']
    result['policy'] = 'different-policy'
    module.write_jsonl(tmp_path / 'captions/worker_00.jsonl', [result])
    assert module.collect_captions(tmp_path, [spec])[0] == {}


def test_valid_pair_is_not_lost_to_later_failure(tmp_path):
    spec = _fixture_spec()
    result = _fixture_result(spec)
    failed = dict(result, error='generation failed', raw_response='truncated')
    module.write_jsonl(tmp_path / 'captions/worker_00.jsonl', [result, failed])
    assert module.collect_captions(tmp_path, [spec])[0]['one']['error'] is None


def test_only_unterminated_invalid_final_jsonl_record_is_tolerated(tmp_path):
    path = tmp_path / 'log.jsonl'
    path.write_text('{"id":"valid"}\n{"id":')
    assert list(module.read_jsonl(path, allow_truncated_tail=True)) == [{'id': 'valid'}]
    with pytest.raises(json.JSONDecodeError):
        list(module.read_jsonl(path))
    path.write_text('{bad}\n{"id":"valid"}\n')
    with pytest.raises(json.JSONDecodeError):
        list(module.read_jsonl(path, allow_truncated_tail=True))


def test_failed_finalize_invalidates_stale_success_marker(tmp_path):
    from argparse import Namespace
    module.write_jsonl(tmp_path / 'registry.jsonl', [_fixture_spec()])
    (tmp_path / 'CAPTIONS_VERIFIED').write_text('{"old":true}')
    with pytest.raises(ValueError, match='missing/invalid'):
        module.finalize(Namespace(root=tmp_path))
    assert not (tmp_path / 'CAPTIONS_VERIFIED').exists()
    assert list(tmp_path.glob('CAPTIONS_VERIFIED.invalidated.*'))


def test_repair_prompt_changes_strategy_and_drops_rejected_draft():
    spec = dict(_fixture_spec(), existing_caption='The camera moves forward through the corridor.')
    feedback = dict(error='empty caption', previous_pair={'subject_motion_description': 'No subjects are visible.'})
    prompt = module.caption_prompt(spec, feedback, 3)
    assert 'subject_presence="none"' in prompt
    assert spec['existing_caption'] not in prompt
    assert 'Final rewrite:' in prompt and 'empty caption' in prompt


def test_explicit_user_exclusions_preserve_full_registry_and_bind_filtered_plan(tmp_path):
    from argparse import Namespace
    spec = _fixture_spec()
    excluded = dict(spec, id='two', window_id='two-window')
    module.write_jsonl(tmp_path / 'registry.jsonl', [spec, excluded])
    original = (tmp_path / 'registry.jsonl').read_bytes()
    module.write_jsonl(tmp_path / 'captions/worker_00.jsonl', [_fixture_result(spec)])
    module.write_jsonl(tmp_path / 'explicit_ids.jsonl', [{'id': 'two'}])
    module.exclude_failed(Namespace(root=tmp_path, authorize_exclude_failed=True,
                                    exclude_ids_path=tmp_path / 'explicit_ids.jsonl'))
    assert (tmp_path / 'registry.jsonl').read_bytes() == original
    assert module.registry_specs(tmp_path) == [spec]
    module.finalize(Namespace(root=tmp_path))
    marker = json.loads((tmp_path / 'CAPTIONS_VERIFIED').read_text())
    assert marker['rows'] == 1
    assert marker['selection_sha256'] == module.file_digest(tmp_path / 'caption_selection.json')
    summary = json.loads((tmp_path / 'subject_motion/data_plan_summary.json').read_text())
    assert summary['selection_sha256'] == marker['selection_sha256']
    module.verify_captions(Namespace(root=tmp_path))
    background_plan = tmp_path / 'background_only/text_data_plan.jsonl'
    original_plan = background_plan.read_text()
    background_plan.write_text(original_plan + '\n')
    with pytest.raises(ValueError, match='plan/selection changed'):
        module.verify_captions(Namespace(root=tmp_path))
    background_plan.write_text(original_plan)
    (tmp_path / 'registry.jsonl').write_text(original.decode() + '\n')
    with pytest.raises(ValueError, match='stale'):
        module.registry_specs(tmp_path)


def test_exclusion_requires_authorization_and_refuses_encoding_in_progress(tmp_path):
    from argparse import Namespace
    with pytest.raises(ValueError, match='authorization'):
        module.exclude_failed(Namespace(root=tmp_path, authorize_exclude_failed=False))
    (tmp_path / 'subject_motion/preprocessed').mkdir(parents=True)
    (tmp_path / 'subject_motion/preprocessed/existing.parquet').touch()
    with pytest.raises(ValueError, match='after encoding'):
        module.exclude_failed(Namespace(root=tmp_path, authorize_exclude_failed=True))
