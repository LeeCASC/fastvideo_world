from argparse import Namespace
import json

import pytest

from tools.prepare_scene_subject_text_20261024 import COMPOSITION, compose_caption, plan
from tools.prepare_union_dual_captions_20261023 import (
    POLICY, finalize, read_jsonl, verify_captions, write_jsonl,
)


def source_fixture(root):
    specs, captions = [], []
    for index in range(2):
        spec = dict(id=f'video-{index}', window_id=f'window-{index}', video=f'/video-{index}.mp4',
                    source_name='source', source_parquet='/source/data.parquet', source_row=index,
                    source_frame_indices=list(range(124)), split='train' if index == 0 else 'validation',
                    existing_caption='', existing_encoded=None)
        scene = 'A stone courtyard contains arched doorways and tiled floors.'
        caption = dict(spec, subject_motion_description='A red truck crosses the dry field.' if index == 0 else scene,
                       background_description=scene, subject_motion_status='observed' if index == 0 else 'not_observed',
                       confidence=.9, quality_flags=[], reused_subject_caption=False, error=None, policy=POLICY,
                       sampled_source_frame_indices=[0, 25, 49, 74, 98, 123], sampled_rgb_sha256=['sha'] * 6)
        specs.append(spec)
        captions.append(caption)
    write_jsonl(root / 'registry.jsonl', specs)
    write_jsonl(root / 'captions/worker_00.jsonl', captions)
    (root / 'inventory_summary.json').write_text(json.dumps({'usable': 2, 'errors': 0}))
    finalize(Namespace(root=root))


def test_background_is_explicit_prefix_and_only_exact_duplicate_sentences_are_removed():
    background = 'A forest borders a dirt road. The sky is blue.'
    subject = 'A forest borders a dirt road. A tan tank reverses along the road.'
    result = compose_caption(background, subject)
    assert result == background + ' A tan tank reverses along the road.'
    assert 'reverses' in result
    assert result.count('A forest borders a dirt road.') == 1
    assert compose_caption(background, background) == background


@pytest.mark.parametrize('background,subject', [
    ('The camera pans across a forest.', 'A tan tank is visible nearby.'),
    ('A tank stands in a forest.', 'A tan tank is visible nearby.'),
    ('A forest surrounds the dirt road.', 'The camera follows a red truck.'),
])
def test_forbidden_components_cannot_be_hidden_by_composition(background, subject):
    with pytest.raises(ValueError):
        compose_caption(background, subject)


def test_frozen_composition_preserves_sources_and_keeps_background_unencoded(tmp_path):
    source, output = tmp_path / 'source', tmp_path / 'output'
    source_fixture(source)
    original = (source / 'subject_motion/text_data_plan.jsonl').read_bytes()
    proof = plan(output, source)
    assert proof['composition'] == COMPOSITION
    assert proof['background_encoding_deferred']
    joined = list(read_jsonl(output / 'subject_motion/text_data_plan.jsonl'))
    backgrounds = list(read_jsonl(output / 'background_only/text_data_plan.jsonl'))
    assert len(joined) == len(backgrounds) == 2
    for subject, background in zip(joined, backgrounds, strict=True):
        assert subject['id'] == background['id']
        assert subject['source_frame_indices'] == background['source_frame_indices']
        assert subject['split'] == background['split']
        assert subject['caption'].startswith(background['caption'])
        assert subject['prompt_composition'] == COMPOSITION
    assert not (output / 'background_only/preprocessed').exists()
    assert (source / 'subject_motion/text_data_plan.jsonl').read_bytes() == original
    verify_captions(Namespace(root=output))
    assert plan(output, source) == proof


def test_mismatched_background_window_or_frozen_output_tampering_is_rejected(tmp_path):
    source, output = tmp_path / 'source', tmp_path / 'output'
    source_fixture(source)
    plan(output, source)
    path = output / 'subject_motion/text_data_plan.jsonl'
    path.write_text(path.read_text() + '\n')
    with pytest.raises(ValueError, match='plan/selection changed'):
        plan(output, source)


def test_original_caption_directory_cannot_be_overwritten(tmp_path):
    with pytest.raises(ValueError, match='original'):
        plan(tmp_path, tmp_path)
