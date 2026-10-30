from copy import deepcopy
from pathlib import Path

import pytest

from tools.js00001_simple_text_20261023 import SCENE, pair_cases


def original_pair():
    base = dict(sample_id='js_00001', seed=123, camera_pose='same.json', action='W', alpha=1.,
                fps=32.524590, frames=124, width=832, height=480, prompt='original long text')
    return [dict(base, id=f'js_00001_{direction}', subject_direction=direction)
            for direction in ('forward', 'backward')]


def test_only_subject_motion_phrase_changes_and_originals_are_untouched():
    original = original_pair()
    saved = deepcopy(original)
    cases = pair_cases(original, Path('/output'))
    assert original == saved
    assert len(cases) == 2
    assert {row['prompt'] for row in cases} == {
        f'{SCENE} The tank moves forward.', f'{SCENE} The tank moves backward.'}
    assert all(row['action'] == 'W' and row['seed'] == 123 for row in cases)
    assert len({row['condition_parquet'] for row in cases}) == 2
    assert all('camera' not in row['prompt'].lower() for row in cases)


def test_mismatched_nontext_pair_is_rejected():
    original = original_pair()
    original[1]['seed'] = 999
    with pytest.raises(ValueError, match='seed'):
        pair_cases(original, Path('/output'))


def test_missing_direction_is_rejected():
    with pytest.raises(ValueError, match='exactly one'):
        pair_cases(original_pair()[:1], Path('/output'))
