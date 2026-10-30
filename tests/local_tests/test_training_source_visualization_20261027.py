"""CPU tests: exact H3 anchor interpolation, camera labels, training RGB crop."""
import hashlib
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tools.visualize_training_sources_20261027 import choose_samples, expand_camera, load_drawers, local_controls, resize_crop


def test_h3_nonuniform_anchors_preserve_actual_input():
    camera = np.broadcast_to(np.eye(4), (37, 4, 4)).copy()
    camera[:, 0, 3] = np.linspace(0, 2, 37)
    expanded, anchors = expand_camera(camera)
    assert anchors[:7].tolist() == [0, 4, 8, 12, 16, 17, 21]
    assert anchors[-1] == 123
    np.testing.assert_allclose(expanded[anchors], camera)


def test_unrebased_camera_rejected():
    camera = np.broadcast_to(np.eye(4), (37, 4, 4)).copy()
    camera[:, 0, 3] = 2
    with pytest.raises(ValueError, match='identity'):
        expand_camera(camera)


@pytest.mark.parametrize('axis,sign,key', [('x', -1, 'I'), ('x', 1, 'K'), ('y', -1, 'J'), ('y', 1, 'L')])
def test_rotation_signs_match_approved_keyboard(axis, sign, key):
    camera = np.broadcast_to(np.eye(4), (37, 4, 4)).copy()
    camera[:, :3, :3] = Rotation.from_euler(axis, np.linspace(0, sign * 20, 37)[:, None], degrees=True).as_matrix()
    expanded, _ = expand_camera(camera)
    dt, angles, _ = local_controls(expanded)
    np.testing.assert_allclose(dt, 0, atol=1e-9)
    if key in 'IK':
        assert np.sign(angles[1:, 0]).tolist() == [1 if key == 'I' else -1] * 123
    else:
        assert np.sign(angles[1:, 1]).tolist() == [1 if key == 'L' else -1] * 123


def test_crop_identity_and_reference_definitions_load():
    frame = np.zeros((480, 832, 3), np.uint8)
    np.testing.assert_array_equal(resize_crop(frame), frame)
    drawers = load_drawers()
    assert 'TrajectoryPanelRenderer' in drawers
    assert '_draw_actions_on_frame' in drawers


def test_selection_is_train_only_and_does_not_pad_small_sources(tmp_path):
    root = tmp_path / 'subject_motion'
    root.mkdir()
    rows = [dict(id=f'a/{i}', source_name='a', video=f'a{i}.mp4', split='train') for i in range(6)]
    rows += [dict(id='b/0', source_name='b', video='b.mp4', split='train'),
             dict(id='a/val', source_name='a', video='heldout.mp4', split='validation')]
    data = ''.join(json.dumps(row) + '\n' for row in rows)
    (root / 'text_data_plan.jsonl').write_text(data)
    (root / 'TEXT_DATA_VERIFIED').touch()
    (root / 'text_data_audit.json').write_text(json.dumps(dict(passed=True, plan_sha256=hashlib.sha256(data.encode()).hexdigest())))
    (tmp_path / 'caption_selection.json').write_text(json.dumps(dict(excluded=[dict(id='rejected/0')])))
    chosen, counts, _ = choose_samples(root, 5, 20261027)
    assert len(chosen) == 6
    assert counts['a']['selected'] == 5
    assert counts['b']['selected'] == 1
    assert all(row['split'] == 'train' for row in chosen)
    assert {row['id'] for row in chosen} == {row['id'] for row in choose_samples(root, 5, 20261027)[0]}
    (tmp_path / 'caption_selection.json').write_text(json.dumps(dict(excluded=[dict(id='b/0')])))
    with pytest.raises(ValueError, match='Excluded ID'):
        choose_samples(root, 5, 20261027)
