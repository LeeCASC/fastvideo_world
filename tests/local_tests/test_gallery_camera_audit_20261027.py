"""CPU audit conventions: actual C2W, raster coordinates, projection direction."""
import numpy as np
from scipy.spatial.transform import Rotation

from tools.audit_gallery_camera_four_20261027 import load_raw_camera, project, rotation_homography


def test_pitch_up_moves_static_background_down_in_raster_coordinates():
    start, end = np.eye(4), np.eye(4)
    end[:3, :3] = Rotation.from_rotvec(np.array([-.1, 0, 0])).as_matrix()
    k = np.array([[1000., 0, 416], [0, 1000, 240], [0, 0, 1]])
    h = rotation_homography(start, end, k, k)
    q = project(h, np.array([[416., 240.]]))[0]
    assert q[1] > 240
    np.testing.assert_allclose(q[0], 416, atol=1e-8)


def test_yaw_right_moves_static_background_left():
    start, end = np.eye(4), np.eye(4)
    end[:3, :3] = Rotation.from_rotvec(np.array([0, .1, 0])).as_matrix()
    k = np.array([[1000., 0, 416], [0, 1000, 240], [0, 0, 1]])
    q = project(rotation_homography(start, end, k, k), np.array([[416., 240.]]))[0]
    assert q[0] < 416


def test_vggt_extrinsics_are_inverted_not_misread_as_camera_centers(tmp_path):
    c2w = np.broadcast_to(np.eye(4), (2, 4, 4)).copy()
    c2w[0, :3, 3] = [2, 3, 4]
    c2w[1, :3, 3] = [3, 5, 8]
    c2w[1, :3, :3] = Rotation.from_rotvec(np.array([.1, .2, .3])).as_matrix()
    path = tmp_path / 'camera.npz'
    np.savez(path, extrinsics=np.linalg.inv(c2w)[:, :3])
    actual = load_raw_camera(path)
    basis = np.diag([1., -1., 1., 1.])
    expected = basis @ (np.linalg.inv(c2w[0]) @ c2w) @ basis
    np.testing.assert_allclose(actual, expected, atol=1e-10)
