"""Read-only CPU provenance / static-background audit of four user-selected GT clips."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import av
import cv2
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tools.visualize_training_sources_20261027 import FONT, digest, expand_camera, local_controls, save_json

GALLERY = Path('/datacc05/lizihao/worldmodel/fastvideowm_train_gallery_20261027')
CASES = [
    'h3_js_easy_fasth3_t2v_js_1000_124/01_ecbeb130f67d',
    'h3_js_easy_fasth3_t2v_js_1000_long/04_71cb15700601',
    'h3_js_easy_h3_i2v_js_easy_124/02_bc2238806ee1',
    'h3_js_easy_h3_i2v_js_easy_long/02_94474f27c4a0',
]
SOURCE = Path('/home/lizihao/projects/worldmodel/vggt-omega/h3_js_easy')
MAP = Path('/datacc05/lizihao/worldmodel/fastvideowm_approved_uncached_v1_20260925/source_mapping.jsonl')


def load_raw_camera(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        e = payload['extrinsics'].astype(np.float64)
    camera = np.broadcast_to(np.eye(4), (len(e), 4, 4)).copy()
    camera[:, :3, :3] = e[:, :3, :3].transpose(0, 2, 1)
    camera[:, :3, 3] = -np.einsum('tij,tj->ti', camera[:, :3, :3], e[:, :3, 3])
    basis = np.diag([1., -1., 1., 1.])
    return basis @ (np.linalg.inv(camera[0]) @ camera) @ basis


def project(homography: np.ndarray, points: np.ndarray) -> np.ndarray:
    h = np.column_stack([points, np.ones(len(points))]) @ homography.T
    return h[:, :2] / h[:, 2:]


def background_mask(case_index: int, height: int, width: int) -> np.ndarray:
    """Conservative hand-defined static regions; no semantic segmentation claim."""
    mask = np.zeros((height, width), np.uint8)
    if case_index in (0, 1):
        mask[:150, :] = 255  # trees / distant background, above moving vehicles
    elif case_index == 2:
        mask[70:390, :150] = 255  # static ruined building edges; avoid central explosion
        mask[70:390, 690:] = 255
    else:
        mask[:170, :] = 255  # distant hillside, above moving tank / dust
    mask[:8] = mask[-8:] = 0
    mask[:, :8] = mask[:, -8:] = 0
    return mask


def rotation_homography(c0: np.ndarray,
                        c1: np.ndarray,
                        k0: np.ndarray,
                        k1: np.ndarray,
                        inverse: bool = False,
                        flip_y: bool = True) -> np.ndarray:
    basis = np.diag([1., -1., 1.]) if flip_y else np.eye(3)
    r = c1[:3, :3].T @ c0[:3, :3]
    if inverse:
        r = r.T
    return k1 @ basis @ r @ basis @ np.linalg.inv(k0)


def background_matches(frames: list[np.ndarray], mask: np.ndarray, camera: np.ndarray, intrinsics: np.ndarray,
                       output: Path) -> list[dict]:
    values = []
    intervals = list(zip(range(0, 120, 8), range(8, 128, 8), strict=False))
    intervals[-1] = (112, 120)
    for a, b in intervals:
        gray0, gray1 = [cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY) for i in (a, b)]
        points = cv2.goodFeaturesToTrack(gray0, 700, .01, 8, mask=mask, blockSize=7)
        if points is None or len(points) < 20:
            continue
        nxt, valid, _ = cv2.calcOpticalFlowPyrLK(gray0, gray1, points, None, winSize=(25, 25), maxLevel=4)
        back, valid_back, _ = cv2.calcOpticalFlowPyrLK(gray1, gray0, nxt, None, winSize=(25, 25), maxLevel=4)
        p, q = points[:, 0], nxt[:, 0]
        keep = valid[:, 0].astype(bool) & valid_back[:, 0].astype(bool)
        keep &= np.linalg.norm(back[:, 0] - p, axis=-1) < 1.
        xy = np.rint(q).astype(int)
        keep &= (xy[:, 0] >= 0) & (xy[:, 0] < 832) & (xy[:, 1] >= 0) & (xy[:, 1] < 480)
        keep &= mask[np.clip(xy[:, 1], 0, 479), np.clip(xy[:, 0], 0, 831)] > 0
        p, q = p[keep], q[keep]
        if len(p) < 16:
            continue
        fitted, inlier_mask = cv2.findHomography(p, q, cv2.RANSAC, 1.5)
        if fitted is None:
            continue
        inliers = inlier_mask[:, 0].astype(bool)
        p, q = p[inliers], q[inliers]
        if len(p) < 12:
            continue
        predictions = {
            'rotation_only':
            rotation_homography(camera[a], camera[b], intrinsics[a], intrinsics[b]),
            'wrong_inverse_rotation':
            rotation_homography(camera[a], camera[b], intrinsics[a], intrinsics[b], inverse=True),
            'wrong_no_y_basis_conversion':
            rotation_homography(camera[a], camera[b], intrinsics[a], intrinsics[b], flip_y=False),
            'zero_motion_fixed_K':
            np.eye(3),
            'fitted_homography':
            fitted,
        }
        errors = {name: np.linalg.norm(project(h, p) - q, axis=-1) for name, h in predictions.items()}
        row = dict(frame0=a,
                   frame1=b,
                   tracked=int(keep.sum()),
                   inliers=len(p),
                   median_observed_flow_px=np.median(q - p, axis=0).tolist(),
                   median_residual_px={
                       name: float(np.median(error))
                       for name, error in errors.items()
                   })
        values.append(row)
        if a in (0, 40, 80):
            im = cv2.cvtColor(frames[a], cv2.COLOR_RGB2BGR)
            im[mask == 0] = (im[mask == 0].astype(float) * .35).astype(np.uint8)
            for point, target, expected in zip(p[::3],
                                               q[::3],
                                               project(predictions['rotation_only'], p)[::3],
                                               strict=False):
                start = tuple(np.rint(point).astype(int))
                cv2.arrowedLine(im, start, tuple(np.rint(target).astype(int)), (0, 255, 0), 1, tipLength=.3)
                cv2.arrowedLine(im, start, tuple(np.rint(expected).astype(int)), (0, 0, 255), 1, tipLength=.3)
            cv2.putText(im, f'{a}->{b}: green=observed BG; red=pose rotation prediction', (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 2)
            cv2.imwrite(str(output / f'background_tracks_{a:03d}.jpg'), im)
    return values


def inspect_case(case_index: int, case: str, mapping: dict, output: Path) -> dict:
    case_dir = GALLERY / case
    meta = json.loads((case_dir / 'metadata.json').read_text())
    source = meta['source_name'].removeprefix('h3_js_easy_')
    stem = Path(meta['video']).stem
    raw_path = SOURCE / source / 'camera_pose' / f'{stem}.npz'
    record_path = SOURCE / source / 'records' / f'{stem}.json'
    raw = load_raw_camera(raw_path)
    record = json.loads(record_path.read_text())
    camera = np.load(case_dir / 'camera_to_world_37.npy').astype(float)
    indices = np.array(meta['source_frame_indices'])
    anchors = np.array(meta['h3_raw_frame_anchors'])
    max_original_translation = float(np.linalg.norm(raw[:, :3, 3], axis=-1).max())
    relative_source = np.linalg.inv(raw[indices[0]]) @ raw[indices]
    raw[:, :3, 3] /= max_original_translation
    raw[:, :3, 3] *= np.array(mapping['translation_scale_xyz'])
    expected = np.linalg.inv(raw[indices[0]]) @ raw[indices[anchors]]
    dest = output / f'{case_index + 1:02d}_{stem}'
    dest.mkdir(parents=True, exist_ok=True)
    expanded, _ = expand_camera(camera)
    intrinsic37 = np.load(case_dir / 'camera_intrinsics_37.npy')
    with np.load(raw_path, allow_pickle=False) as payload:
        expected_k = payload['intrinsics'][indices[anchors]].astype(float).copy()
    # The source pose metadata states VGGT used 672x384, not the RGB 1344x768.
    # 672x384 -> 840x480 -> center crop 832x480 (4 pixels on either side).
    expected_k[:, 0, :] *= 1.25
    expected_k[:, 1, :] *= 1.25
    expected_k[:, 0, 2] -= 4
    expected_k[:, 0, :] /= 832
    expected_k[:, 1, :] /= 480
    intrinsics = np.empty((124, 3, 3))
    for i in range(3):
        for j in range(3):
            intrinsics[:, i, j] = np.interp(np.arange(124), anchors, intrinsic37[:, i, j])
    intrinsics = np.diag([832., 480., 1.]) @ intrinsics
    with av.open(str(case_dir / 'raw.mp4')) as container:
        frames = [frame.to_ndarray(format='rgb24') for frame in container.decode(video=0)]
    mask = background_mask(case_index, 480, 832)
    cv2.imwrite(str(dest / 'background_mask.png'), mask)
    observations = background_matches(frames, mask, expanded, intrinsics, dest)
    save_json(dest / 'background_tracks_metrics.json', observations)
    dt, angular, _ = local_controls(camera)
    t = camera[:, :3, 3]
    local_abs_translation = abs(dt).sum(axis=0)
    angle_sum = angular.sum(axis=0)
    angle_abs = abs(angular).sum(axis=0)
    median_bg = {}
    if observations:
        for key in observations[0]['median_residual_px']:
            median_bg[key] = float(np.median([v['median_residual_px'][key] for v in observations]))
    result = dict(
        id=meta['id'],
        video=meta['video'],
        gallery_directory=str(case_dir),
        original_pose=str(raw_path),
        original_record=str(record_path),
        original_pose_sha256=digest(raw_path.read_bytes()),
        actual_training_parquet=meta['conditioned_parquet'],
        source_frames=[int(indices[0]), int(indices[-1])],
        source_seconds=[float(indices[0] / 24), float(indices[-1] / 24)],
        full_video_frames=len(raw),
        original_intended_camera=record.get('controls', record.get('prompt')),
        original_derived_actions=mapping['actions'],
        camera_cache_reconstruction_max_abs_error=float(np.max(abs(expected - camera))),
        intrinsics_reconstruction_max_abs_error=float(np.max(abs(expected_k - intrinsic37))),
        raw_full_video_max_camera_translation=max_original_translation,
        old_translation_multiplier=1 / max_original_translation,
        translation_scale_xyz=mapping['translation_scale_xyz'],
        raw_window_max_camera_translation=float(np.linalg.norm(relative_source[:, :3, 3], axis=-1).max()),
        cached_max_camera_translation=float(np.linalg.norm(t, axis=-1).max()),
        cached_end_translation=t[-1].tolist(),
        cached_path_length=float(np.linalg.norm(dt, axis=-1).sum()),
        cumulative_local_abs_translation_xyz=local_abs_translation.tolist(),
        cumulative_local_signed_pitch_yaw_roll_deg=angle_sum.tolist(),
        cumulative_local_abs_pitch_yaw_roll_deg=angle_abs.tolist(),
        background_pair_count=len(observations),
        median_of_pair_median_background_residual_px=median_bg,
        background_roi='manual fixed ROI; not automated segmentation',
        source_quality_approved=mapping['quality_approved'],
        automatic_gpu_qc_pass=mapping['gpu_qc_pass'],
        warning=
        '2D background matching cannot identify a unique metric camera translation; fitted homography can include translation/plane effects',
        report_directory=str(dest))
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    times = anchors / 24
    for axis, name in enumerate('XYZ'):
        axes[0].plot(times, t[:, axis], label=name)
        axes[1].plot(times[1:], dt[1:, axis], label=name)
    axes[0].set_title('Persisted training camera center (arbitrary training units)')
    axes[1].set_title('Local translation increments per H3 anchor (unequal time spans)')
    for axis, name in enumerate(['pitch up', 'yaw right', 'roll']):
        axes[2].plot(times, np.cumsum(angular, axis=0)[:, axis], label=name)
    axes[2].set_title('Accumulated local angular increments (degrees)')
    for ax in axes:
        ax.legend()
        ax.grid()
        ax.set_xlabel('Training-window seconds')
    fig.tight_layout()
    fig.savefig(dest / 'camera_metrics.png')
    plt.close(fig)
    sheet = Image.new('RGB', (832 * 2, (480 + 32) * 3), 'black')
    font = ImageFont.truetype(FONT, 20)
    for slot, index in enumerate([0, 24, 49, 74, 99, 123]):
        x, y = slot % 2 * 832, slot // 2 * 512
        sheet.paste(Image.fromarray(frames[index]), (x, y + 32))
        ImageDraw.Draw(sheet).text((x + 10, y + 3), f'target={index} source={indices[index]}', fill='white', font=font)
    sheet.save(dest / 'raw_six_frames.jpg', quality=92)
    save_json(dest / 'result.json', result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    cv2.setRNGSeed(20261027)
    wanted = {json.loads((GALLERY / path / 'metadata.json').read_text())['id'] for path in CASES}
    mapping = {}
    for line in MAP.open():
        row = json.loads(line)
        key = row['source_name'] + '/' + row['sample_id']
        if key in wanted:
            mapping[key] = row
    results = []
    for index, case in enumerate(CASES):
        meta = json.loads((GALLERY / case / 'metadata.json').read_text())
        result = inspect_case(index, case, mapping[meta['id']], args.output)
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    save_json(args.output / 'results.json', results)


if __name__ == '__main__':
    main()
