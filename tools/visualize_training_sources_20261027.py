"""CPU-only exact-window gallery of the ACTIVE, approved FastVideoWM train split.

Reuse the requested Cosmos vwm_datasets drawing definitions, not its random
dataset sampler (which could select rejected/validation windows). Never load a
model or modify source media, training conditions, or a training configuration.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from fractions import Fraction
import hashlib
import html
from io import BytesIO
import json
import math
import multiprocessing
import os
from pathlib import Path
import time
from typing import Any

import av
import cv2
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path('/datacc05/lizihao/worldmodel/fastvideowm_union_scene_subject_text_20261024/subject_motion')
RENDERER = Path('/home/lizihao/projects/cosmos-predict2-worldmodel/third_party/vwm_datasets/'
                'vwm_datasets/tools/visualize/render.py')
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
VERSION = 'approved_train_cosmos_gallery_v1'


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp.json')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(temp, path)


def load_drawers() -> dict[str, Any]:
    """Actual pure reference definitions, without importing dataset/CUDA stacks."""
    names = {
        '_wrap_delta_deg_to_180',
        '_traj_normalize',
        '_traj_draw_view_frustum',
        '_traj_draw_camera_local',
        '_draw_actions_on_frame',
        'TrajectoryPanelRenderer',
        '_wrap_deg_360',
        '_se3_to_pyr_hy_yup',
    }
    tree = ast.parse(RENDERER.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef | ast.ClassDef) and node.name in names]
    if {node.name for node in nodes} != names:
        raise ValueError('Reference renderer API changed')
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    namespace: dict[str, Any] = dict(np=np,
                                     cv2=cv2,
                                     plt=plt,
                                     math=math,
                                     Poly3DCollection=Poly3DCollection,
                                     FormatStrFormatter=FormatStrFormatter,
                                     wrap_deg_180=lambda x: (x + 180.) % 360. - 180.)
    exec(compile(module, str(RENDERER), 'exec'), namespace)
    return namespace


def choose_samples(root: Path, per_source: int, seed: int) -> tuple[list[dict], dict, str]:
    """Fixed hash sample, train only; do not substitute excluded or held-out rows."""
    plan = root / 'text_data_plan.jsonl'
    plan_hash = digest(plan.read_bytes())
    audit = json.loads((root / 'text_data_audit.json').read_text())
    if not (root / 'TEXT_DATA_VERIFIED').exists() or not audit['passed'] or audit['plan_sha256'] != plan_hash:
        raise ValueError('Active full-data audit gate is absent or stale')
    selection = json.loads((root.parent / 'caption_selection.json').read_text())
    excluded = {row['id'] for row in selection['excluded']}
    pools: dict[str, list[dict]] = defaultdict(list)
    with plan.open() as handle:
        for index, line in enumerate(handle):
            spec = json.loads(line)
            if spec['split'] != 'train':
                continue
            if spec['id'] in excluded:
                raise ValueError('Excluded ID in active train plan')
            conditioned = root / 'preprocessed/train' / f'worker_{index % 8:02d}' / f'data_{index // 8:06d}.parquet'
            pools[spec['source_name']].append(dict(spec, conditioned_parquet=str(conditioned), plan_index=index))
    chosen, counts = [], {}
    for source, rows in sorted(pools.items()):
        # Prefer separate source videos rather than five windows of one video.
        rows.sort(key=lambda r: digest(f'{seed}:{r["id"]}'.encode()))
        distinct: dict[str, dict] = {}
        for row in rows:
            distinct.setdefault(row['video'], row)
        take = list(distinct.values())[:per_source]
        if len(take) < min(per_source, len(rows)):
            ids = {r['id'] for r in take}
            take.extend(r for r in rows if r['id'] not in ids)
            take = take[:per_source]
        for ordinal, row in enumerate(take, 1):
            row['case_name'] = f'{ordinal:02d}_{digest(row["id"].encode())[:12]}'
        chosen.extend(take)
        counts[source] = dict(train_rows=len(rows), selected=len(take), unique_videos=len(distinct))
    return chosen, counts, plan_hash


def expand_camera(camera: np.ndarray, frames: int = 124) -> tuple[np.ndarray, np.ndarray]:
    """H3 [1,4,4,4,4] spans; only presentation is interpolated, never training."""
    camera = np.asarray(camera, dtype=np.float64)
    if camera.shape != (37, 4, 4) or not np.isfinite(camera).all():
        raise ValueError('Expected 37 finite persisted training camera matrices')
    if not np.allclose(camera[0], np.eye(4), atol=1e-4):
        raise ValueError('Training camera first frame is not identity')
    if not np.allclose(camera[:, :3, :3] @ camera[:, :3, :3].transpose(0, 2, 1), np.eye(3), atol=.01):
        raise ValueError('Camera rotations are not rigid')
    anchors = np.cumsum(np.resize([1, 4, 4, 4, 4], len(camera))) - 1
    if anchors[-1] != frames - 1:
        raise ValueError('H3 temporal alignment mismatch')
    out = np.broadcast_to(np.eye(4), (frames, 4, 4)).copy()
    out[:, :3, :3] = Slerp(anchors, Rotation.from_matrix(camera[:, :3, :3]))(np.arange(frames)).as_matrix()
    for axis in range(3):
        out[:, axis, 3] = np.interp(np.arange(frames), anchors, camera[:, axis, 3])
    if not np.allclose(out[anchors], camera, atol=1e-5):
        raise ValueError('Interpolation changed actual camera anchors')
    return out, anchors


def local_controls(camera: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    dt = np.zeros((len(camera), 3))
    dt[1:] = np.einsum('tji,tj->ti', camera[:-1, :3, :3], np.diff(camera[:, :3, 3], axis=0))
    angular = np.zeros_like(dt)
    angular[1:] = Rotation.from_matrix(camera[:-1, :3, :3].transpose(0, 2, 1) @ camera[1:, :3, :3]).as_rotvec()
    angular *= 180. / np.pi * np.array([-1., 1., -1.])
    # Presentation-only adaptive threshold; full numeric values are always kept.
    threshold = max(float(np.median(np.linalg.norm(dt[1:], axis=-1))) * .2, 1e-6)
    return dt, angular, threshold


def resize_crop(frame: np.ndarray, height: int = 480, width: int = 832) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = max(height / h, width / w)
    rh, rw = max(height, round(h * scale)), max(width, round(w * scale))
    frame = cv2.resize(frame, (rw, rh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    top, left = (rh - height) // 2, (rw - width) // 2
    return frame[top:top + height, left:left + width].copy()


def load_window(video: str, indices: list[int]) -> list[np.ndarray]:
    if len(indices) != 124 or min(indices) < 0 or any(b < a for a, b in zip(indices, indices[1:], strict=False)):
        raise ValueError('Expected exact monotonic 124-frame training window')
    wanted, decoded = set(indices), {}
    with av.open(video) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 1
        for index, frame in enumerate(container.decode(stream)):
            if index in wanted:
                decoded[index] = resize_crop(frame.to_ndarray(format='rgb24'))
            if index >= indices[-1]:
                break
    if set(decoded) != wanted:
        raise ValueError(f'Missing video frames: {video}')
    return [decoded[index] for index in indices]


def write_video(path: Path, frames: list[np.ndarray], fps: float) -> None:
    with av.open(str(path), mode='w') as container:
        stream = container.add_stream('libx264', rate=Fraction(str(fps)))
        stream.width, stream.height = frames[0].shape[1], frames[0].shape[0]
        stream.pix_fmt = 'yuv420p'
        stream.options = dict(crf='20', preset='fast', threads='1')
        for rgb in frames:
            for packet in stream.encode(av.VideoFrame.from_ndarray(rgb, format='rgb24')):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def banner(caption: str, title: str, width: int) -> np.ndarray:
    font, title_font = ImageFont.truetype(FONT, 17), ImageFont.truetype(FONT, 18)
    lines, current = [], ''
    for word in caption.split():
        new = (current + ' ' + word).strip()
        if current and font.getlength(new) > width - 28:
            lines.append(current)
            current = word
        else:
            current = new
    if current:
        lines.append(current)
    height = 74 + len(lines) * 22
    height += height % 2
    im = Image.new('RGB', (width, height), (20, 24, 30))
    draw = ImageDraw.Draw(im)
    draw.text((14, 8), title, font=title_font, fill=(255, 220, 40))
    draw.text((14, 32),
              'GT TRAIN WINDOW | numeric camera INPUT (interpolated display); not measured video motion',
              font=font,
              fill=(180, 210, 255))
    for i, line in enumerate(lines):
        draw.text((14, 60 + i * 22), line, font=font, fill='white')
    return np.asarray(im)


def verify_video(path: Path, count: int, fps: float) -> None:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if abs(float(stream.average_rate) - fps) > .001 or sum(1 for _ in container.decode(stream)) != count:
            raise ValueError(f'Output duration/frame count mismatch: {path}')


def render_case(job: tuple[dict, str]) -> dict:
    spec, output = job
    cv2.setNumThreads(1)
    started = time.monotonic()
    dest = Path(output) / spec['source_name'] / spec['case_name']
    dest.mkdir(parents=True, exist_ok=True)
    meta_path = dest / 'metadata.json'
    if (dest / 'COMPLETE').exists():
        previous = json.loads(meta_path.read_text())
        if previous['id'] != spec['id'] or previous['caption_sha256'] != spec['caption_sha256']:
            raise ValueError('Existing visualization has different inputs')
        for name in ('raw.mp4', 'visualized.mp4'):
            verify_video(dest / name, 124, previous['fps'])
        return previous
    actual_path = Path(spec['conditioned_parquet'])
    sidecar = json.loads(actual_path.with_suffix('.audit.json').read_text())
    columns = ['id', 'source_name', 'caption', 'source_frame_indices', 'fps', 'source_fps', 'num_frames']
    for stem in ('camera_to_world', 'camera_intrinsics'):
        columns += [stem + suffix for suffix in ('_bytes', '_shape', '_dtype')]
    rows = pq.ParquetFile(actual_path).read(columns=columns, use_threads=False).to_pylist()
    if len(rows) != 1:
        raise ValueError('Expected one-row actual training Parquet')
    row = rows[0]
    for key in ('id', 'source_name', 'caption', 'source_frame_indices'):
        if row[key] != spec[key]:
            raise ValueError(f'Actual training row mismatch: {key}')
    for key in ('id', 'video', 'source_frame_indices', 'caption_sha256'):
        if sidecar[key] != spec[key]:
            raise ValueError(f'Actual training provenance mismatch: {key}')
    if digest(row['caption'].encode()) != spec['caption_sha256']:
        raise ValueError('Actual caption SHA mismatch')
    camera = np.frombuffer(row['camera_to_world_bytes'],
                           dtype=row['camera_to_world_dtype']).reshape(row['camera_to_world_shape'])
    intrinsics = np.frombuffer(row['camera_intrinsics_bytes'],
                               dtype=row['camera_intrinsics_dtype']).reshape(row['camera_intrinsics_shape'])
    for name, key in (('camera_to_world', 'camera_sha256'), ('camera_intrinsics', 'intrinsics_sha256')):
        if digest(row[name + '_bytes']) != sidecar[key]:
            raise ValueError(f'Actual training tensor hash mismatch: {name}')
    expanded, anchors = expand_camera(camera)
    dt, angular, threshold = local_controls(expanded)
    source = pq.ParquetFile(spec['source_parquet']).read(columns=['id', 'first_frame_png'],
                                                         use_threads=False).to_pylist()[spec['source_row']]
    if source['id'] != spec['id'] or digest(source['first_frame_png']) != sidecar['first_frame_png_sha256']:
        raise ValueError('Cached training first-frame provenance mismatch')
    (dest / 'first_frame.png').write_bytes(source['first_frame_png'])
    frames = load_window(spec['video'], spec['source_frame_indices'])
    cached = np.asarray(Image.open(BytesIO(source['first_frame_png'])).convert('RGB'))
    first_mae = float(np.abs(frames[0].astype(float) - cached).mean())
    # Fail rather than hide a crop/window disagreement by replacing RGB frame0.
    if first_mae > 3.:
        raise ValueError(f'Redecoded versus cached first-frame MAE {first_mae:.3f} > 3: {spec["id"]}')
    np.save(dest / 'camera_to_world_37.npy', camera)
    np.save(dest / 'camera_to_world_display_124.npy', expanded)
    np.save(dest / 'camera_intrinsics_37.npy', intrinsics)
    fps = float(row['fps'])
    write_video(dest / 'raw.mp4', frames, fps)
    drawers = load_drawers()
    pyr = drawers['_se3_to_pyr_hy_yup'](expanded)
    panel = drawers['TrajectoryPanelRenderer'](out_h=480, out_w=960)
    panel.set_sequence(expanded[:, :3, 3], pyr, view_se3=expanded)
    header = banner(row['caption'], f'{spec["source_name"]} | {spec["id"]} | 124f @ {fps:g}fps', 1792)
    states, previews = [], []
    trails: list = []
    try:
        # Streaming output: never accumulate the large annotated frames.
        with av.open(str(dest / 'visualized.mp4'), mode='w') as container:
            stream = container.add_stream('libx264', rate=Fraction(str(fps)))
            stream.width, stream.height, stream.pix_fmt = 1792, 480 + len(header), 'yuv420p'
            stream.options = dict(crf='22', preset='fast', threads='1')
            for i, frame in enumerate(frames):
                keys = set()
                if abs(dt[i, 2]) > threshold:
                    keys.add('W' if dt[i, 2] > 0 else 'S')
                if abs(dt[i, 0]) > threshold:
                    keys.add('D' if dt[i, 0] > 0 else 'A')
                pitch, yaw, roll = angular[i]
                rotations = ('I' if pitch > 0 else 'K') if abs(pitch) > 1e-4 else ''
                rotations += ('L' if yaw > 0 else 'J') if abs(yaw) > 1e-4 else ''
                bgr = drawers['_draw_actions_on_frame'](cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                                                        keys,
                                                        float(pitch),
                                                        float(yaw),
                                                        trails,
                                                        100.,
                                                        3,
                                                        False,
                                                        *map(float, pyr[i]),
                                                        current_position=tuple(expanded[i, :3, 3]),
                                                        xz_delta=(float(dt[i, 0]), float(dt[i, 2])))
                cv2.putText(bgr,
                            f'{i + 1}/124 src={spec["source_frame_indices"][i]} | {"".join(sorted(keys))}{rotations}',
                            (430, 30), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 255), 2)
                for trail in trails:
                    trail['ttl'] -= 1
                trails = [trail for trail in trails if trail['ttl'] > 0]
                rgb = np.concatenate([cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), panel.render(i)], axis=1)
                rgb = np.concatenate([header, rgb], axis=0)
                for packet in stream.encode(av.VideoFrame.from_ndarray(rgb, format='rgb24')):
                    container.mux(packet)
                if i in (0, 41, 82, 123):
                    previews.append(Image.fromarray(rgb))
                states.append(
                    dict(frame=i,
                         source_frame=spec['source_frame_indices'][i],
                         keys=sorted(keys),
                         rotation_labels=rotations,
                         local_translation=dt[i].tolist(),
                         local_pitch_yaw_roll_deg=angular[i].tolist()))
            for packet in stream.encode():
                container.mux(packet)
    finally:
        panel.close()
    for name in ('raw.mp4', 'visualized.mp4'):
        verify_video(dest / name, 124, fps)
    previews[0].save(dest / 'preview.jpg', quality=90)
    strip = Image.new('RGB', (896 * 2, previews[0].height))
    half_height = previews[0].height // 2
    for i, preview in enumerate(previews):
        strip.paste(preview.resize((896, half_height)), ((i % 2) * 896, (i // 2) * half_height))
    strip.save(dest / 'contact_sheet.jpg', quality=90)
    save_json(dest / 'controls.json', states)
    metadata = dict(spec,
                    fps=fps,
                    num_frames=124,
                    source_fps=row['source_fps'],
                    original_camera_shape=list(camera.shape),
                    display_camera_interpolation='SLERP rotations + linear translation',
                    h3_raw_frame_anchors=anchors.tolist(),
                    video_is='redecoded original GT window, not VAE reconstruction',
                    first_frame_redecode_mae=first_mae,
                    sources_read_only=True,
                    split_verified='train',
                    camera_condition_sha256=sidecar['camera_sha256'],
                    presentation_move_threshold=threshold,
                    translation_units='persisted training units; not necessarily physical meters',
                    hud='numeric camera input; not estimated visual camera motion',
                    renderer=str(RENDERER),
                    renderer_sha256=digest(RENDERER.read_bytes()),
                    full_prompt_displayed=True,
                    output_directory=str(dest),
                    elapsed_seconds=time.monotonic() - started,
                    version=VERSION)
    save_json(meta_path, metadata)
    (dest / 'COMPLETE').touch()
    return metadata


def write_index(output: Path, results: list[dict], counts: dict, plan_sha: str) -> None:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in results:
        grouped[row['source_name']].append(row)
    parts = [
        '<!doctype html><meta charset="utf-8"><title>当前训练数据可视化</title>',
        '<style>body{font:16px sans-serif;background:#15181e;color:#ddd;margin:28px}'
        'a{color:#83c9ff}video{width:100%}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:24px}'
        '.card{background:#222833;padding:16px;overflow-wrap:anywhere}img{width:100%}p{line-height:1.5}</style>',
        '<h1>当前 FastVideoWM 训练数据：每来源最多5条</h1>',
        f'<p>仅 active train split；共{len(results)}条／{len(counts)}个来源。固定哈希抽样，不按画面好坏挑选。'
        '所有条目来自已通过验收的实际训练清单，不含validation和83条排除数据。'
        '这是当前训练输入巡检，并不保证已有自动过滤没有漏检。</p>', '<p>视频为训练对应的原始GT窗口（124帧、24fps、480×832），不是latent解码。'
        '轨迹读取实际训练Parquet的37个C2W锚点，SLERP＋线性插值仅用于显示124帧；'
        '坐标为X右/Y上/Z前，I上转、K下转、J左转、L右转。平移单位是训练单位，不一定是米。'
        'WASD阈值仅用于显示，数值、原始锚点、完整文本均保留。</p>',
        '<p><a href="manifest.jsonl">样本manifest</a> | <a href="summary.json">来源统计及校验</a></p>'
    ]
    for source, count in sorted(counts.items()):
        parts.append(
            f'<h2>{html.escape(source)} — 训练{count["train_rows"]}条，展示{count["selected"]}条</h2><div class="grid">')
        for row in sorted(grouped[source], key=lambda r: r['case_name']):
            rel = f'{source}/{row["case_name"]}'
            parts.append(
                f'<div class="card"><b>{html.escape(row["id"])}</b>'
                f'<video controls preload="none" poster="{rel}/preview.jpg" src="{rel}/visualized.mp4"></video>'
                f'<p>{html.escape(row["caption"])}</p>'
                f'<p><a href="{rel}/raw.mp4">无叠加GT</a> | <a href="{rel}/contact_sheet.jpg">四帧图</a> | '
                f'<a href="{rel}/first_frame.png">训练首帧</a> | <a href="{rel}/metadata.json">完整信息</a> | '
                f'<a href="{rel}/camera_to_world_37.npy">训练C2W</a> | <a href="{rel}/controls.json">逐帧控制</a></p>'
                f'<details><summary>来源视频</summary>{html.escape(row["video"])}</details></div>')
        parts.append('</div>')
    (output / 'index.html').write_text('\n'.join(parts))
    (output / 'manifest.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in results))
    save_json(
        output / 'summary.json',
        dict(version=VERSION,
             selected=len(results),
             sources=counts,
             split='train',
             plan_sha256=plan_sha,
             actual_train_rows=sum(c['train_rows'] for c in counts.values()),
             selected_source_counts=dict(Counter(r['source_name'] for r in results)),
             excluded_rows_visualized=0,
             gpu_used=False,
             max_first_frame_redecode_mae=max(r['first_frame_redecode_mae'] for r in results)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--per-source', type=int, default=5)
    parser.add_argument('--seed', type=int, default=20261027)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--limit', type=int, default=0, help='Smoke test only; do not mark as full delivery')
    args = parser.parse_args()
    if args.per_source < 1 or not 1 <= args.workers <= 4:
        parser.error('per-source >=1 and 1..4 bounded CPU workers required')
    selected, counts, plan_sha = choose_samples(args.root, args.per_source, args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    save_json(args.output / 'selection.json', dict(rows=selected, sources=counts, plan_sha256=plan_sha, seed=args.seed))
    if args.limit:
        selected = selected[:args.limit]
    results = []
    failures = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = {pool.submit(render_case, (row, str(args.output))): row for row in selected}
        for future in as_completed(futures):
            row = futures[future]
            try:
                result = future.result()
                results.append(result)
                print(f'VIS_READY {len(results)}/{len(selected)} {row["id"]} {result["elapsed_seconds"]:.1f}s',
                      flush=True)
            except Exception as error:
                failures.append(dict(id=row['id'], error=repr(error)))
                print('VIS_FAILED', row['id'], repr(error), flush=True)
            save_json(args.output / 'progress.json', dict(completed=len(results),
                                                          total=len(selected),
                                                          failures=failures))
    save_json(args.output / 'failures.json', failures)
    if results:
        write_index(args.output, results, counts, plan_sha)
    if failures:
        raise RuntimeError(f'{len(failures)} cases failed; no rejected-data substitution')
    if not args.limit:
        (args.output / 'VISUALIZATION_COMPLETE').touch()
    print('VISUALIZATION_FINISHED', len(results), str(args.output / 'index.html'), flush=True)


if __name__ == '__main__':
    main()
