"""CPU-only, streaming Cosmos-style INPUT HUD; raw videos are never modified."""
from __future__ import annotations

import argparse
import ast
from concurrent.futures import ProcessPoolExecutor
import fcntl
from fractions import Fraction
import hashlib
import html
import json
import math
import os
from pathlib import Path
import time
from typing import Any
from collections.abc import Callable

import av
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation, Slerp

COSMOS_UTILS = Path(
    '/home/lizihao/projects/military_world_model_agent/cosmos-predict2-worldmodel/eval/methods/utils.py')
FONT = Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
UI_VERSION = 'cosmos_input_hud_prompt_v1'
cv2.setNumThreads(1)


def file_hash(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(f'.tmp.{os.getpid()}')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(temp, path)


def load_cosmos_drawer() -> Callable[..., Any]:
    # Reuse the requested repo's actual four pure OpenCV drawing functions.
    # Loading just these definitions avoids importing mediapy and the rest of
    # Cosmos's video helpers into every FastVideo GPU decode process.
    names = {'_blend_overlay', '_draw_panel_badge', '_extract_mouse_deltas', 'draw_actions_genie3_style'}
    tree = ast.parse(COSMOS_UTILS.read_text())
    definitions: list[ast.stmt] = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    if {node.name for node in definitions if isinstance(node, ast.FunctionDef)} != names:
        raise ValueError('Cosmos renderer API changed; inspect before visualizing')
    namespace = {'cv2': cv2, 'np': np, 'math': math}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(COSMOS_UTILS), 'exec'), namespace)
    return namespace['draw_actions_genie3_style']


def text_is_used(meta: dict[str, Any]) -> bool:
    value = meta.get('scene_or_subject_text_used', meta.get('text_condition_used', False))
    if isinstance(value, bool):
        return value
    if str(value).lower() in ('true', 'false'):
        return str(value).lower() == 'true'
    raise ValueError('Unknown text-conditioning state; do not mislabel the prompt')


def numeric_controls(camera: np.ndarray, frames: int) -> list[dict[str, Any]]:
    camera = np.asarray(camera, dtype=np.float64)
    if camera.ndim != 3 or camera.shape[1:] != (4, 4) or not np.isfinite(camera).all():
        raise ValueError('Invalid actual generated-input camera path')
    if len(camera) == 37 and frames == 124:
        anchors = np.cumsum(np.resize([1, 4, 4, 4, 4], 37)) - 1
    else:
        anchors = np.linspace(0, frames - 1, len(camera))
    rotations = Slerp(anchors, Rotation.from_matrix(camera[:, :3, :3]))(np.arange(frames)).as_matrix()
    translation = np.stack([np.interp(np.arange(frames), anchors, camera[:, axis, 3]) for axis in range(3)], axis=-1)
    dt = np.zeros((frames, 3))
    angles = np.zeros((frames, 3))
    dt[1:] = np.einsum('tji,tj->ti', rotations[:-1], np.diff(translation, axis=0))
    relative = np.swapaxes(rotations[:-1], 1, 2) @ rotations[1:]
    rv = Rotation.from_matrix(relative).as_rotvec() * 180 / np.pi
    # Camera coordinates: X-right, Y-up, Z-forward. I-up is negative Rx.
    angles[1:] = rv * np.array([-1, 1, -1])
    threshold = max(float(np.median(np.linalg.norm(dt[1:], axis=-1))) * .2, 1e-6)
    states = []
    for movement, angular in zip(dt, angles, strict=False):
        keys = []
        if abs(movement[2]) > threshold:
            keys.append('W' if movement[2] > 0 else 'S')
        if abs(movement[0]) > threshold:
            keys.append('D' if movement[0] > 0 else 'A')
        states.append(
            dict(keys=keys,
                 pitch=float(angular[0]),
                 yaw=float(angular[1]),
                 roll=float(angular[2]),
                 translation=movement.tolist(),
                 angular_units='degrees_per_output_frame'))
    return states


def constant_controls(action: str, frames: int) -> list[dict[str, Any]]:
    if action not in 'WASDIJKL' or len(action) != 1:
        raise ValueError(f'Unsupported camera text instruction: {action}')
    pitch = {'I': 1., 'K': -1.}.get(action, 0.)
    yaw = {'J': -1., 'L': 1.}.get(action, 0.)
    return [
        dict(keys=[action] if action in 'WASD' else [],
             pitch=pitch,
             yaw=yaw,
             roll=0.,
             angular_units='categorical_camera_text_display_only') for _ in range(frames)
    ]


def wrap_text(text: str, font: Any, width: int) -> list[str]:
    lines = []
    current = ''
    for word in str(text).split():
        candidate = (current + ' ' + word).strip()
        if current and font.getlength(candidate) > width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def prompt_banner(meta: dict[str, Any], width: int) -> tuple[np.ndarray, str]:
    font = ImageFont.truetype(str(FONT), 16)
    title = ImageFont.truetype(str(FONT), 17)
    enabled = text_is_used(meta)
    prompt = str(meta.get('scene_prompt', meta.get('prompt', ''))) if enabled else ''
    if enabled and not prompt.strip():
        raise ValueError('Text-enabled case has no actual prompt')
    lines = wrap_text(prompt, font, width - 32) if enabled else ['No scene/subject text tokens were passed to the DiT.']
    height = max(200, 60 + len(lines) * 21)
    height += (height % 2)
    image = Image.new('RGB', (width, height), (17, 20, 25))
    draw = ImageDraw.Draw(image)
    draw.text((16, 10), 'PROMPT INPUT: ' + ('ON' if enabled else 'OFF'), font=title, fill=(255, 210, 0))
    for index, line in enumerate(lines):
        draw.text((16, 37 + index * 21), line, font=font, fill=(235, 235, 235))
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR), prompt


def validate_overlay(path: str | Path, frames: int, width: int, height: int, fps: float) -> None:
    cap = cv2.VideoCapture(str(path))
    count = 0
    try:
        if not cap.isOpened() or abs(cap.get(cv2.CAP_PROP_FPS) - fps) > .001:
            raise ValueError(f'Invalid HUD MP4/FPS: {path}')
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame.shape != (height, width, 3):
                raise ValueError('HUD video dimensions changed')
            count += 1
    finally:
        cap.release()
    if count != frames:
        raise ValueError(f'HUD frame count mismatch: {count} != {frames}')


def annotate_case(metadata_path: str | Path, root: str | Path) -> dict[str, Any]:
    metadata_path, root = Path(metadata_path), Path(root)
    meta = json.loads(metadata_path.read_text())
    raw = metadata_path.parent / 'raw.mp4'
    relative = metadata_path.parent.relative_to(root / 'videos')
    dest = (root / 'visualized' / relative).with_suffix('.mp4')
    sidecar = dest.with_suffix('.hud.json')
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.with_suffix('.hud.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        fingerprints = dict(raw_sha256=file_hash(raw),
                            metadata_sha256=file_hash(metadata_path),
                            renderer_sha256=file_hash(COSMOS_UTILS),
                            hud_script_sha256=file_hash(Path(__file__)),
                            ui_version=UI_VERSION)
        camera_path = metadata_path.parent / 'camera_to_world.npy'
        if camera_path.exists():
            fingerprints['camera_sha256'] = file_hash(camera_path)
        if dest.exists() and sidecar.exists():
            prior = json.loads(sidecar.read_text())
            if prior['fingerprints'] == fingerprints and prior['output_sha256'] == file_hash(dest):
                return dict(output=str(dest), status='verified_existing')
            raise ValueError(f'Existing HUD has different provenance; use a new visualization directory: {dest}')
        cap = cv2.VideoCapture(str(raw))
        if not cap.isOpened():
            raise ValueError(f'Cannot open raw video: {raw}')
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if camera_path.exists():
            controls = numeric_controls(np.load(camera_path), frames)
            control_label = f"INPUT numeric C2W | alpha={meta.get('alpha',1):g}"
            control_source = 'actual generated-input numeric C2W; 37 anchors expanded onto native video frames'
        else:
            controls = constant_controls(meta['action'], frames)
            control_label = f"INPUT camera text | {meta['action']}"
            control_source = 'actual constant per-latent camera text; not a numeric trajectory estimate'
        banner, prompt = prompt_banner(meta, width)
        draw_actions = load_cosmos_drawer()
        temporary = dest.with_name(dest.stem + f'.tmp.{os.getpid()}.mp4')
        rate = Fraction(str(fps)).limit_denominator(1_000_000)
        trails: list[dict[str, Any]] = []
        count = 0
        try:
            with av.open(str(temporary), 'w') as container:
                stream = container.add_stream('libx264', rate=rate, options={'crf': '18', 'preset': 'fast'})
                stream.width = width
                stream.height = height + len(banner)
                stream.pix_fmt = 'yuv420p'
                stream.codec_context.thread_count = 2
                for index, state in enumerate(controls):
                    ok, frame = cap.read()
                    if not ok:
                        raise ValueError('Raw video ended before declared frame count')
                    # Cosmos uses a 0.3 UI threshold. Gain8 is ONLY a display
                    # scale so slow rotations remain visible, never a model input.
                    gain = 1. if state['angular_units'].startswith('categorical') else 8.
                    pitch, yaw, roll = (state[k] * gain for k in ('pitch', 'yaw', 'roll'))
                    magnitude = math.hypot(pitch, yaw)
                    if magnitude > .05:
                        scale = min(max(magnitude * 6, 24), 140)
                        direction = math.atan2(-pitch, yaw)
                        trails.append(
                            dict(start=(width // 2, height // 2),
                                 end=(int(width / 2 + math.cos(direction) * scale),
                                      int(height / 2 + math.sin(direction) * scale)),
                                 ttl=3))
                    frame = draw_actions(frame,
                                         set(state['keys']),
                                         dict(pitch_delta=pitch, yaw_delta=yaw, roll_delta=roll),
                                         trails,
                                         3,
                                         panel_label='GEN | ' + str(meta.get('id', metadata_path.parent.name)),
                                         control_label=control_label)
                    for trail in trails:
                        trail['ttl'] -= 1
                    trails = [trail for trail in trails if trail['ttl'] > 0]
                    displayed = np.vstack([frame, banner.copy()])
                    cv2.putText(displayed, f"INPUT display, not measured motion | {index+1}/{frames} | {fps:.3f} FPS",
                                (16, height + len(banner) - 12), cv2.FONT_HERSHEY_SIMPLEX, .44, (170, 180, 190), 1,
                                cv2.LINE_AA)
                    if index == frames // 2:
                        cv2.imwrite(str(dest.with_suffix('.jpg')), displayed)
                    output = av.VideoFrame.from_ndarray(displayed, format='bgr24')
                    output.pts = index
                    output.time_base = 1 / rate
                    for packet in stream.encode(output):
                        container.mux(packet)
                    count += 1
                for packet in stream.encode():
                    container.mux(packet)
            validate_overlay(temporary, frames, width, height + len(banner), fps)
            os.replace(temporary, dest)
        finally:
            cap.release()
            if temporary.exists():
                temporary.unlink()
        atomic_json(
            sidecar,
            dict(fingerprints=fingerprints,
                 output_sha256=file_hash(dest),
                 raw=str(raw),
                 metadata=str(metadata_path),
                 output=str(dest),
                 frames=count,
                 fps=fps,
                 width=width,
                 height=height + len(banner),
                 prompt_used=text_is_used(meta),
                 displayed_prompt=prompt,
                 source_prompt_not_injected=meta.get('prompt') if not text_is_used(meta) else None,
                 controls=controls,
                 control_source=control_source,
                 rotation_display_gain=8 if camera_path.exists() else 1,
                 original_video_untouched=True,
                 metrics_must_use_raw=True))
        print('HUD_COMPLETE', dest, flush=True)
        return dict(output=str(dest), status='created')


def _job(pair: tuple[Path, Path]) -> dict[str, Any]:
    return annotate_case(*pair)


def render_root(root: str | Path, workers: int = 4) -> bool:
    root = Path(root)
    metadata = [
        p for p in sorted((root / 'videos').rglob('metadata.json')) if (p.parent / 'raw.mp4').exists() and any(
            (p.parent / marker).exists() for marker in ('VIDEO_COMPLETE', 'COMPLETE'))
    ]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_job, [(p, root) for p in metadata]))
    rows = []
    for result in results:
        path = Path(result['output'])
        sidecar = json.loads(path.with_suffix('.hud.json').read_text())
        rel = str(path.relative_to(root / 'visualized'))
        rows.append(f'<tr><td><a href="{html.escape(rel)}">{html.escape(rel)}</a></td>'
                    f'<td>{"ON" if sidecar["prompt_used"] else "OFF"}</td></tr>')
    visualized = root / 'visualized'
    visualized.mkdir(parents=True, exist_ok=True)
    (visualized / 'index.html'
     ).write_text('<meta charset="utf-8"><h1>Input-control visualization</h1>'
                  '<p>Input HUD, not measured generated-video motion. Metrics use original raw videos.</p><table>' +
                  ''.join(rows) + '</table>')
    manifest = root / 'inference_manifest.json'
    expected = len(json.loads(manifest.read_text())) if manifest.exists() else None
    complete = expected is not None and len(results) == expected
    atomic_json(
        root / 'visualization_summary.json',
        dict(videos=len(results),
             expected_videos=expected,
             complete=complete,
             output=str(visualized),
             renderer=str(COSMOS_UTILS),
             raw_untouched=True))
    if complete:
        (root / 'VISUALIZATION_COMPLETE').touch()
    print('HUD_ROOT', root, len(results), '/', expected, flush=True)
    return complete


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--roots', type=Path, nargs='+', required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--watch-until', type=Path, help='Wait for this generation root and its HUD to complete')
    args = parser.parse_args()
    roots = args.roots
    while True:
        done = {root: render_root(root, args.workers) for root in roots}
        if not args.watch_until:
            break
        if done.get(args.watch_until) and (args.watch_until / 'INFERENCE_COMPLETE').exists():
            break
        if (args.watch_until / 'NEEDS_ATTENTION').exists():
            raise RuntimeError('Generation failed; existing HUDs retained')
        roots = [args.watch_until]
        time.sleep(300)


if __name__ == '__main__':
    main()
