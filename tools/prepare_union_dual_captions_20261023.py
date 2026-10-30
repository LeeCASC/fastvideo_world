"""Exact-window dual captions, with read-only reuse of the 115,988-row H3 union.

Descriptions are alternatives, never concatenated. Background-only excludes
foreground subjects as well as their actions. Numeric camera inputs are shared.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any
from collections.abc import Iterator

import numpy as np
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
BASE = Path('/datacc05/lizihao/worldmodel')
DEFAULT_ROOT = BASE / 'fastvideowm_union_dual_text_20261023'
OLD_RAW = BASE / 'fastvideowm_h3_latent_approved_v1_20260925/intermediate/all'
OLD_MAP = BASE / 'fastvideowm_approved_uncached_v1_20260925/source_mapping.jsonl'
OLD_SPLITS = BASE / 'fastvideowm_action_only_v1_20260925'
NEW_RAW = BASE / 'vwm_unified_124_243_20261016/124f/vae_latents/minimax_h3_prope_union_input'
NEW_REGISTRY = BASE / 'vwm_unified_124_243_20261016/124f/vlm_qwen3vl_strict_50599/registry_strict_50599.jsonl'
TEXT_ROOT = BASE / 'fastvideowm_scene_text_20261022'
POLICY = 'exact_window_dual_subject_motion_background_only_v1'
RECOVERY_VERSION = 'dual_caption_tail_recovery_20261024_v2'
CAMERA = re.compile(
    r'\b(camera|view|shot|viewpoint|framing|perspective|pans?|panned|panning|tilts?|tilted|tilting|zooms?|zoomed|zooming|tracking|dolly|soundtrack|audio)\b',
    re.I)
FOREGROUND = re.compile(
    r'\b(tanks?|vehicles?|cars?|trucks?|buses?|trains?|locomotives?|aircraft|helicopters?|planes?|drones?|ships?|boats?|people|persons?|men|women|man|woman|children|child|soldiers?|drivers?|animals?|dogs?|cats?|pedestrians?|bicycles?|motorcycles?|tractors?|robots?|actors?|subjects?)\b',
    re.I)
MOTION = re.compile(
    r'\b(moving|moves|drives|driving|walks|walking|runs|running|flies|flying|turns|turning|advances|advancing|reverses|reversing|backward|backwards|forward|forwards|approaches|approaching|follows|following|drifts|drifting|stationary)\b',
    re.I)
ABSENCE_ONLY = re.compile(
    r'^(?:(?:there\s+(?:is|are)\s+)?no\s+(?:(?:visible|foreground|main|independent|physical|discernible|observable|apparent)\s+){0,3}(?:subjects?|people|persons?|objects?|actors?|movements?|motion|activity)\b|'
    r'(?:no|not\s+any)\s+(?:subject|object|physical)\s+motion\b)', re.I)
STATIC_ONLY = re.compile(
    r'^(?:the\s+)?(?:buildings?|objects?|scene|environment|landscape|structures?)\s+'
    r'(?:remain|remains|are|is|appear|appears)\s+(?:static|unchanged|motionless|stationary)\b', re.I)


def camera_text(text: str) -> bool:
    # Architectural framing is a material/structure, not cinematic framing.
    checked = re.sub(r'\b(?:wooden|wood|timber|steel|metal|window|door)\s+framing\b',
                     'structural supports',
                     text,
                     flags=re.I)
    return bool(CAMERA.search(checked))


def foreground_subject_text(text: str) -> bool:
    # Only unambiguous infrastructure nouns are exempt. Bare "tank" in an
    # industrial scene still requires regeneration, not a guessed sense.
    checked = re.sub(r'\b(?:storage|water|oil|fuel|gas|septic|spherical|cylindrical)\s+tanks?\b',
                     'storage containers',
                     text,
                     flags=re.I)
    # Railway infrastructure, concession stands, and a clothing-store label
    # are not foreground trains, animals, or people. Keep actual actors banned.
    checked = re.sub(r'\btrain\s+(?:stations?|platforms?|tracks?|yards?)\b',
                     'railway infrastructure',
                     checked,
                     flags=re.I)
    checked = re.sub(r'\bhot\s+dog\s+(?:carts?|stands?|stalls?)\b', 'food stands', checked, flags=re.I)
    checked = re.sub(r"\bmen['’]s\s+clothing\s+store\b", 'clothing store', checked, flags=re.I)
    return bool(FOREGROUND.search(checked))


def background_violation(text: str) -> bool:
    if foreground_subject_text(text):
        return True
    sentences = re.split(r'(?<=[.!?])\s+', text)
    for sentence in sentences:
        # A boardwalk WITH TABLES AND BENCHES runs through a forest is still
        # static geometry. Do not blanket-allow "runs" in actor sentences.
        if re.match(
                r'^(?:(?:a|an|the)\s+)?(?:[\w-]+\s+){0,4}'
                r'(?:road|street|river|stream|path|trail|lane|route|boardwalk|walkway|fence|wall)\b',
                sentence,
                flags=re.I):
            sentence = re.sub(r'\b(?:runs?|turns?)\b', 'extends', sentence, flags=re.I)
        if MOTION.search(sentence):
            return True
    return False


def read_jsonl(path: Path | str, *, allow_truncated_tail: bool = False) -> Iterator[dict[str, Any]]:
    with Path(path).open() as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    if allow_truncated_tail and not line.endswith('\n'):
                        print(f'PRESERVED_TRUNCATED_TAIL {path}:{number}', file=sys.stderr, flush=True)
                        return
                    raise


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def file_digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f'.tmp.{os.getpid()}')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f'.tmp.{os.getpid()}')
    with temporary.open('w') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    os.replace(temporary, path)


def selection_hash(root: Path) -> str | None:
    path = root / 'caption_selection.json'
    return file_digest(path) if path.exists() else None


def registry_specs(root: Path) -> list[dict[str, Any]]:
    """Apply only the explicitly authorized, hash-bound caption exclusion list."""
    specs = list(read_jsonl(root / 'registry.jsonl'))
    path = root / 'caption_selection.json'
    if not path.exists():
        return specs
    selection = json.loads(path.read_text())
    if (selection.get('authorization') != 'user_20261024_discard_persistent_caption_failures'
            or selection.get('registry_sha256') != file_digest(root / 'registry.jsonl')
            or selection.get('original_rows') != len(specs)):
        raise ValueError('Invalid/stale authorized caption selection')
    excluded = selection.get('excluded', [])
    by_id = {spec['id']: spec for spec in specs}
    ids = [row['id'] for row in excluded]
    if not 1 <= len(ids) <= 83 or len(set(ids)) != len(ids):
        raise ValueError('Caption exclusions must be 1-83 unique explicitly recorded IDs')
    for row in excluded:
        spec = by_id.get(row['id'])
        if not spec or not row.get('reason') or any(
                row.get(key) != spec.get(key) for key in ('source_name', 'video', 'window_id', 'source_frame_indices')):
            raise ValueError('Caption exclusion source/window mismatch')
    included = [spec for spec in specs if spec['id'] not in set(ids)]
    if selection.get('included_rows') != len(included):
        raise ValueError('Caption selection count mismatch')
    return included


def exclude_failed(args: argparse.Namespace) -> None:
    """Freeze a small persistent-failure list only after explicit user approval."""
    import fcntl
    if not args.authorize_exclude_failed:
        raise ValueError('Explicit user authorization is required to exclude failed captions')
    with (args.root / 'chain.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (args.root / 'caption_selection.json').exists():
            raise ValueError('Caption selection already frozen; do not overwrite it')
        variant = args.root / 'subject_motion'
        if any((variant / 'preprocessed').rglob('*.parquet')) or any(
            (variant / 'runs/train/checkpoints').glob('checkpoint-*')):
            raise ValueError('Do not change caption selection after encoding/training starts')
        specs = list(read_jsonl(args.root / 'registry.jsonl'))
        ids_path = getattr(args, 'exclude_ids_path', None)
        if ids_path:
            requested = list(read_jsonl(ids_path))
            requested_ids = [row['id'] for row in requested]
            if len(requested_ids) != len(
                    set(requested_ids)) or not set(requested_ids).issubset({spec['id']
                                                                            for spec in specs}):
                raise ValueError('Explicit exclusion IDs must be unique members of the original registry')
            excluded = [
                dict(spec, reason='user_discarded_all_initial_83_caption_retry_candidates') for spec in specs
                if spec['id'] in set(requested_ids)
            ]
        else:
            captions, _, failures = collect_captions(args.root, specs)
            excluded = [
                dict(spec, reason=failures.get(spec['id'], {}).get('error') or 'missing_valid_dual_caption')
                for spec in specs if spec['id'] not in captions
            ]
        if not 1 <= len(excluded) <= 83:
            raise ValueError(f'Expected 1-83 persistent caption failures, found {len(excluded)}')
        selection = dict(authorization='user_20261024_discard_persistent_caption_failures',
                         registry_sha256=file_digest(args.root / 'registry.jsonl'),
                         original_rows=len(specs),
                         included_rows=len(specs) - len(excluded),
                         excluded=excluded,
                         source_files_deleted=False,
                         source_counts=dict(Counter(row['source_name'] for row in excluded)))
        atomic_json(args.root / 'caption_selection.json', selection)
        write_jsonl(args.root / 'caption_excluded_ids.jsonl', excluded)
        registry_specs(args.root)
        print('CAPTION_SELECTION_FROZEN',
              json.dumps({
                  k: v
                  for k, v in selection.items() if k != 'excluded'
              }),
              flush=True)


def window_key(video: str | Path, indices: list[int]) -> tuple[str, tuple[int, ...]]:
    return str(Path(video).resolve()), tuple(int(i) for i in indices)


def validate_pair(pair: dict[str, Any]) -> dict[str, Any]:
    motion = str(pair.get('subject_motion_description', '')).strip()
    background = str(pair.get('background_description', '')).strip()
    if not 5 <= len(motion.split()) <= 90 or not 5 <= len(background.split()) <= 60:
        raise ValueError('empty, uninformative or oversized caption')
    if camera_text(motion) or camera_text(background):
        raise ValueError('camera/audio text in a caption')
    if background_violation(background):
        raise ValueError('foreground subject or motion in background-only caption')
    if all(ABSENCE_ONLY.search(s.strip()) for s in re.split(r'(?<=[.!?])\s+', motion)):
        raise ValueError('absence statement is not an informative scene caption')
    if 'confidence' not in pair:
        raise ValueError('missing confidence; regenerate caption')
    confidence = float(pair['confidence'])
    if not 0 <= confidence <= 1:
        raise ValueError('invalid confidence')
    if not isinstance(pair.get('quality_flags', []), list):
        raise ValueError('invalid quality_flags')
    if pair.get('subject_motion_status') not in ('observed', 'uncertain', 'not_observed'):
        raise ValueError('invalid subject_motion_status')
    presence = pair.get('subject_presence')
    if presence is not None and presence not in ('visible', 'none', 'uncertain'):
        raise ValueError('invalid subject_presence')
    if presence == 'none' and pair['subject_motion_status'] != 'not_observed':
        raise ValueError('no visible subject requires not_observed motion')
    if presence == 'none' and background_violation(motion):
        raise ValueError('subject_presence none contradicts actor or motion description')
    return dict(pair, subject_motion_description=motion, background_description=background)


def clean_pair_sentences(pair: dict[str, Any]) -> dict[str, Any]:
    """Remove whole forbidden sentences, never invent or reverse a motion claim.

    Raw model responses remain in append-only logs. An empty/short description
    still fails validation and must be regenerated; there is no generic fallback.
    """
    result = dict(pair)
    flags = list(result.get('quality_flags', []))
    for key in ('subject_motion_description', 'background_description'):
        original = str(result.get(key, '')).strip()
        sentences = re.split(r'(?<=[.!?])\s+', original)
        retained = []
        for sentence in sentences:
            forbidden = camera_text(sentence)
            if key == 'background_description':
                forbidden = forbidden or background_violation(sentence)
            elif ABSENCE_ONLY.search(
                    sentence) or result.get('subject_presence') == 'none' and STATIC_ONLY.search(sentence):
                forbidden = True
            elif (result.get('subject_presence') == 'none' and MOTION.search(sentence)
                  and not foreground_subject_text(sentence)):
                # E.g. "The distant hills remain stationary" or "The road
                # curves forward" is not actor behavior. Remove the whole
                # motion-bearing clause; preserve any actual-actor contradiction
                # for validation/relabeling rather than silently hiding it.
                forbidden = True
            if not forbidden:
                retained.append(sentence)
        cleaned = ' '.join(retained).strip()
        result[key] = cleaned
        if cleaned != original:
            flags.append(f'removed_forbidden_sentences:{key}')
            if key == 'subject_motion_description':
                result['subject_motion_status'] = 'not_observed' if result.get(
                    'subject_presence') == 'none' else 'uncertain'
    if (result.get('subject_presence') == 'none' and result.get('subject_motion_status') == 'not_observed'
            and len(result['subject_motion_description'].split()) < 5):
        # Explicit no-subject VLM output permits the two variants to describe
        # the SAME observed environment. Never use a canned caption or infer
        # absence from the older ambiguous not_observed motion status alone.
        scene = result['background_description']
        if 5 <= len(scene.split()) <= 60 and not camera_text(scene) and not background_violation(scene):
            result['subject_motion_description'] = scene
            result['caption_mode'] = 'environment_only_explicit_no_subject'
            flags.append('paired_vlm_environment_reused_after_explicit_no_subject')
    result['quality_flags'] = flags
    return validate_pair(result)


def metadata(path: Path) -> tuple[Path, list[dict[str, Any]]]:
    columns = [
        'id', 'source_name', 'source_frame_indices', 'num_frames', 'vae_latent_shape', 'first_frame_latent_shape'
    ]
    return path, pq.read_table(path, columns=columns, use_threads=False).to_pylist()


def inventory(args: argparse.Namespace) -> None:
    root = args.root
    root.mkdir(parents=True, exist_ok=True)
    old_mapping = {f"{r['source_name']}/{r['sample_id']}": r for r in read_jsonl(OLD_MAP)}
    original = {}
    heldout_videos = set()
    for split in ('train', 'validation'):
        for r in read_jsonl(OLD_SPLITS / f'{split}_ids.jsonl'):
            original[r['global_id']] = r
            if split == 'validation':
                heldout_videos.add(str(Path(r['video']).resolve()))
    existing = {}
    for index, row in enumerate(read_jsonl(TEXT_ROOT / 'text_data_plan.jsonl')):
        path = TEXT_ROOT / 'preprocessed' / row['split'] / f'worker_{index % 8:02d}' / f'data_{index // 8:06d}.parquet'
        existing[row['id']] = dict(row, existing_encoded=str(path))
        if row['split'] == 'validation':
            heldout_videos.add(str(Path(row['video']).resolve()))
    new_registry = {r['global_id']: r for r in read_jsonl(NEW_REGISTRY)}

    # Legacy dynamic descriptions are reusable ONLY with proven entire-window
    # equality, not merely matching stems, keyboard names or six sample frames.
    legacy_windows = {
        r['global_id']: r
        for r in read_jsonl(BASE / 'h3_all_video_captions_v1/manifests/all_windows.jsonl')
    }
    legacy_captions = {}
    for path in sorted((BASE / 'h3_all_video_captions_v1/qwen3vl_scene_subject_v1').glob('captions_shard_*.jsonl')):
        for caption in read_jsonl(path):
            spec = legacy_windows.get(caption.get('global_id'))
            text = caption.get('scene_description', '')
            if spec and not caption.get('error') and caption.get(
                    'prompt_policy'
            ) == 'observed_scene_subject_motion_no_camera_v1' and text and not CAMERA.search(text):
                legacy_captions[window_key(spec['video'], spec['source_frame_indices'])] = caption

    paths = sorted(OLD_RAW.rglob('*.parquet')) + sorted(NEW_RAW.rglob('*.parquet'))
    rows, errors, seen = [], [], set()
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=12) as pool:
        for count, (path, records) in enumerate(pool.map(metadata, paths), 1):
            for position, record in enumerate(records):
                sid = record['id']
                if sid in seen:
                    raise ValueError(f'Duplicate ID {sid}')
                seen.add(sid)
                if tuple(record['vae_latent_shape']) != (24, 37, 30, 52) or tuple(
                        record['first_frame_latent_shape']) != (24, 1, 30, 52):
                    raise ValueError(f'Wrong latent grid {sid}')
                indices = record['source_frame_indices']
                if len(indices) != 124 or any(a > b for a, b in zip(indices, indices[1:], strict=False)):
                    raise ValueError(f'Wrong source frame plan {sid}')
                is_old = sid in old_mapping
                source = original.get(old_mapping[sid]['global_id']) if is_old else new_registry.get(sid)
                if not source:
                    errors.append(dict(id=sid, reason='missing_original_video_mapping'))
                    continue
                video = str(Path(source['video']).resolve())
                if not Path(video).is_file():
                    errors.append(dict(id=sid, video=video, reason='missing_original_video'))
                    continue
                if not is_old and indices != source['source_frame_indices']:
                    raise ValueError(f'New caption/source frame mismatch {sid}')
                previous = existing.get(sid)
                legacy = legacy_captions.get(window_key(video, indices))
                draft = previous['caption'] if previous else (legacy or {}).get('scene_description', '')
                row = dict(id=sid,
                           global_id=sid,
                           source_name=record['source_name'],
                           batch='old_65389' if is_old else 'new_50599',
                           source_parquet=str(path),
                           source_row=position,
                           video=video,
                           source_frame_indices=indices,
                           window_id=f'{sid}:124f:{digest(json.dumps(indices))[:12]}',
                           split='validation' if video in heldout_videos else 'train',
                           existing_caption=draft,
                           caption_reuse_basis='exact_current_text_binding' if previous else
                           ('exact_legacy_window' if legacy else 'missing'),
                           existing_encoded=previous.get('existing_encoded') if previous else None,
                           original_global_id=source.get('global_id', sid))
                rows.append(row)
            if count % 500 == 0:
                print(f'INVENTORY {count}/{len(paths)} rows={len(rows)} elapsed={time.monotonic()-started:.1f}s',
                      flush=True)
    rows.sort(key=lambda r: (r['source_parquet'], r['source_row']))
    write_jsonl(root / 'registry.jsonl', rows)
    write_jsonl(root / 'inventory_errors.jsonl', errors)
    summary = dict(expected=115988,
                   seen=len(seen),
                   usable=len(rows),
                   errors=len(errors),
                   batch_counts=dict(Counter(r['batch'] for r in rows)),
                   source_counts=dict(Counter(r['source_name'] for r in rows)),
                   split_counts=dict(Counter(r['split'] for r in rows)),
                   caption_reuse_counts=dict(Counter(r['caption_reuse_basis'] for r in rows)),
                   sources_read_only=True,
                   vae_reencoding=False,
                   policy=POLICY)
    atomic_json(root / 'inventory_summary.json', summary)
    print(json.dumps(summary), flush=True)
    if len(seen) != 115988 or errors or len(rows) != 115988:
        raise ValueError('Incomplete union inventory; inspect inventory_errors.jsonl before captioning')


def check_caption_provenance(row: dict[str, Any], spec: dict[str, Any]) -> None:
    sampled = [spec['source_frame_indices'][int(i)] for i in np.rint(np.linspace(0, 123, 6))]
    if (row.get('policy') != POLICY or row.get('window_id') != spec['window_id'] or row.get('video') != spec['video']
            or row.get('source_frame_indices') != spec['source_frame_indices']
            or row.get('sampled_source_frame_indices') != sampled or len(row.get('sampled_rgb_sha256', [])) != 6):
        raise ValueError('caption video/window/frame provenance mismatch')


def collect_captions(root: Path, specs: list[dict[str, Any]]) -> tuple[dict, dict, dict]:
    """One resolver for planning, worker resume, and finalization.

    Re-read append-only evidence rather than trusting a stale missing-ID list.
    A valid raw pair wins over sentence repairs; later failures never erase it.
    """
    by_id = {s['id']: s for s in specs}
    if len(by_id) != len(specs):
        raise ValueError('duplicate registry ID')
    valid: dict[str, dict[str, Any]] = {}
    repairs: dict[str, dict[str, Any]] = {}
    failures: dict[str, dict[str, Any]] = {}
    paths = sorted((root / 'captions').glob('worker_*.jsonl'))
    if (root / 'caption_repairs.jsonl').exists():
        paths = [root / 'caption_repairs.jsonl', *paths]
    for path in paths:
        for row in read_jsonl(path, allow_truncated_tail=path.parent.name == 'captions'):
            key = row.get('id')
            if not isinstance(key, str) or key not in by_id:
                continue
            failure = failures.setdefault(key, dict(attempts=0, error='', previous_pair={}))
            failure['attempts'] += 1
            try:
                check_caption_provenance(row, by_id[key])
            except ValueError as error:
                failure['error'] = str(error)
                continue
            if not row.get('error'):
                try:
                    payload = validate_pair(row)
                    if path.name == 'caption_repairs.jsonl':
                        repairs[key] = payload
                    else:
                        valid[key] = payload
                        repairs.pop(key, None)
                    continue
                except ValueError as error:
                    failure['error'] = str(error)
            if key in valid:
                continue
            try:
                raw = row.get('raw_response', '')
                payload = json.loads(raw[raw.find('{'):raw.rfind('}') + 1])
                failure['previous_pair'] = {
                    k: str(payload.get(k, ''))[:600]
                    for k in ('subject_motion_description', 'background_description')
                }
                payload = clean_pair_sentences(payload)
                repaired = dict(row, **payload)
                repaired.update(error=None,
                                reused_subject_caption=False,
                                sanitation='whole forbidden sentences removed; no facts or motion invented',
                                recovery_version=RECOVERY_VERSION)
                repairs[key] = repaired
            except (ValueError, TypeError, KeyError) as error:
                failure['error'] = f'{type(error).__name__}: {error}'
    valid.update({k: v for k, v in repairs.items() if k not in valid})
    return valid, repairs, failures


def plan(args: argparse.Namespace) -> None:
    specs = registry_specs(args.root)
    captions, repairs, failures = collect_captions(args.root, specs)
    pending = [dict(s, caption_feedback=failures.get(s['id'], {})) for s in specs if s['id'] not in captions]
    write_jsonl(args.root / 'caption_repairs.jsonl', [r for k, r in repairs.items() if k in captions])
    write_jsonl(args.root / 'caption_pending.jsonl', pending)
    write_jsonl(args.root / 'caption_missing_ids.jsonl',
                [dict(id=r['id'], reason='missing_valid_dual_caption') for r in pending])
    summary = dict(total=len(specs),
                   valid_pairs=len(captions),
                   pending=len(pending),
                   workers=args.workers,
                   source_counts=dict(Counter(r['source_name'] for r in pending)),
                   error_counts=dict(Counter(r['caption_feedback'].get('error', 'missing') for r in pending)),
                   registry_sha256=file_digest(args.root / 'registry.jsonl'),
                   pending_sha256=file_digest(args.root / 'caption_pending.jsonl'),
                   policy=POLICY,
                   selection_sha256=selection_hash(args.root),
                   recovery_version=RECOVERY_VERSION)
    atomic_json(args.root / 'caption_pending_summary.json', summary)
    print('CAPTION_PLAN', json.dumps(summary), flush=True)


PROMPT = '''Analyze six chronological frames from the SAME exact video window.
Return JSON only, with these keys:
{"subject_motion_description":"...","background_description":"...","subject_presence":"visible|none|uncertain","subject_motion_status":"observed|uncertain|not_observed","existing_caption_supported":false,"confidence":0.0,"quality_flags":[]}
subject_motion_description: 1-3 factual English sentences, maximum 65 words,
describing environment, visible main subjects and their appearance, and ONLY
their clearly observed physical behavior. Global image displacement or an object
getting larger/smaller does NOT establish subject motion: it can be parallax.
Forward/backward is relative to the subject's own heading, NOT screen movement.
Do not guess forward motion. If physical motion or heading is ambiguous, omit
the direction and mark uncertain. Do not invent static/stationary claims.
If NO foreground subject is visible, set subject_presence="none" and motion
status="not_observed". Still provide an informative description of the ACTUAL
environment, materials, architecture, vegetation and lighting in the frames.
Do NOT write "No subjects are visible", leave it empty, invent an actor, or
describe observer movement. In this case both descriptions may describe the
same environment. This is scene-only evidence, not a missing-actor placeholder.
background_description: 1-2 static English sentences, maximum 40 words, ONLY
terrain, roads, buildings, vegetation, sky, weather, lighting, materials, setting.
Exclude ALL foreground vehicles, tanks, people, animals, aircraft, ships, other
main actors and their identities, appearance, behavior, motion or motion traces.
Do not say subjects are absent. Do not mention the removal of any subject.
BOTH versions must exclude camera motion/control, viewpoint, cinematic framing, shot,
perspective, tracking/following, pan/tilt/zoom, camera-static claims and audio.
The optional existing caption below may be reused ONLY if every claim is
supported by these frames, including subject motion. Set existing_caption_supported
to true only then. Do not copy unsupported motion claims. The subject_motion_status
must be observed, uncertain, or not_observed. Confidence must be 0 to 1.
Existing caption: {draft}
'''


def caption_prompt(row: dict[str, Any], feedback: dict[str, Any], attempt: int) -> str:
    draft = row.get('existing_caption', '')
    if camera_text(draft) or not 5 <= len(draft.split()) <= 90 or attempt > 1:
        draft = ''
    prompt = PROMPT.replace('{draft}', draft or '(none; describe only these frames)')
    if feedback.get('error'):
        prompt += (
            '\nRepair focus: ' + str(feedback['error'])[:300] +
            '\nRegenerate independently from the images. Replace forbidden sentences with specific visible scene facts; '
            'do not repeat absence statements or reuse rejected draft motion. '
            'Storage/water/oil tanks are industrial structures, not combat vehicles. '
            'Avoid ambiguous wording such as bare tank, view, or framing.\n')
    if attempt >= 2:
        prompt += ('\nUse concise wording: at most two sentences in the first description and one in the second. '
                   'Name the visible environment directly; never narrate what is revealed or how the view changes.\n')
    if attempt >= 3:
        prompt += (
            '\nFinal rewrite: stop listing repeated objects. Group similar visible objects into one noun phrase. '
            'Prefer literal materials, colors, layout and lighting; do not guess any subject motion.\n')
    if feedback.get('error'):
        prompt += ('\nBackground repair: write one independent sentence naming only the visible ground, '
                   'buildings, vegetation, sky and lighting. Do not include trains, rail cars, boats, '
                   'food trucks, industrial tanks or actors in that sentence. Describe visible actors only '
                   'in subject_motion_description. Use subject_presence="visible" if any actor is visible, '
                   'even if it is parked, distant, or its physical movement is uncertain. '
                   'Do not write a generic replacement or infer scenery that is not in the images.\n')
    return prompt


def generate_batch(model: Any, processor: Any, frame_groups: list[Any], prompts: list[str], device: Any) -> list[str]:
    import torch
    texts, images = [], []
    for frames, prompt in zip(frame_groups, prompts, strict=True):
        messages = [{
            'role': 'user',
            'content': [*[{
                'type': 'image',
                'image': f
            } for f in frames], {
                'type': 'text',
                'text': prompt
            }]
        }]
        texts.append(processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        images.extend(frames)
    inputs = processor(text=texts, images=images, padding=True, return_tensors='pt').to(device)
    with torch.inference_mode():
        outputs = model.generate(**inputs,
                                 max_new_tokens=448,
                                 do_sample=False,
                                 eos_token_id=processor.tokenizer.eos_token_id,
                                 pad_token_id=processor.tokenizer.eos_token_id)
    return processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)


def caption(args: argparse.Namespace) -> None:
    planned = args.root / 'caption_pending.jsonl'
    if planned.exists():
        summary = json.loads((args.root / 'caption_pending_summary.json').read_text())
        if (summary['registry_sha256'] != file_digest(args.root / 'registry.jsonl')
                or summary['pending_sha256'] != file_digest(planned)
                or summary.get('selection_sha256') != selection_hash(args.root)
                or summary.get('recovery_version') != RECOVERY_VERSION):
            raise ValueError('Stale pending-caption plan; run plan before starting workers')
        rows = list(read_jsonl(planned))[args.worker::args.workers]
    else:
        rows = registry_specs(args.root)[args.worker::args.workers]
    done, _, failures = collect_captions(args.root, rows)
    rows = [r for r in rows if r['id'] not in done]
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        print(f'CAPTION_NOTHING_PENDING worker={args.worker}', flush=True)
        return
    # Import/load CUDA components only after the exact-window resume check.
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    sys.path.insert(0, '/home/lizihao/projects/worldmodel/H3-World/code/vwm')
    from caption_scenes_qwen3vl import load_frames
    torch.set_num_threads(2)
    tag = f'recovery_{time.time_ns()}'
    output = args.root / 'captions' / f'worker_{args.worker:02d}_{tag}.jsonl'
    output.parent.mkdir(parents=True, exist_ok=True)
    model_path = '/datacc05/lizihao/models/worldarena/qwenvl3'
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    processor.tokenizer.padding_side = 'left'
    model = Qwen3VLForConditionalGeneration.from_pretrained(model_path,
                                                            torch_dtype=torch.bfloat16,
                                                            attn_implementation='sdpa',
                                                            local_files_only=True).to('cuda:0').eval()
    print(f'VLM_MODEL_READY worker={args.worker} pending={len(rows)} batch={args.batch_size}', flush=True)
    device = torch.device('cuda:0')
    started, finished, valid = time.monotonic(), 0, 0

    def load(row):
        indices = [row['source_frame_indices'][int(i)] for i in np.rint(np.linspace(0, 123, 6))]
        return row, indices, load_frames(Path(row['video']), indices, 384)

    with output.open('a') as handle, ThreadPoolExecutor(max_workers=2) as decode_pool:
        for start in range(0, len(rows), args.batch_size):
            group, loaded = rows[start:start + args.batch_size], []
            for row, future in [(r, decode_pool.submit(load, r)) for r in group]:
                try:
                    loaded.append(future.result())
                except Exception as error:
                    handle.write(json.dumps(dict(id=row['id'], policy=POLICY, error=f'decode: {error}')) + '\n')
            if not loaded:
                continue
            prompts = [
                caption_prompt(r, failures.get(r['id'], r.get('caption_feedback', {})), args.attempt)
                for r, _, _ in loaded
            ]
            try:
                outputs = generate_batch(model, processor, [f for _, _, f in loaded], prompts, device)
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                outputs = [
                    generate_batch(model, processor, [f], [p], device)[0]
                    for (_, _, f), p in zip(loaded, prompts, strict=False)
                ]
            for (row, indices, frames), raw in zip(loaded, outputs, strict=True):
                result = dict(id=row['id'],
                              window_id=row['window_id'],
                              video=row['video'],
                              source_frame_indices=row['source_frame_indices'],
                              sampled_source_frame_indices=indices,
                              sampled_rgb_sha256=[hashlib.sha256(f.tobytes()).hexdigest() for f in frames],
                              policy=POLICY,
                              recovery_version=RECOVERY_VERSION,
                              attempt=args.attempt,
                              raw_response=raw,
                              model=model_path)
                try:
                    payload = json.loads(raw[raw.find('{'):raw.rfind('}') + 1])
                    payload = clean_pair_sentences(payload)
                    reused = False
                    if row.get('existing_caption') and payload.get('existing_caption_supported') is True:
                        try:
                            candidate = validate_pair(dict(payload, subject_motion_description=row['existing_caption']))
                        except ValueError:
                            # Keep the valid image-grounded generated result;
                            # never replace it with a forbidden old draft.
                            payload['quality_flags'].append('rejected_existing_caption_by_policy')
                        else:
                            payload, reused = candidate, True
                    result.update(payload, reused_subject_caption=reused, error=None)
                    valid += 1
                except Exception as error:
                    result['error'] = f'{type(error).__name__}: {error}'
                handle.write(json.dumps(result, ensure_ascii=False) + '\n')
                finished += 1
            handle.flush()
            os.fsync(handle.fileno())
            elapsed = time.monotonic() - started
            progress = dict(worker=args.worker,
                            completed_this_run=finished,
                            valid_this_run=valid,
                            remaining_this_run=len(rows) - min(start + args.batch_size, len(rows)),
                            elapsed_seconds=elapsed,
                            videos_per_second=finished / max(elapsed, .001),
                            batch_size=args.batch_size)
            atomic_json(args.root / 'progress' / f'worker_{args.worker:02d}.json', progress)
            print('CAPTION_PROGRESS', json.dumps(progress), flush=True)


def finalize(args: argparse.Namespace) -> None:
    specs = registry_specs(args.root)
    captions, repairs, _ = collect_captions(args.root, specs)
    write_jsonl(args.root / 'caption_repairs.jsonl', list(repairs.values()))
    missing = [dict(id=r['id'], reason='missing_valid_dual_caption') for r in specs if r['id'] not in captions]
    write_jsonl(args.root / 'caption_missing_ids.jsonl', missing)
    if missing:
        marker = args.root / 'CAPTIONS_VERIFIED'
        if marker.exists():
            marker.rename(args.root / f'CAPTIONS_VERIFIED.invalidated.{time.time_ns()}')
        raise ValueError(f'{len(missing)} captions missing/invalid; rerun resumable workers; no training allowed')
    motion_plan: list[dict[str, Any]] = []
    background_plan: list[dict[str, Any]] = []
    registry_hash = file_digest(args.root / 'registry.jsonl')
    for spec in specs:
        cap = captions[spec['id']]
        if cap['window_id'] != spec['window_id'] or cap['source_frame_indices'] != spec['source_frame_indices'] or cap[
                'video'] != spec['video']:
            raise ValueError(f'Caption/video/window mismatch {spec["id"]}')
        sampled = [spec['source_frame_indices'][int(i)] for i in np.rint(np.linspace(0, 123, 6))]
        if cap.get('sampled_source_frame_indices') != sampled or len(cap.get('sampled_rgb_sha256', [])) != 6:
            raise ValueError(f'Caption sampled frame provenance mismatch {spec["id"]}')
        for key, target in (('subject_motion_description', motion_plan), ('background_description', background_plan)):
            text = cap[key]
            target.append(
                dict(spec,
                     caption=text,
                     caption_sha256=digest(text),
                     caption_window=spec['window_id'],
                     reused_subject_caption=cap['reused_subject_caption'],
                     caption_policy=POLICY,
                     registry_sha256=registry_hash,
                     subject_motion_status=cap['subject_motion_status'],
                     caption_confidence=cap['confidence'],
                     caption_quality_flags=cap.get('quality_flags', [])))
            target[-1]['caption_subject_presence'] = cap.get('subject_presence', 'not_recorded')
    plan_hashes = {}
    for variant, plan in (('subject_motion', motion_plan), ('background_only', background_plan)):
        write_jsonl(args.root / variant / 'text_data_plan.jsonl', plan)
        plan_hashes[variant] = file_digest(args.root / variant / 'text_data_plan.jsonl')
        atomic_json(
            args.root / variant / 'data_plan_summary.json',
            dict(total=len(plan),
                 split_counts=dict(Counter(r['split'] for r in plan)),
                 source_counts=dict(Counter(r['source_name'] for r in plan)),
                 variant=variant,
                 same_video_windows=True,
                 vae_reencoding=False,
                 policy=POLICY,
                 selection_sha256=selection_hash(args.root),
                 registry_sha256=registry_hash,
                 plan_sha256=plan_hashes[variant]))
    atomic_json(
        args.root / 'CAPTIONS_VERIFIED',
        dict(rows=len(specs),
             policy=POLICY,
             registry_sha256=registry_hash,
             selection_sha256=selection_hash(args.root),
             plan_sha256=plan_hashes,
             repaired_caption_pairs=len(repairs),
             recovery_version=RECOVERY_VERSION,
             reused_subject_captions=sum(c['reused_subject_caption'] for c in captions.values()),
             background_checks='no camera/audio/foreground-subject/motion keywords; automated, not human approval'))
    print(f'CAPTIONS_VERIFIED {len(specs)} pairs', flush=True)


def verify_captions(args: argparse.Namespace) -> None:
    """Check a frozen caption gate without rescanning all raw VLM retries."""
    root = args.root
    specs = registry_specs(root)
    marker = json.loads((root / 'CAPTIONS_VERIFIED').read_text())
    registry_sha256 = file_digest(root / 'registry.jsonl')
    if (marker.get('rows') != len(specs) or marker.get('registry_sha256') != registry_sha256
            or marker.get('selection_sha256') != selection_hash(root) or marker.get('policy') != POLICY
            or marker.get('recovery_version') != RECOVERY_VERSION):
        raise ValueError('Stale caption verification gate')
    for variant in ('subject_motion', 'background_only'):
        plan_sha256 = file_digest(root / variant / 'text_data_plan.jsonl')
        summary = json.loads((root / variant / 'data_plan_summary.json').read_text())
        if (marker.get('plan_sha256', {}).get(variant) != plan_sha256 or summary.get('plan_sha256') != plan_sha256
                or summary.get('total') != len(specs) or summary.get('registry_sha256') != registry_sha256
                or summary.get('selection_sha256') != selection_hash(root)):
            raise ValueError('Frozen caption plan/selection changed after verification')
    print(f'FROZEN_CAPTION_GATE_OK rows={len(specs)}', flush=True)


def encode(args: argparse.Namespace) -> None:
    # Full schema encoder/audit already exercised by the preceding real-text run.
    # Reuse unchanged new-row Parquet+audit pairs without duplicating VAE bytes.
    from tools.prepare_text_training_20261022 import encode as encode_existing
    variant_root = args.root / 'subject_motion'
    rows = list(read_jsonl(variant_root / 'text_data_plan.jsonl'))[args.worker::args.workers]
    for position, row in enumerate(rows):
        old_path = row.get('existing_encoded')
        if not old_path or row['caption'] != row['existing_caption']:
            continue
        source = Path(old_path)
        audit = source.with_suffix('.audit.json')
        if not source.is_file() or not audit.is_file():
            raise ValueError(f'Missing existing verified encoding {source}')
        old_audit = json.loads(audit.read_text())
        if (old_audit['id'] != row['id'] or old_audit['caption_sha256'] != row['caption_sha256']
                or Path(old_audit['source_parquet']).resolve() != Path(row['source_parquet']).resolve()
                or old_audit['source_row'] != row['source_row']):
            raise ValueError(f'Reused encoding source provenance mismatch {row["id"]}')
        previous = pq.read_table(source, columns=['id', 'caption', 'source_frame_indices'],
                                 use_threads=False).to_pylist()
        if (len(previous) != 1 or previous[0]['id'] != row['id'] or previous[0]['caption'] != row['caption']
                or previous[0]['source_frame_indices'] != row['source_frame_indices']):
            raise ValueError(f'Reused encoding entire-window mismatch {row["id"]}')
        destination = variant_root / 'preprocessed' / row[
            'split'] / f'worker_{args.worker:02d}' / f'data_{position:06d}.parquet'
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            os.link(source.resolve(), destination)
        elif not destination.samefile(source):
            raise ValueError(f'Unexpected existing output {destination}')
        # A new derived sidecar owns this registry's window identity; never
        # overwrite the previous recipe's sidecar through a symlink.
        atomic_json(
            destination.with_suffix('.audit.json'),
            dict(old_audit,
                 caption_window=row['caption_window'],
                 video=row['video'],
                 source_frame_indices=row['source_frame_indices'],
                 registry_sha256=row['registry_sha256'],
                 reused_from=str(source)))
    encode_existing(argparse.Namespace(root=variant_root, worker=args.worker, workers=args.workers))
    for position, row in enumerate(rows):
        audit_path = variant_root / 'preprocessed' / row[
            'split'] / f'worker_{args.worker:02d}' / f'data_{position:06d}.audit.json'
        metadata_value = json.loads(audit_path.read_text())
        atomic_json(
            audit_path,
            dict(metadata_value,
                 caption_window=row['caption_window'],
                 video=row['video'],
                 source_frame_indices=row['source_frame_indices'],
                 registry_sha256=row['registry_sha256'],
                 **({
                     key: row[key]
                     for key in ('prompt_composition', 'component_background_sha256', 'component_subject_sha256')
                 } if row.get('prompt_composition') else {})))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'command',
        choices=['inventory', 'plan', 'caption', 'finalize', 'verify_captions', 'encode', 'audit', 'exclude_failed'])
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--worker', type=int, default=0)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--attempt', type=int, default=1)
    parser.add_argument('--authorize-exclude-failed', action='store_true')
    parser.add_argument('--exclude-ids-path', type=Path)
    args = parser.parse_args()
    if args.workers < 1 or not 0 <= args.worker < args.workers or args.batch_size < 1:
        parser.error('invalid worker or batch size')
    if args.command == 'audit':
        from tools.audit_scene_text_20261022 import audit_root
        audit_root(args.root / 'subject_motion', workers=12)
    else:
        globals()[args.command](args)


if __name__ == '__main__':
    main()
