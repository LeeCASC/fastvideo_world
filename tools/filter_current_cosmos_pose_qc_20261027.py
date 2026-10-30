"""CPU-only Cosmos-reference motion QC of frozen native H3 training windows.

Read source dense poses, REAL 37 H3 anchors and video latent, without changing
source data or live training. Reference functions are imported, not reimplemented.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any
from collections.abc import Iterator

import numpy as np
import pyarrow.parquet as pq

REFERENCE = Path('/datacc05/lizihao/worldmodel/cosmos_pose_qc_reference')
DEFAULT_INPUT = Path('/datacc05/lizihao/worldmodel/fastvideowm_union_scene_subject_poseqc_20261027/subject_motion')
DEFAULT_OUTPUT = Path('/datacc05/lizihao/worldmodel/fastvideowm_cosmos_pose_qc_20261027')
OLD_MANIFEST = Path('/datacc05/lizihao/worldmodel/fastvideowm_action_only_v1_20260925')
NEW_MANIFEST = Path(
    '/datacc05/lizihao/worldmodel/vwm_unified_124_243_20261016/124f/vlm_qwen3vl_strict_50599/registry_strict_50599.jsonl'
)
VERSION = 'native_h3_cosmos_reference_v1_20261027'
REF: Any = None
INPUT_ROOT = DEFAULT_INPUT


def load_reference() -> Any:
    """Standalone CPU process only; no native FastVideo/CUDA model imports."""
    global REF
    if REF is None:
        sys.path.insert(0, str(REFERENCE))
        # When called via -m tools..., retain native tools AND reference tools.
        if 'tools' in sys.modules and hasattr(sys.modules['tools'], '__path__'):
            sys.modules['tools'].__path__.append(str(REFERENCE / 'tools'))
        from tools.dataset import filter_cosmos_dataset_pose_qc
        REF = filter_cosmos_dataset_pose_qc
    return REF


def sha(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def json_rows(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def dump(path: str | Path, obj: Any) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def h3_offsets(frames: int = 124) -> np.ndarray:
    if frames != 124:
        raise ValueError('Only the current frozen 124f native H3 dataset is supported')
    return np.cumsum(np.resize(np.array([1, 4, 4, 4, 4]), 37)) - 1


def relative(pose: np.ndarray) -> np.ndarray:
    return np.linalg.inv(pose[0]) @ pose


def tensor(row: dict, key: str) -> np.ndarray:
    return np.frombuffer(row[key + '_bytes'], dtype=np.dtype(row[key + '_dtype'])).reshape(row[key + '_shape'])


def dataset_name(source: str) -> str:
    return source[len('train_v2_'):] if source.startswith('train_v2_') else source


def dense_camera(descriptor: dict, indices: np.ndarray) -> np.ndarray | None:
    """Reproduce the actual historical producer; NEVER fit scale to anchors."""
    path = descriptor.get('pose_path')
    if not path:
        return None
    with np.load(path, allow_pickle=False) as z:
        if 'camera_to_world' in z:
            pose = np.asarray(z['camera_to_world'], dtype=np.float64)
        elif 'c2w' in z:
            pose = np.asarray(z['c2w'], dtype=np.float64)
        elif 'extrinsics' in z:
            e = np.asarray(z['extrinsics'], dtype=np.float64)
            if e.ndim != 3 or e.shape[1:] != (3, 4):
                raise ValueError(f'Invalid W2C extrinsics shape: {e.shape}')
            pose = np.broadcast_to(np.eye(4), (len(e), 4, 4)).copy()
            pose[:, :3, :3] = e[:, :3, :3].transpose(0, 2, 1)
            pose[:, :3, 3] = -np.einsum('tij,tj->ti', pose[:, :3, :3], e[:, :3, 3])
        else:
            raise ValueError('Declared source pose has no supported C2W/W2C field')
    if pose.ndim != 3 or pose.shape[1:] != (4, 4) or not np.isfinite(pose).all():
        raise ValueError('Invalid/nonfinite dense source camera')
    if indices.min() < 0 or indices.max() >= len(pose):
        raise ValueError(f'Source indices exceed dense pose length {len(pose)}')
    if descriptor['dense_kind'] == 'old_calibrated':
        pose = relative(pose)
        if descriptor['source_family'] in {'h3_js_easy', 'js_refine_data'}:
            basis = np.diag([1., -1., 1., 1.])
            pose = basis @ pose @ basis
        maximum = np.linalg.norm(pose[:, :3, 3], axis=1).max(initial=0.)
        if maximum > 1e-8:
            pose[:, :3, 3] /= maximum
        factors = np.asarray(descriptor['translation_scale_xyz'], dtype=np.float64)
        if factors.shape != (3, ) or not np.isfinite(factors).all() or (factors <= 0).any() or (factors > 4).any():
            raise ValueError('Invalid historical translation calibration')
        pose[:, :3, 3] *= factors
        # The historical producer persisted calibrated geometry in float32.
        pose = pose.astype(np.float32).astype(np.float64)
    elif descriptor['dense_kind'] != 'new_direct':
        raise ValueError('Unknown dense provenance')
    return relative(pose[indices])


def actions(pose: np.ndarray, indices: np.ndarray, source: str) -> dict:
    ref = load_reference()
    cfg = ref.get_action_thresholds(dataset_name(source))
    _, mouse, discrete = ref.infer_canonical_actions_from_view_se3(pose.astype(np.float32),
                                                                   frame_indices=indices,
                                                                   **cfg)
    prompts = [
        ref.prompt_from_canonical_actions(discrete['move'][i], discrete['yaw'][i], discrete['pitch'][i],
                                          discrete['roll'][i]) for i in range(len(pose))
    ]
    threshold = cfg['rot_action_thr'] * max(1, int(np.diff(indices).max()))
    reversals = []
    for axis in range(3):
        active = mouse[1:, axis]
        active = active[np.abs(active) > threshold]
        reversals.append(int((active[1:] * active[:-1] < 0).sum()))
    return dict(distinct_action_states=len(set(prompts[1:])),
                stable_action_changes=ref.count_stable_action_changes(prompts),
                rotation_reversals=reversals,
                thresholds=cfg)


def inspect(descriptor: dict) -> dict:
    ref = load_reference()
    result = {k: descriptor[k] for k in ('id', 'source_name', 'split', 'conditioned_relative_path')}
    result.update(reasons=[], warnings=[], dense_pose_available=False)
    path = INPUT_ROOT / descriptor['conditioned_relative_path']
    try:
        before = path.stat()
        columns = ['id', 'source_name', 'caption', 'num_frames', 'fps', 'source_fps', 'source_frame_indices']
        for name in ('vae_latent', 'camera_to_world'):
            columns += [name + suffix for suffix in ('_bytes', '_shape', '_dtype')]
        table = pq.ParquetFile(path).read(columns=columns, use_threads=False)
        if table.num_rows != 1:
            raise ValueError('Expected exactly one conditioned row')
        x = table.to_pylist()[0]
        if x['id'] != descriptor['id'] or x['source_name'] != descriptor['source_name']:
            raise ValueError('Conditioned ID/source mismatch')
        if hashlib.sha256(x['caption'].encode()).hexdigest() != descriptor['caption_sha256']:
            raise ValueError('Caption differs from frozen plan')
        indices = np.asarray(x['source_frame_indices'], dtype=np.int64)
        if x['num_frames'] != 124 or indices.shape != (124, ) or (np.diff(indices) < 0).any():
            raise ValueError('Invalid native video window / non-monotonic indices')
        if not np.array_equal(indices, descriptor['source_frame_indices']):
            raise ValueError('Source window differs from frozen plan')
        pose = tensor(x, 'camera_to_world').astype(np.float64)
        if pose.shape != (37, 4, 4):
            raise ValueError('Expected actual 37 native H3 camera anchors')
        result['reasons'].extend(ref.matrix_errors(pose))
        if result['reasons']:
            return result
        result['first_camera_identity_error'] = float(np.abs(pose[0] - np.eye(4)).max())
        if result['first_camera_identity_error'] > .005:
            result['reasons'].append('first_camera_not_identity')
        dense = dense_camera(descriptor, indices)
        result['dense_pose_available'] = dense is not None
        result['dense_pose_path'] = descriptor.get('pose_path')
        result['source_fps'] = x['source_fps']
        result['unique_source_frames'] = len(np.unique(indices))
        offsets = h3_offsets()
        if dense is not None:
            result['reasons'].extend(ref.matrix_errors(dense))
            if result['reasons']:
                return result
            error = float(np.abs(relative(dense)[offsets] - relative(pose)).max())
            result['dense_cache_max_abs_error'] = error
            if error > 1e-4:
                result['reasons'].append('dense_training_pose_mismatch')
                return result
            # Repeated source frames from legitimate FPS conversion are not
            # additional observations. Do NOT interpolate 37 anchors to 124.
            unique = np.r_[True, np.diff(indices) > 0]
            observed = dense[unique]
            observed_indices = indices[unique]
            if len(observed) < 2:
                raise ValueError('Fewer than two distinct source frames')
            result['dense_motion'] = ref.motion_stats(observed)
            if ref.dense_jitter_rejected(result['dense_motion']):
                result['reasons'].append('dense_translation_jitter')
            action = actions(observed, observed_indices, descriptor['source_name'])
            action['origin'] = 'recomputed_observed_dense_pose'
        else:
            result['warnings'].append('dense_pose_unavailable_training_anchors_checked')
            action = actions(pose, indices[offsets], descriptor['source_name'])
            action['origin'] = 'recomputed_actual_h3_anchors'
        metrics = ref.motion_stats(pose)
        result['training_motion'] = metrics
        result['actions'] = action
        length_scale = 123 / 60
        rules = ref.RULES
        if metrics['max_rotation_step_deg'] > rules['max_rotation_step_deg']:
            result['reasons'].append('abnormal_rotation_jump')
        if metrics['displacement_jumps']:
            result['reasons'].append('abnormal_translation_jump')
        if metrics['translation_reversals'] > math.floor(rules['max_translation_reversals'] * length_scale):
            result['reasons'].append('translation_jitter')
        if max(action['rotation_reversals']) > math.floor(rules['max_rotation_reversals_per_axis'] * length_scale):
            result['reasons'].append('rotation_jitter')
        if action['distinct_action_states'] > math.floor(rules['max_distinct_action_states'] * length_scale):
            result['reasons'].append('too_many_action_states')
        if action['stable_action_changes'] > math.floor(rules['max_action_changes'] * length_scale):
            result['reasons'].append('too_many_action_changes')
        latent = tensor(x, 'vae_latent')
        if latent.shape != (24, 37, 30, 52) or not np.isfinite(latent).all():
            raise ValueError('Invalid native H3 latent shape/finite values')
        small = latent[:, :, ::4, ::4].astype(np.float32)
        transitions = np.sqrt(np.mean(np.diff(small, axis=1)**2, axis=(0, 2, 3)))
        ratio = float(transitions.max() / max(1e-8, np.median(transitions)))
        result['latent_transition_ratio'] = ratio
        if ratio > rules['max_latent_transition_vs_median']:
            result['reasons'].append('abnormal_latent_transition')
        sidecar = json.loads(path.with_suffix('.audit.json').read_text())
        if hashlib.sha256(x['vae_latent_bytes']).hexdigest() != sidecar['target_latent_sha256']:
            raise ValueError('Persisted video latent hash differs from parent audit')
        if hashlib.sha256(x['camera_to_world_bytes']).hexdigest() != sidecar['camera_sha256']:
            raise ValueError('Persisted camera hash differs from parent audit')
        after = path.stat()
        result.update(file_size=after.st_size, mtime_ns=after.st_mtime_ns)
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            result['reasons'].append('changed_during_inspection')
    except Exception as exc:
        result['reasons'].append('inspection_error')
        result['error'] = repr(exc)
    return result


def initialize(root: str | Path) -> None:
    global INPUT_ROOT
    INPUT_ROOT = Path(root)
    import torch
    torch.set_num_threads(1)
    load_reference()


def descriptors(rows: list[dict]) -> list[dict]:
    wanted = {r['original_global_id'] for r in rows}
    old = {}
    for name in ('train_ids.jsonl', 'validation_ids.jsonl'):
        for row in json_rows(OLD_MANIFEST / name):
            if row['global_id'] in wanted:
                old[row['global_id']] = row
    new = {r['global_id']: r for r in json_rows(NEW_MANIFEST) if r['global_id'] in wanted}
    result = []
    paths = set()
    for r in rows:
        d = {
            k: r[k]
            for k in ('id', 'source_name', 'split', 'caption_sha256', 'source_frame_indices',
                      'conditioned_relative_path')
        }
        rel = Path(d['conditioned_relative_path'])
        if rel.is_absolute() or '..' in rel.parts or rel.parts[:2] != ('preprocessed', r['split']):
            raise ValueError('Unsafe or wrong-split conditioned path')
        if str(rel) in paths:
            raise ValueError('Duplicate conditioned path')
        paths.add(str(rel))
        key = r['original_global_id']
        source = old.get(key) if r['batch'] == 'old_65389' else new.get(key)
        if source:
            if str(source['video']) != r['video'] and not Path(source['video']).samefile(r['video']):
                raise ValueError(f'Dense source video provenance mismatch: {key}')
            if r['batch'] == 'old_65389':
                d.update(pose_path=source['pose_path'],
                         dense_kind='old_calibrated',
                         source_family=source['source_family'],
                         translation_scale_xyz=source['translation_scale_xyz'])
            else:
                if source['source_frame_indices'] != r['source_frame_indices']:
                    raise ValueError(f'Dense source window provenance mismatch: {key}')
                d.update(pose_path=source['camera'], dense_kind='new_direct')
        result.append(d)
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate frozen IDs')
    return result


def finalize(output: Path, rows: list[dict], results: list[dict], spec: dict) -> None:
    by_id = {r['id']: r for r in results}
    if len(by_id) != len(rows) or set(by_id) != {r['id'] for r in rows}:
        raise ValueError('Incomplete / duplicate audit coverage')
    counts: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    reasons: collections.Counter = collections.Counter()
    for row in rows:
        audit = by_id[row['id']]
        c = counts[row['source_name']]
        c['before'] += 1
        status = 'rejected' if audit['reasons'] else 'accepted'
        c[status] += 1
        c[f'{status}_{row["split"]}'] += 1
        c['dense_available'] += int(audit['dense_pose_available'])
        reasons.update(set(audit['reasons']))
    with (output / 'accepted.jsonl').open('w') as accepted, (output / 'rejected.jsonl').open('w') as rejected:
        for row in rows:
            audit = by_id[row['id']]
            record = {
                **row, 'conditioned_data_root': str(INPUT_ROOT),
                'conditioned_parquet': str(INPUT_ROOT / row['conditioned_relative_path']),
                'numeric_pose_qc': {
                    'audit': str(output / 'audit.jsonl'),
                    'version': VERSION,
                    'reasons': audit['reasons']
                }
            }
            target = rejected if audit['reasons'] else accepted
            target.write(json.dumps(record, ensure_ascii=False) + '\n')
    summary = {
        'input_rows':
        len(rows),
        'accepted':
        sum(c['accepted'] for c in counts.values()),
        'rejected':
        sum(c['rejected'] for c in counts.values()),
        'by_source':
        dict(sorted(counts.items())),
        'reasons_overlapping':
        dict(reasons.most_common()),
        'dense_pose_available':
        sum(r['dense_pose_available'] for r in results),
        'inspection_errors':
        sum('inspection_error' in r['reasons'] for r in results),
        'accepted_manifest_sha256':
        sha(output / 'accepted.jsonl'),
        'input_spec':
        spec,
        'training_switched':
        False,
        'visual_pose_agreement_verified':
        False,
        'limitations': [
            'Numeric motion QC does not establish visual GT/pose agreement.',
            'H3 has 37 nonuniform anchors, not Cosmos/Wan 31 uniform anchors.',
            'Repeated source indices from valid FPS resampling are deduplicated for dense action statistics.',
            'Actual H3 latent transitions use the reference ratio threshold; VAE families differ.',
            'Anchor-only coverage is reported, never filled using interpolation.',
            'No live trainer, input manifest, source pose/video, caption or latent was changed.'
        ]
    }
    dump(output / 'summary.json', summary)
    lines = [
        '# 当前 FastVideoWM 训练集：Cosmos 参考规则复筛', '',
        f'输入 {len(rows):,} 条，数值规则通过 {summary["accepted"]:,} 条，剔除 {summary["rejected"]:,} 条。', '',
        '数量单位为当前训练视频窗口，不是动作次数。包含训练和验证，原 split 不变。', '', '|来源|筛选前|保留训练|保留验证|保留合计|剔除|', '|---|---:|---:|---:|---:|---:|'
    ]
    for source, c in sorted(counts.items()):
        lines.append(
            f'|{source}|{c["before"]:,}|{c["accepted_train"]:,}|{c["accepted_validation"]:,}|{c["accepted"]:,}|{c["rejected"]:,}|'
        )
    lines += [
        '', '## 规则与适配', '', '直接调用参考包 matrix_errors、motion_stats、dense_jitter_rejected 及相机动作推导函数。',
        '124帧次数阈值按123/60缩放：锚点平移反向≤6次、各旋转轴反向≤14次、动作状态≤16种、稳定动作切换≤24次；',
        '旋转单步≤45°；位移跳变同时超过其他步均值5倍及有效步长中位数3倍；latent变化比≤4。', 'dense抖动拒绝条件为：反向≥4次、反向率>25%、路程/净位移>3，同时成立。',
        'dense使用真实源相机文件，核对实际37个H3锚点（最大误差≤1e-4），不插值伪造逐帧姿态。', '复现旧生产器的坐标系/尺度变换仅用于核对，未平滑、重标或改写训练姿态；新数据按原生产器直接读取。',
        '合法FPS转换带来的重复源帧，在dense统计中去重；H3的非均匀时间锚点保持原样。', '',
        f'完整源轨迹覆盖 {summary["dense_pose_available"]:,}/{len(rows):,}；处理错误 {summary["inspection_errors"]:,}。',
        '注意：这是数值质量过滤，不等于画面与相机几何真值已逐条核实。通过项复用已验收的文本、首帧和latent。', '', '## 拒绝原因（可重叠）', '', '|原因|数量|', '|---|---:|'
    ]
    lines += [f'|{k}|{v:,}|' for k, v in reasons.most_common()]
    lines += [
        '', '## 文件与训练状态', '', f'- 保留完整数据清单：`{output / "accepted.jsonl"}`', f'- 剔除ID和完整溯源：`{output / "rejected.jsonl"}`',
        f'- 逐条数值：`{output / "audit.jsonl"}`', f'- 逐来源统计：`{output / "summary.json"}`',
        '- 未修改当前训练配置；本次清单需要显式绑定到后续训练，不能仅沿用旧data_path。', '- 未删除任何原始视频、pose、VAE、条件缓存或checkpoint。'
    ]
    (output / '过滤结果_zh.md').write_text('\n'.join(lines) + '\n')
    dump(
        output / 'NUMERIC_POSE_QC_COMPLETE', {
            'input_plan_sha256': spec['input_plan_sha256'],
            'audit_sha256': sha(output / 'audit.jsonl'),
            'accepted_manifest_sha256': summary['accepted_manifest_sha256']
        })
    print(json.dumps({k: v for k, v in summary.items() if k != 'input_spec'}, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--pilot-per-source', type=int, default=0)
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 32:
        raise ValueError('CPU readers must be bounded to 1..32')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Run with CUDA_VISIBLE_DEVICES= (CPU-only)')
    initialize(args.input_root)
    plan = args.input_root / 'text_data_plan.jsonl'
    rows = list(json_rows(plan))
    if args.pilot_per_source:
        selected: collections.Counter = collections.Counter()
        pilot = []
        for row in rows:
            if selected[row['source_name']] < args.pilot_per_source:
                pilot.append(row)
                selected[row['source_name']] += 1
        rows = pilot
    tasks = descriptors(rows)
    spec = {
        'version': VERSION,
        'input_root': str(args.input_root),
        'input_plan_sha256': sha(plan),
        'rows': len(rows),
        'pilot_per_source': args.pilot_per_source,
        'reference_hashes': {
            str(p.relative_to(REFERENCE)): sha(p)
            for p in sorted(REFERENCE.rglob('*.py'))
        },
        'adapter_sha256': sha(__file__),
        'rules': load_reference().RULES,
        'dense_rules': load_reference().DENSE_JITTER_RULE,
        'h3_offsets': h3_offsets().tolist()
    }
    args.output.mkdir(parents=True, exist_ok=True)
    spec_path = args.output / 'run_spec.json'
    if spec_path.exists():
        if json.loads(spec_path.read_text()) != spec:
            raise ValueError('Refusing to resume a changed QC plan/rule/adapter')
    else:
        dump(spec_path, spec)
    audit_path = args.output / 'audit.jsonl'
    existing = list(json_rows(audit_path)) if audit_path.exists() else []
    done = {r['id'] for r in existing}
    if len(done) != len(existing) or not done.issubset({r['id'] for r in rows}):
        raise ValueError('Duplicate/unexpected existing audit IDs')
    pending = [r for r in tasks if r['id'] not in done]
    started = time.monotonic()
    initial_completed = len(existing)
    results = existing
    with audit_path.open('a') as handle, concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,
                                                                                initializer=initialize,
                                                                                initargs=(args.input_root, )) as pool:
        # Bounded submission; do not allocate 115k futures / large latent batches.
        for start in range(0, len(pending), 512):
            for result in pool.map(inspect, pending[start:start + 512], chunksize=4):
                results.append(result)
                handle.write(json.dumps(result, ensure_ascii=False) + '\n')
            handle.flush()
            elapsed = time.monotonic() - started
            rate = (len(results) - initial_completed) / max(elapsed, 1e-6)
            progress = dict(completed=len(results),
                            total=len(rows),
                            elapsed_seconds=elapsed,
                            rows_per_second=rate,
                            remaining_seconds=(len(rows) - len(results)) / max(rate, 1e-6),
                            rejected=sum(bool(r['reasons']) for r in results),
                            inspection_errors=sum('inspection_error' in r['reasons'] for r in results))
            dump(args.output / 'progress.json', progress)
            print(json.dumps(progress), flush=True)
    # Freeze input hash still agrees after the full scan.
    if sha(plan) != spec['input_plan_sha256']:
        raise RuntimeError('Input plan changed during QC')
    finalize(args.output, rows, results, spec)


if __name__ == '__main__':
    main()
