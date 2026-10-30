"""Two short prompts; reuse yesterday's exact image, camera, seed and weights."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml

PREVIOUS = Path('/datacc05/lizihao/worldmodel/fastvideowm_union_step800_subject_text_on_20261022')
SCENE = 'A tank is on a dirt road in a forest.'


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f'.tmp.{os.getpid()}')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(temporary, path)


def pair_cases(source: list[dict[str, Any]], output: Path) -> list[dict[str, Any]]:
    cases = []
    for original in source:
        if original['sample_id'] != 'js_00001':
            continue
        row = deepcopy(original)
        direction = row['subject_direction']
        if direction not in ('forward', 'backward'):
            raise ValueError('Unsupported subject direction')
        row.update(prompt=f'{SCENE} The tank moves {direction}.',
                   first_frame_description=SCENE,
                   subject_description=f'The tank moves {direction}.',
                   protocol_id='js00001_short_subject_text_v1',
                   condition_parquet=str(output / 'conditions' / row['id'] / 'data.parquet'),
                   description_source='AI-inspected same first frame; deliberately shortened by user request')
        cases.append(row)
    if len(cases) != 2 or {r['subject_direction'] for r in cases} != {'forward', 'backward'}:
        raise ValueError('Expected exactly one matched js_00001 pair')
    for key in ('seed', 'camera_pose', 'action', 'alpha', 'fps', 'frames', 'width', 'height'):
        if cases[0][key] != cases[1][key]:
            raise ValueError(f'Pair differs in non-text condition {key}')
    return sorted(cases, key=lambda row: row['id'])


def setup(output: Path) -> None:
    source = json.loads((PREVIOUS / 'inference_manifest.json').read_text())
    cases = pair_cases(source, output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / 'inference_manifest.json'
    if manifest.exists() and json.loads(manifest.read_text()) != cases:
        raise ValueError('Refusing to overwrite different frozen inputs')
    save(manifest, cases)
    old = json.loads((PREVIOUS / 'checkpoints.json').read_text())
    checkpoint = Path(old['frozen_checkpoint']).resolve(strict=True)
    if not (checkpoint / 'dcp/.metadata').is_file():
        raise ValueError('Incomplete checkpoint-800 snapshot')
    view = output / 'checkpoint' / checkpoint.name
    view.parent.mkdir(parents=True, exist_ok=True)
    if not view.exists():
        view.symlink_to(checkpoint, target_is_directory=True)
    if view.resolve() != checkpoint:
        raise ValueError('Model view changed')
    source_input = (PREVIOUS / 'inputs/js_00001').resolve(strict=True)
    if not (source_input / 'COMPLETE').exists():
        raise ValueError('Native first-frame/camera inputs not verified')
    target = output / 'inputs/js_00001'
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        target.symlink_to(source_input, target_is_directory=True)
    if target.resolve() != source_input:
        raise ValueError('First-frame input changed')
    config = yaml.safe_load((PREVIOUS / 'inference_config.yaml').read_text())
    config['training']['distributed'].update(num_gpus=4, hsdp_replicate_dim=1, hsdp_shard_dim=4)
    config_path = output / 'inference_config.yaml'
    if config_path.exists() and yaml.safe_load(config_path.read_text()) != config:
        raise ValueError('Inference configuration changed')
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    save(
        output / 'checkpoints.json',
        dict(old,
             frozen_checkpoint=str(view),
             requested_cases=2,
             config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
             checkpoint_hash_reference=str(PREVIOUS / 'checkpoints.json'),
             experiment_type='short prompt-only intervention on yesterday checkpoint-800',
             previous_long_prompt_run=str(PREVIOUS),
             inference_gpus=4,
             parallel_videos=2))
    save(output / 'status.json', dict(phase='setup', expected_videos=2, text_condition_used=True))


def audit(output: Path) -> None:
    import numpy as np
    from scripts.evaluation.portable_js_easy_h3 import read_row

    cases = json.loads((output / 'inference_manifest.json').read_text())
    if len(cases) != 2:
        raise ValueError('Only two cases are authorized')
    rows = [read_row(Path(c['condition_parquet'])) for c in cases]
    for key in ('vae_latent_bytes', 'first_frame_latent_bytes', 'camera_intrinsics_bytes', 'camera_to_world_bytes'):
        if rows[0][key] != rows[1][key]:
            raise ValueError(f'Non-text pair input changed: {key}')
    for case, row in zip(cases, rows, strict=True):
        if row['caption'] != case['prompt']:
            raise ValueError('Short prompt was not encoded')
        encoding = json.loads((Path(case['condition_parquet']).parent / 'encoding.json').read_text())
        if encoding['prompt_sha256'] != hashlib.sha256(case['prompt'].encode()).hexdigest():
            raise ValueError('Prompt hash mismatch')
        tags = np.frombuffer(row['text_token_tags_bytes'], dtype=row['text_token_tags_dtype'])
        if not (tags == 1).any() or not (tags == 0).any() or len(tags) > 1024:
            raise ValueError('Actual text/vision tokens missing or truncated')
    if rows[0]['text_embedding_bytes'] == rows[1]['text_embedding_bytes']:
        raise ValueError('Opposite prompts have identical embeddings')
    save(
        output / 'text_condition_audit.json',
        dict(passed=True, cases=2, matched_image_camera_seed=True, full_short_text_encoded=True,
             embeddings_differ=True))
    (output / 'TEXT_CONDITIONS_COMPLETE').touch()


def finish(output: Path) -> None:
    import cv2
    import torch
    from tools.visualize_h3_controls import render_root

    cases = json.loads((output / 'inference_manifest.json').read_text())
    latents = []
    for case in cases:
        folder = output / 'videos' / case['output_key']
        if not (folder / 'VIDEO_COMPLETE').is_file():
            raise ValueError('Decoded output missing')
        metadata = json.loads((folder / 'metadata.json').read_text())
        if not metadata['scene_or_subject_text_used'] or metadata['conditioning_audit'][
                'text_token_count_entering_dit'] <= 0:
            raise ValueError('Short text did not enter actual DiT')
        cap = cv2.VideoCapture(str(folder / 'raw.mp4'))
        count = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame.shape != (480, 832, 3):
                    raise ValueError('Invalid output geometry')
                count += 1
        finally:
            cap.release()
        if count != 124:
            raise ValueError(f'Incomplete video: {count} frames')
        latents.append(torch.load(folder / 'latents.pt', map_location='cpu', weights_only=True).float())
    delta = float((latents[0] - latents[1]).square().mean().sqrt())
    save(
        output / 'text_response_summary.json',
        dict(videos_complete=2,
             checkpoint='union/checkpoint-800',
             forward_backward_latent_rms_difference=delta,
             text_entered_dit=True,
             caveat='Sensitivity is not proof of correct subject direction; no matched GT'))
    if not render_root(output, workers=2):
        raise ValueError('HUD incomplete')
    (output / 'INFERENCE_COMPLETE').touch()
    save(
        output / 'status.json',
        dict(phase='complete',
             videos_complete=2,
             text_condition_used=True,
             next_action='resume eight-GPU text training'))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['setup', 'audit', 'finish'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    globals()[args.command](args.output)


if __name__ == '__main__':
    main()
