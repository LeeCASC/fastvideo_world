"""Configure the approved subject-motion run only after complete data audits."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import yaml

from tools.prepare_union_dual_captions_20261023 import registry_specs, selection_hash


def configure(root: Path, template_path: Path | None = None) -> None:
    variant = root / 'subject_motion'
    counts = json.loads((variant / 'data_plan_summary.json').read_text())
    audit = json.loads((variant / 'text_data_audit.json').read_text())
    captions = json.loads((root / 'CAPTIONS_VERIFIED').read_text())
    if not audit['passed'] or audit['rows'] != counts['total'] or captions['rows'] != counts['total']:
        raise ValueError('Incomplete or inconsistent union text data audit')
    selection_sha256 = selection_hash(root)
    expected_total = 115988
    if selection_sha256 is not None:
        included = registry_specs(root)
        selection = json.loads((root / 'caption_selection.json').read_text())
        if selection['original_rows'] != 115988:
            raise ValueError('Authorized caption selection must preserve the original full-union registry')
        expected_total = len(included)
        with (variant / 'text_data_plan.jsonl').open() as handle:
            planned_ids = [json.loads(line)['id'] for line in handle if line.strip()]
        if len(planned_ids) != len(set(planned_ids)) or set(planned_ids) != {s['id'] for s in included}:
            raise ValueError('Training plan differs from authorized caption selection')
    if (counts['total'] != expected_total or counts.get('selection_sha256') != selection_sha256
            or captions.get('selection_sha256') != selection_sha256):
        raise ValueError('Do not silently train on a partial union')
    plan_hash = hashlib.sha256((variant / 'text_data_plan.jsonl').read_bytes()).hexdigest()
    registry_hash = hashlib.sha256((root / 'registry.jsonl').read_bytes()).hexdigest()
    if (counts.get('plan_sha256') != plan_hash or audit.get('plan_sha256') != plan_hash
            or captions.get('plan_sha256', {}).get('subject_motion') != plan_hash
            or counts.get('registry_sha256') != registry_hash or captions.get('registry_sha256') != registry_hash
            or audit['split_counts'] != counts['split_counts']):
        raise ValueError('Frozen plan/registry/split changed after caption or payload verification')
    composition = counts.get('prompt_composition')
    if composition and (composition != 'background_plus_subject_exact_sentence_dedup_v1'
                        or captions.get('prompt_composition') != composition):
        raise ValueError('Background/subject composition changed after verification')
    if composition and shutil.disk_usage(root).free < 512 * 2**30:
        raise ValueError('Insufficient checkpoint reserve: free at least 512 GiB before starting the 35-save run')
    new_checkpoints = variant / 'runs/train/checkpoints'
    if (root / 'TRAINING_COMPLETE').exists() or any(new_checkpoints.glob('checkpoint-*')):
        raise ValueError('New-run checkpoints already exist: resume explicitly, never reset to parent100')
    prior = template_path or Path('/datacc05/lizihao/worldmodel/fastvideowm_scene_text_20261022/training_config.yaml')
    config = yaml.safe_load(prior.read_text())
    distributed = config['training']['distributed']
    data = config['training']['data']
    accumulation = config['training']['loop']['gradient_accumulation_steps']
    if (distributed['num_gpus'] != 8 or distributed['sp_size'] != 2 or data['train_batch_size'] * accumulation * 4 != 32
            or config['models']['student']['vision_only_conditioning'] is not False
            or config['training']['loop']['max_train_steps'] != 3000):
        raise ValueError('Authorized eight-GPU/SP2/BS32/text-on/3000-step recipe changed')
    config['training']['data']['data_path'] = str(variant / 'preprocessed/train')
    config['training']['checkpoint']['output_dir'] = str(variant / 'runs/train/checkpoints')
    config['training']['checkpoint']['resume_from_checkpoint'] = ''
    config['training']['tracker']['run_name'] = f'h3_union_subject_motion_text_{expected_total}_20261023'
    if composition:
        config['training']['tracker']['run_name'] = f'h3_union_scene_subject_motion_text_{expected_total}_20261024'
    config['callbacks']['worldmodel_validation']['data_path'] = str(variant / 'preprocessed/validation')
    config['callbacks']['text_input_guard']['output_dir'] = str(variant / 'runs/train')
    output = variant / 'training_config.yaml'
    if output.exists() and yaml.safe_load(output.read_text()) != config:
        raise ValueError('Refusing to overwrite a different existing training configuration')
    output.write_text(yaml.safe_dump(config, sort_keys=False))
    print(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    configure(parser.parse_args().root)
