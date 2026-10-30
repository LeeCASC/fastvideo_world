"""Read-only H3/FastVideoWM model inventory; write reports, never delete models."""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import re
from zoneinfo import ZoneInfo

ROOTS = [
    '/datacc05/lizihao/worldmodel',
    '/datacc05/lizihao_wm/worldmodel',
    '/datacc05/lizihao_wm/worldmodel_prope',
    '/datacc05/lizihao/outputs/h3_world',
    '/home/lizihao/projects/worldmodel/H3-World/checkpoints',
    '/home/lizihao/projects/military_world_model_agent/FastVideoWM/outputs',
    '/home/lizihao/projects/military_world_model_agent/FastVideoWM/checkpoints',
    '/home/lizihao/projects/military_world_model_agent/minWM/checkpoints',
    '/datacc05/lizihao/models/h3_world/MiniMax-H3',
    '/datacc05/lizihao/models/worldarena/qwenvl3',
]
SKIP = {
    'preprocessed',
    'intermediate',
    'vae_latents',
    'video_latents',
    'first_frames',
    'poses',
    'captions',
    'datasets',
    'dataset',
    'visualized',
    'inputs',
    'conditions',
    'registry',
    'gpu_qc',
    '.git',
    '.venv',
    '__pycache__',
    'node_modules',
    'transfer_packages',
}
PARENT = '/datacc05/lizihao/worldmodel/fastvideowm_scene_text_20261022/runs/train/checkpoints/checkpoint-100'
BASES = [
    '/datacc05/lizihao_wm/worldmodel/ckpts/MiniMax-H3',
    '/datacc05/lizihao/models/h3_world/MiniMax-H3',
    '/datacc05/lizihao/models/worldarena/qwenvl3',
]


def discover() -> tuple[dict[Path, str], list[str]]:
    targets: dict[Path, str] = {}
    errors: list[str] = []
    for value in ROOTS:
        root = Path(value)
        if not root.exists():
            continue
        for current, dirs, files in os.walk(root, followlinks=False, onerror=lambda e: errors.append(str(e))):
            path = Path(current)
            if path.name.startswith('model_checkpoint_inventory_'):
                dirs[:] = []
                continue
            # A single DCP unit is one deletion/retention decision; individual
            # rank shards are not independent models.
            # Inference folders also use checkpoint-N names and contain
            # W.pt/J.pt OUTPUT latents. Those are not model checkpoints.
            has_weights = any(
                name.endswith(('.distcp', '.safetensors', '.ckpt')) or (name.endswith(('.pt', '.pth')) and re.match(
                    r'(?:model|pytorch_model|student|adapter|optimizer|weights|state_dict)[._-]', name))
                for name in files)
            is_checkpoint = bool(re.fullmatch(r'checkpoint-\d+', path.name)) and ('dcp' in dirs or has_weights)
            if is_checkpoint or ('dcp' in dirs and (path / 'dcp/.metadata').is_file()):
                targets[path] = 'DCP checkpoint'
                dirs[:] = []
                continue
            for name in dirs:
                child = path / name
                if child.is_symlink() and re.fullmatch(r'checkpoint-\d+', name):
                    targets[child] = 'checkpoint symlink'
            weights = [
                name for name in files
                if name.endswith(('.safetensors', '.ckpt', '.pth', '.pt', '.bin')) and (name.endswith(
                    ('.safetensors', '.ckpt')) or re.search(r'model|step|epoch|lora|optimizer|pytorch', name, re.I))
            ]
            if weights:
                targets[path] = 'weight directory'
            dirs[:] = [name for name in dirs if name not in SKIP and not (path / name).is_symlink()]
    return targets, errors


def describe(item: tuple[Path, str]) -> dict:
    path, kind = item
    if path.is_symlink():
        return dict(path=str(path),
                    kind=kind,
                    logical_gib=0.0,
                    allocated_gib=0.0,
                    unshared_gib=0.0,
                    files=0,
                    shared_files=0,
                    checkpoint_names='',
                    status='symlink only',
                    note=f'link -> {path.resolve()}')
    logical = allocated = unshared = count = shared = 0
    names: list[str] = []
    inode_seen: set[tuple[int, int]] = set()
    # For DCP include state/metadata/RNG. For a weight directory include only
    # immediate weight/state files; avoid adding nested models twice.
    paths = path.rglob('*') if kind == 'DCP checkpoint' else path.iterdir()
    for file in paths:
        if file.is_symlink() or not file.is_file():
            continue
        if kind != 'DCP checkpoint' and file.suffix not in {'.safetensors', '.ckpt', '.pth', '.pt', '.bin'}:
            continue
        info = file.stat()
        logical += info.st_size
        inode = (info.st_dev, info.st_ino)
        if inode not in inode_seen:
            inode_seen.add(inode)
            allocated += info.st_blocks * 512
            if info.st_nlink == 1:
                unshared += info.st_blocks * 512
        count += 1
        shared += int(info.st_nlink > 1)
        if kind != 'DCP checkpoint':
            names.append(file.name)
    status = 'weight files'
    if kind == 'DCP checkpoint':
        shards = list((path / 'dcp').glob('*.distcp'))
        status = f'metadata={int((path / "dcp/.metadata").is_file())}; rank_shards={len(shards)}'
    note = ''
    if str(path) == PARENT:
        note = '当前已批准续训父模型：请勿删除'
    elif str(path).startswith('/datacc05/lizihao/models/h3_world/MiniMax-H3'):
        note = 'H3-World原生推理基础模型；删除会影响旧H3-World流程'
    elif any(str(path).startswith(base) for base in BASES):
        note = '当前编码/训练依赖的基础模型或编码器：请勿整目录删除'
    if shared:
        note += ('；' if note else '') + '含硬链接共享文件，删除本目录不一定释放对应空间'
    return dict(path=str(path),
                kind=kind,
                logical_gib=round(logical / 2**30, 3),
                allocated_gib=round(allocated / 2**30, 3),
                unshared_gib=round(unshared / 2**30, 3),
                files=count,
                shared_files=shared,
                checkpoint_names='; '.join(sorted(names)),
                status=status,
                note=note)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    targets, errors = discover()
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = sorted(pool.map(describe, targets.items()), key=lambda r: (-r['allocated_gib'], r['path']))
    args.output.mkdir(parents=True, exist_ok=True)
    fields = [
        'path', 'kind', 'logical_gib', 'allocated_gib', 'unshared_gib', 'files', 'shared_files', 'checkpoint_names',
        'status', 'note'
    ]
    with (args.output / 'checkpoints.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with (args.output / 'weight_files.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['file', 'directory', 'size_gib', 'hardlinks', 'note'])
        writer.writeheader()
        for row in rows:
            if row['kind'] != 'weight directory':
                continue
            for file in sorted(Path(row['path']).iterdir()):
                if file.is_file() and file.suffix in {'.safetensors', '.ckpt', '.pth', '.pt', '.bin'}:
                    info = file.stat()
                    writer.writerow(
                        dict(
                            file=str(file),
                            directory=row['path'],
                            size_gib=round(info.st_size / 2**30, 4),
                            hardlinks=info.st_nlink,
                            note='H3 step文件' if re.match(r'step-\d+\.safetensors$', file.name) else '可能是基础模型分片，不要单独删除'))
    (args.output / 'model_directories.txt').write_text(''.join(row['path'] + '\n' for row in rows))
    payload = dict(generated_jst=datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(),
                   scan_roots=ROOTS,
                   excluded_data_directories=sorted(SKIP),
                   errors=errors,
                   directory_count=len(rows),
                   entries=rows)
    (args.output / 'inventory.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        path = Path(row['path'])
        group = str(path.parent) if row['kind'] in {'DCP checkpoint', 'checkpoint symlink'} else str(path)
        groups[group].append(row)
    lines = [
        '# H3-World / FastVideoWM 模型和checkpoint目录清单',
        '',
        f'生成时间：{payload["generated_jst"]}。只做文件元数据统计，没有加载模型，也没有删除文件。',
        '',
        f'共 {len(rows)} 个模型/权重目录。逐checkpoint完整表见 checkpoints.csv，纯路径见 model_directories.txt。',
        '',
        'H3-World逐个step文件及容量见 weight_files.csv（同时列出基础模型分片，注意区分）。',
        '',
        '## 删除前注意',
        '',
        '- 当前续训父模型必须保留：`' + PARENT + '`。',
        '- 当前基础模型/编码器依赖：' + '、'.join('`' + p + '`' for p in BASES) + '。',
        '- DCP的全部rank shard、metadata、RNG及optimizer共同组成一个checkpoint，不可只删除其中一张卡的文件。',
        '- 本项目部分DCP只保存可训练参数/优化器，不是独立完整的H3；保留checkpoint时也要保留对应基础模型和配置。',
        '- 含硬链接或软链接的评测快照可能与训练目录共享模型，删除一份不一定释放权重空间。',
        '- 容量为文件元数据估计；共享文件跨目录不能重复相加，NFS/ZFS快照也会影响实际释放量。',
        '- unshared_gib仅表示本清单目录内nlink=1的文件块，不是承诺可释放容量；请自行核对依赖。',
        '- 本清单排除了视频、latent、pose等训练数据，不要将整个worldmodel根目录当作模型目录删除。',
        '',
        '本次父checkpoint-100实测约13.39GiB；同配方35份约469GiB（未来大小可能变化）。用户已选择继续保留35份，本工具不改保留策略。',
        '',
        '## 各实验模型目录',
        '',
        '| 模型/实验目录 | CKPT目录数 | 文件占用GiB（可共享） | 内容/最新编号 | 注意 |',
        '| --- | ---: | ---: | --- | --- |',
    ]
    ordered = sorted(groups.items(), key=lambda kv: -sum(r['allocated_gib'] for r in kv[1]))
    for group, members in ordered:
        checkpoints = sorted((Path(r['path']).name for r in members if r['kind'] != 'weight directory'),
                             key=lambda s: int(s.split('-')[-1]) if s.split('-')[-1].isdigit() else 0)
        detail = ', '.join(checkpoints) if checkpoints else members[0]['checkpoint_names']
        notes = '；'.join(sorted({r['note'] for r in members if r['note']}))
        lines.append(f'| `{group}` | {len(checkpoints)} | {sum(r["allocated_gib"] for r in members):.2f} | '
                     f'{detail} | {notes} |')
    lines.extend([
        '', '## 扫描边界', '', *['- `' + root + '`' for root in ROOTS], '', '仅覆盖上列本项目及本人模型根目录，不是其他用户或整台服务器的模型枚举。', '',
        '扫描异常：' + (json.dumps(errors, ensure_ascii=False) if errors else '无。')
    ])
    (args.output / 'README_zh.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(output=str(args.output), directories=len(rows), errors=errors, largest=rows[:8]),
                     ensure_ascii=False),
          flush=True)


if __name__ == '__main__':
    main()
