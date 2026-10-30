from pathlib import Path

from tools import inventory_h3_model_checkpoints_20261024 as inventory


def test_generated_action_latents_are_not_model_checkpoints(tmp_path, monkeypatch):
    output = tmp_path / 'videos/checkpoint-750'
    output.mkdir(parents=True)
    (output / 'J.pt').write_bytes(b'generated latent, not model weights')
    (output / 'metadata.json').write_text('{}')
    monkeypatch.setattr(inventory, 'ROOTS', [str(tmp_path)])
    targets, errors = inventory.discover()
    assert not errors and not targets


def test_dcp_shards_are_grouped_and_shared_blocks_disclosed(tmp_path, monkeypatch):
    path = tmp_path / 'run/checkpoints/checkpoint-100'
    (path / 'dcp').mkdir(parents=True)
    (path / 'dcp/.metadata').write_bytes(b'metadata')
    shard = path / 'dcp/__0_0.distcp'
    shard.write_bytes(b'shard')
    (tmp_path / 'linked_shard').hardlink_to(shard)
    monkeypatch.setattr(inventory, 'ROOTS', [str(tmp_path)])
    targets, _ = inventory.discover()
    assert targets == {path: 'DCP checkpoint'}
    row = inventory.describe((path, targets[path]))
    assert row['shared_files'] == 1
    assert row['status'] == 'metadata=1; rank_shards=1'
    assert '硬链接' in row['note']


def test_scanning_excludes_dataset_latents_and_does_not_remove_files(tmp_path, monkeypatch):
    latent = tmp_path / 'vae_latents/model_cache.pt'
    latent.parent.mkdir()
    latent.write_bytes(b'dataset')
    weights = tmp_path / 'h3/step-100.safetensors'
    weights.parent.mkdir()
    weights.write_bytes(b'weights')
    monkeypatch.setattr(inventory, 'ROOTS', [str(tmp_path)])
    targets, _ = inventory.discover()
    assert targets == {weights.parent: 'weight directory'}
    assert latent.read_bytes() == b'dataset' and weights.read_bytes() == b'weights'


def test_symlink_checkpoint_is_an_alias_not_a_second_model(tmp_path):
    target = tmp_path / 'real/checkpoint-100'
    target.mkdir(parents=True)
    alias = tmp_path / 'checkpoint-100'
    alias.symlink_to(target, target_is_directory=True)
    row = inventory.describe((Path(alias), 'checkpoint symlink'))
    assert row['allocated_gib'] == 0.0 and row['files'] == 0
    assert str(target) in row['note']
