"""CPU-only regression for inference model bootstrap and honest text metadata."""
import ast
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/evaluation/portable_js_easy_h3.py"


def config_builder():
    tree = ast.parse(SCRIPT.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "inference_config")
    namespace = {"deepcopy": deepcopy}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SCRIPT), "exec"), namespace)
    return namespace["inference_config"]


def source_config():
    return {
        "models": {"student": {"init_from_training_checkpoint": "/model/parent-100",
                               "vision_only_conditioning": False}},
        "training": {"checkpoint": {"resume_from_checkpoint": "/model/old-1100"}},
    }


def test_explicit_bootstrap_preserves_model_but_not_training_resume():
    source = source_config()
    result = config_builder()(source, preserve_init_checkpoint=True)
    assert result["models"]["student"]["init_from_training_checkpoint"] == "/model/parent-100"
    assert result["training"]["checkpoint"]["resume_from_checkpoint"] == ""
    assert result["models"]["student"]["vision_only_conditioning"] is False
    assert source == source_config()


def test_default_legacy_model_bootstrap_behavior_is_unchanged():
    result = config_builder()(source_config())
    assert result["models"]["student"]["init_from_training_checkpoint"] == ""
    assert result["training"]["checkpoint"]["resume_from_checkpoint"] == ""


def test_four_gpu_override_changes_only_the_inference_mesh():
    source = source_config()
    source["training"]["distributed"] = dict(num_gpus=8, sp_size=2, tp_size=1,
                                            hsdp_replicate_dim=2, hsdp_shard_dim=4)
    result = config_builder()(source, preserve_init_checkpoint=True, num_gpus=4)
    assert result["training"]["distributed"] == dict(num_gpus=4, sp_size=2, tp_size=1,
                                                    hsdp_replicate_dim=1, hsdp_shard_dim=4)
    assert source["training"]["distributed"]["num_gpus"] == 8
    assert result["models"] == source["models"]


def test_unsupported_inference_mesh_is_rejected():
    with pytest.raises(ValueError, match="four or eight"):
        config_builder()(source_config(), num_gpus=6)


def test_status_does_not_mislabel_new_checkpoint_as_800():
    source = SCRIPT.read_text()
    assert '"checkpoint_step": 800' not in source
    assert '"checkpoint_step": int(args.checkpoint.name.split("-")[-1])' in source


def test_score_text_support_is_derived_from_manifest():
    source = (ROOT / "scripts/evaluation/score_portable_js_easy_h3.py").read_text()
    assert 'uses_text = any(case.get("text_condition_used", False) for case in cases)' in source
    assert 'unsupported by current vision-only checkpoint' not in source
