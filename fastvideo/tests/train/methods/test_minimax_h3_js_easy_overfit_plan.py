"""CPU tests for the measured-pose JS-easy diagnostic selector."""

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from examples.train.prepare_minimax_h3_js_easy_overfit import eligible_groups, freeze_plan, parquet_indices
from examples.train.stage_minimax_h3_js_easy_overfit import stage_rows
from examples.train.verify_minimax_h3_vae_cache import _selected_ids
from fastvideo.dataset.dataloader.schema import pyarrow_schema_minimax_h3_worldmodel_prope_vae


def _source_row(scene: int, direction: str, *, axis_fraction: float = 0.95) -> dict:
    global_id = f"h3_js_easy:h3_i2v_js_easy_124_js_{scene:05d}_{direction}_fast_camera_only"
    return {
        "global_id": global_id,
        "sample_id": f"sample_{scene}_{direction}",
        "source_name": "h3_js_easy_h3_i2v_js_easy_124",
        "gpu_qc_pass": True,
        "quality_approved": False,
        "actions": {
            direction: {
                "axis_fraction": axis_fraction,
                "direction_consistency": 0.95,
            }
        },
    }


def test_js_easy_plan_keeps_measured_direction_and_expands_to_all_eight():
    rows = [_source_row(scene, direction) for scene in (1, 2) for direction in "WASDIJKL"]
    train_ids = {row["global_id"] for row in rows if "js_00001_" in row["global_id"]}
    groups = eligible_groups(rows, train_ids, min_axis_fraction=0.8, min_direction_consistency=0.8)
    phases, indices, report = freeze_plan(groups, pair_limit=1, seed=3)

    assert indices == {}
    assert report["available_pairs"] == {"JL": 1, "IK": 1, "AD": 1, "WS": 1}
    assert report["phase_rows"] == {"stage1_jl": 2, "stage2_jlik": 4, "stage3_all8": 8}
    assert {row["nominal_direction"] for row in phases["stage3_all8"]} == set("WASDIJKL")
    assert all(not row["quality_approved"] for row in phases["stage3_all8"])


def test_js_easy_plan_rejects_weak_direction_and_missing_conditioned_row():
    rows = [_source_row(1, direction) for direction in "WASDIJKL"]
    rows[0]["actions"]["W"]["axis_fraction"] = 0.1
    train_ids = {row["global_id"] for row in rows}
    groups = eligible_groups(rows, train_ids, min_axis_fraction=0.8, min_direction_consistency=0.8)

    assert "W" not in next(iter(groups.values()))
    try:
        freeze_plan(groups, pair_limit=1, seed=3)
    except ValueError as exc:
        assert "WS" in str(exc)
    else:
        raise AssertionError("missing paired direction must fail the plan")


def test_js_easy_stage_reuses_existing_vae_rows_and_builds_ordered_indices(tmp_path):
    source = tmp_path / "vae"
    source.mkdir()
    table = pa.Table.from_pylist(
        [{"id": "source/j"}, {"id": "source/l"}], schema=pyarrow_schema_minimax_h3_worldmodel_prope_vae
    )
    pq.write_table(table, source / "data.parquet")
    plan = tmp_path / "stage3_all8_ids.jsonl"
    plan.write_text(
        json.dumps({"parquet_id": "source/l"}) + "\n" + json.dumps({"parquet_id": "source/j"}) + "\n"
    )

    report = stage_rows(plan, source, tmp_path / "staged", shard_size=1)
    assert report["rows"] == 2 and report["shards"] == 2
    staged = tmp_path / "staged/intermediate/train"
    assert pq.read_table(staged / "data_000000.parquet", columns=["id"])["id"].to_pylist() == ["source/l"]
    assert pq.read_table(staged / "data_000001.parquet", columns=["id"])["id"].to_pylist() == ["source/j"]
    assert parquet_indices(staged) == {"source/l": 0, "source/j": 1}


def test_frozen_vae_audit_selection_rejects_duplicate_ids(tmp_path):
    plan = tmp_path / "ids.jsonl"
    plan.write_text(json.dumps({"parquet_id": "source/j"}) + "\n" + json.dumps({"parquet_id": "source/l"}) + "\n")
    assert _selected_ids(plan) == {"source/j", "source/l"}
    plan.write_text(json.dumps({"parquet_id": "source/j"}) + "\n" + json.dumps({"parquet_id": "source/j"}) + "\n")
    with pytest.raises(ValueError, match="unique"):
        _selected_ids(plan)
