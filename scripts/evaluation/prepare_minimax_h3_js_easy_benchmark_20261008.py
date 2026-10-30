# SPDX-License-Identifier: Apache-2.0
"""Encode the exact JS-easy benchmark first frames for H3 action-only evaluation.

Existing conditioned rows provide the latent shape and calibrated intrinsics;
neither their first-frame latent nor their Qwen vision tokens are reused.
The source video latent is only a shape carrier for the inference evaluator.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.distributed as dist

from fastvideo.dataset.dataloader.schema import pyarrow_schema_minimax_h3_worldmodel_prope
from fastvideo.pipelines.basic.minimax_h3.packing import MINIMAX_H3_KEYFRAME_ENCODE_SEED
from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_conditioning import MiniMaxH3ConditioningStage
from fastvideo.pipelines.preprocess.minimax_h3_worldmodel.encoding import encode_h3_worldmodel_condition
from fastvideo.pipelines.preprocess.minimax_h3_worldmodel.preprocess import (
    _fastvideo_args,
    _initialize_worker,
    _load_component,
    _load_model_index,
)


CASES = {
    "js_00004": ("worker_000/data_000015.parquet", 4, "000524_5e175d44c6b5b986"),
    "js_00007": ("worker_003/data_000013.parquet", 12, "000588_0499df98ce2e3004"),
    "js_00009": ("worker_005/data_000000.parquet", 3, "000642_5f3ca4ec1b5af8fb"),
    "js_00012": ("worker_007/data_000018.parquet", 5, "000697_cffa075021b96e87"),
}
CAPTION = "A video scene observed from a moving camera."


def _crop_benchmark_image(path: Path) -> tuple[Image.Image, tuple[int, int]]:
    with Image.open(path) as raw:
        pixels = np.asarray(raw.convert("RGB"))
    source_hw = pixels.shape[:2]
    scale = max(480 / source_hw[0], 832 / source_hw[1])
    resized_h = max(480, round(source_hw[0] * scale))
    resized_w = max(832, round(source_hw[1] * scale))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(pixels, (resized_w, resized_h), interpolation=interpolation)
    top, left = (resized_h - 480) // 2, (resized_w - 832) // 2
    cropped = np.ascontiguousarray(resized[top:top + 480, left:left + 832])
    if cropped.shape != (480, 832, 3):
        raise ValueError(f"Unexpected benchmark crop: {cropped.shape}")
    return Image.fromarray(cropped), source_hw


def _read_cached_row(path: Path, index: int, expected_suffix: str) -> dict:
    if not pq.read_schema(path).equals(pyarrow_schema_minimax_h3_worldmodel_prope, check_metadata=False):
        raise ValueError(f"Unexpected cache schema: {path}")
    row = pq.read_table(path).slice(index, 1).to_pylist()[0]
    if not row["id"].endswith("/" + expected_suffix):
        raise ValueError(f"Unexpected cached source for {path}:{index}: {row['id']}")
    if row["first_frame_latent_shape"] != [24, 1, 30, 52] or row["vae_latent_shape"] != [24, 37, 30, 52]:
        raise ValueError(f"Unexpected cached H3 latent geometry: {row['id']}")
    return row


def _encode_keyframe(vae: object, image: Image.Image, device: torch.device) -> np.ndarray:
    pixels = np.asarray(image)
    video = torch.from_numpy(pixels.copy()).permute(2, 0, 1)[None, :, None].to(
        device=device, dtype=torch.float32
    ).div_(255.0)
    with torch.no_grad():
        posterior = vae.encode_keyframe(vae.normalize_pixels(video)).latent_dist
        generator = torch.Generator("cpu").manual_seed(MINIMAX_H3_KEYFRAME_ENCODE_SEED)
        sample = posterior.sample(generator=generator).to(torch.float16).float()
        latent = vae.normalize_latents(sample).squeeze(0).float().cpu().contiguous().numpy()
    if latent.shape != (24, 1, 30, 52) or not np.isfinite(latent).all():
        raise ValueError(f"Invalid native H3 keyframe latent: {latent.shape}")
    return latent


def _set_float32_tensor(row: dict, name: str, value: np.ndarray) -> None:
    array = np.ascontiguousarray(value, dtype=np.float32)
    row[f"{name}_bytes"] = array.tobytes()
    row[f"{name}_shape"] = list(array.shape)
    row[f"{name}_dtype"] = "float32"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--master-port", type=int, default=29983)
    parser.add_argument("--all-benchmark-images", action="store_true",
                        help="Prepare every js_* first frame; uncatalogued cases use a verified shape carrier")
    args = parser.parse_args()

    cases_to_prepare = dict(CASES)
    if args.all_benchmark_images:
        image_paths = sorted(args.benchmark_root.glob("js_*/sample_001_first_frame.png"))
        if len(image_paths) != 17:
            raise ValueError(f"Expected 17 JS-easy benchmark first frames, found {len(image_paths)}")
        # The cached target video is read for latent shape only. In this run,
        # the exact benchmark PNG replaces both the H3 keyframe latent and
        # Qwen vision tokens; inference supplies a fresh numeric camera path.
        carrier = CASES["js_00004"]
        for image_path in image_paths:
            cases_to_prepare.setdefault(image_path.parent.name, carrier)

    cases = {}
    for name, (shard, index, expected_suffix) in cases_to_prepare.items():
        image_path = args.benchmark_root / name / "sample_001_first_frame.png"
        cache_path = args.cache_root / shard
        if not image_path.is_file() or not cache_path.is_file():
            raise FileNotFoundError(f"Missing benchmark image or conditioned cache: {name}")
        image, source_hw = _crop_benchmark_image(image_path)
        source_row = _read_cached_row(cache_path, index, expected_suffix)
        case_dir = args.output_root / name
        parquet_path = case_dir / "data_000000.parquet"
        image_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()
        if parquet_path.is_file():
            metadata = json.loads((case_dir / "source.json").read_text(encoding="utf-8"))
            if metadata.get("benchmark_png_sha256") != image_hash or metadata.get("cached_source_id") != source_row["id"]:
                raise ValueError(f"Existing benchmark condition does not match source: {case_dir}")
            if pq.ParquetFile(parquet_path).metadata.num_rows != 1:
                raise ValueError(f"Existing benchmark condition has multiple rows: {parquet_path}")
            print(f"reusing exact benchmark condition: {name}", flush=True)
            continue
        cases[name] = (image_path, image, source_hw, image_hash, source_row, case_dir)

    if cases:
        device = _initialize_worker(args.device_index, args.master_port)
        model_path = args.model_path.resolve()
        model_index = _load_model_index(model_path)
        fastvideo_args = _fastvideo_args(model_path)
        vae = _load_component("vae", model_path, model_index, fastvideo_args)
        vae.eval()
        keyframes = {}
        for name, (_, image, _, _, _, _) in cases.items():
            keyframes[name] = _encode_keyframe(vae, image, device)
            print(f"encoded benchmark keyframe: {name}", flush=True)
        del vae
        gc.collect()
        torch.cuda.empty_cache()

        tokenizer = _load_component("tokenizer", model_path, model_index, fastvideo_args)
        processor = _load_component("processor", model_path, model_index, fastvideo_args)
        conditioner = _load_component("text_encoder", model_path, model_index, fastvideo_args)
        stage = MiniMaxH3ConditioningStage(conditioner=conditioner, tokenizer=tokenizer, processor=processor)
        for name, (image_path, image, source_hw, image_hash, source_row, case_dir) in cases.items():
            embedding, tags = encode_h3_worldmodel_condition(
                stage, fastvideo_args, caption=CAPTION, first_frame=image
            )
            row = dict(source_row)
            row["id"] = f"js_easy_benchmark/{name}"
            row["source_name"] = "js_easy_benchmark"
            row["caption"] = CAPTION
            _set_float32_tensor(row, "first_frame_latent", keyframes[name])
            _set_float32_tensor(row, "text_embedding", embedding.numpy())
            _set_float32_tensor(row, "text_token_tags", tags.numpy())
            case_dir.mkdir(parents=True, exist_ok=True)
            image.save(case_dir / "input_first_frame.png")
            output_path = case_dir / "data_000000.parquet"
            temporary = output_path.with_suffix(".parquet.tmp")
            pq.write_table(pa.Table.from_pylist([row], schema=pyarrow_schema_minimax_h3_worldmodel_prope),
                           temporary, compression="zstd")
            os.replace(temporary, output_path)
            metadata = {
                "case_id": name,
                "benchmark_png": str(image_path.resolve()),
                "benchmark_png_sha256": image_hash,
                "benchmark_hw": list(source_hw),
                "condition_hw": [480, 832],
                "cached_source_id": source_row["id"],
                "cached_target_latent_used_for_shape_only": True,
                "keyframe_and_qwen_vision_encoded_from_exact_benchmark_png": True,
                "scene_text_entering_dit": False,
            }
            (case_dir / "source.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
            print(f"wrote exact benchmark condition: {name} -> {output_path}", flush=True)
        if dist.is_initialized():
            dist.destroy_process_group()
    print(f"Prepared {len(cases_to_prepare)} JS-easy benchmark first frames in {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
