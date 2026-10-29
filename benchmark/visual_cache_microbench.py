"""Isolated vision-encoder benchmark for the visual feature cache (GPU).

Replays the driver's image stream on Alpamayo 1.5's Qwen3-VL encoder: each
step feeds `--cameras` x `--context` frames (a window sliding by one frame per
camera), repeated once per trajectory sample as `generate` does. Compares the
original `get_image_features` with the cached one (exact strategy):

* encoder latency per step (CUDA-synchronized),
* number of images actually encoded,
* largest deviation of the cached features from the uncached ones. Each image
  is encoded independently, so deviations come only from bf16 kernels picking
  different tilings for different batch sizes.

Frames are random: neither timing nor exact-reuse correctness depends on image
content. Near-duplicate reuse depends on real scene motion; evaluate it in
closed loop.

Usage (repo root, on a GPU node):
  HF_HOME=$PWD/data/hf_cache HF_HUB_OFFLINE=1 \
    uv run python benchmark/visual_cache_microbench.py
"""

import argparse
import json
import time

import torch
from alpamayo1_5 import helper
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from alpasim_driver.schema import VisualCacheConfig
from alpasim_driver.visual_cache import install_visual_cache


def synchronized_ms(start: float) -> float:
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1e3


def max_deviation(cached, reference) -> dict[str, float]:
    cached_embeds, cached_deepstack = cached
    ref_embeds, ref_deepstack = reference
    pairs = [(torch.cat(cached_embeds), torch.cat(ref_embeds))] + list(
        zip(cached_deepstack, ref_deepstack, strict=True)
    )
    abs_dev = max((c.float() - r.float()).abs().max().item() for c, r in pairs)
    scale = max(r.float().abs().max().item() for _, r in pairs)
    return {"max_abs_deviation": abs_dev, "max_rel_deviation": abs_dev / scale}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", default="nvidia/Alpamayo-1.5-10B")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--cameras", type=int, default=4)
    parser.add_argument("--context", type=int, default=4)
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--max-entries", type=int, default=64)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--width", type=int, default=1900)
    parser.add_argument("--output", default="benchmark/visual_cache_microbench.json")
    args = parser.parse_args()

    device = torch.device("cuda")
    model = Alpamayo1_5.from_pretrained(
        args.checkpoint, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(device)
    model.eval()
    processor = helper.get_processor(model.tokenizer)
    qwen = model.vlm.model
    original = qwen.get_image_features
    encoder = install_visual_cache(
        qwen, VisualCacheConfig(enabled=True, max_entries=args.max_entries)
    )
    encoded_counts: list[float] = []
    encoder.on_call = lambda counters: encoded_counts.append(counters["vision_encoded"])

    generator = torch.Generator(device=device).manual_seed(0)
    frames = torch.randint(
        0,
        256,
        (args.cameras, args.steps + args.context, 3, args.height, args.width),
        dtype=torch.uint8,
        device=device,
        generator=generator,
    )

    def step_inputs(step: int) -> tuple[torch.Tensor, torch.Tensor]:
        images = [
            frames[camera, step + offset]
            for camera in range(args.cameras)
            for offset in range(args.context)
        ]
        processed = processor.image_processor(images=images, return_tensors="pt")
        pixel_values = processed["pixel_values"].to(device)
        grid_thw = processed["image_grid_thw"].to(device)
        # generate() repeats a sample's images once per returned sequence.
        return pixel_values.repeat(args.samples, 1), grid_thw.repeat(args.samples, 1)

    records = []
    with torch.no_grad():
        # Warm up kernels on both paths with frames outside the measured stream.
        warmup = step_inputs(args.steps)
        original(*warmup)
        torch.cuda.synchronize()

        memory_before = torch.cuda.memory_allocated()
        for step in range(args.steps):
            inputs = step_inputs(step)

            start = time.perf_counter()
            reference = original(*inputs)
            uncached_ms = synchronized_ms(start)

            start = time.perf_counter()
            cached = encoder.get_image_features(*inputs)
            cached_ms = synchronized_ms(start)

            records.append(
                {
                    "step": step,
                    "images": int(inputs[1].shape[0]),
                    "encoded": encoded_counts[-1],
                    "uncached_ms": uncached_ms,
                    "cached_ms": cached_ms,
                    **max_deviation(cached, reference),
                }
            )
            print(json.dumps(records[-1]))
        cache_memory_gib = (torch.cuda.memory_allocated() - memory_before) / 2**30

    # Step 0 fills the cache; later steps are the steady state.
    steady = records[1:]
    summary = {
        "gpu": torch.cuda.get_device_name(),
        "images_per_step": records[0]["images"],
        "encoded_per_step_steady": sum(r["encoded"] for r in steady) / len(steady),
        "uncached_ms_mean": sum(r["uncached_ms"] for r in steady) / len(steady),
        "cached_ms_mean": sum(r["cached_ms"] for r in steady) / len(steady),
        "max_abs_deviation": max(r["max_abs_deviation"] for r in records),
        "max_rel_deviation": max(r["max_rel_deviation"] for r in records),
        "cache_memory_gib": cache_memory_gib,
        "args": vars(args),
    }
    summary["speedup"] = summary["uncached_ms_mean"] / summary["cached_ms_mean"]
    print(json.dumps(summary, indent=2))
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "steps": records}, f, indent=2)


if __name__ == "__main__":
    main()
