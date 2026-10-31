"""Accuracy and latency of the driver's VQA weather labels (GPU).

For each scene in the hand-labelled weather preset, takes evenly spaced
front-camera frames the driver received in a recorded rollout (`driver_camera_image`
entries of rollout.asl), asks Alpamayo 1.5 for the weather with the driver's own
`classify` with its weather question, and compares the majority answer with the hand label.

Usage (repo root, on a GPU node):
  HF_HOME=$PWD/data/hf_cache HF_HUB_OFFLINE=1 \
    uv run python benchmark/weather_label_eval.py --rollouts benchmark/runs/alpamayo1_5
"""

import argparse
import asyncio
import collections
import glob
import io
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from alpasim_driver.models.alpamayo1_5_model import Alpamayo15Model
from alpasim_driver.models.base import CameraFrame
from alpasim_driver.scene_labels import WEATHER
from alpasim_driver.schema import ModelConfig
from alpasim_utils.logs import async_read_pb_log
from PIL import Image

CAMERA = "camera_front_wide_120fov"
LABELS_FILE = Path("src/wizard/configs/exp/weather/dev_scenes.yaml")


async def _front_frames(asl_path: str) -> list[CameraFrame]:
    frames = []
    async for entry in async_read_pb_log(asl_path):
        if entry.HasField("driver_camera_image"):
            image = entry.driver_camera_image.camera_image
            if image.logical_id == CAMERA:
                rgb = Image.open(io.BytesIO(image.image_bytes)).convert("RGB")
                frames.append(CameraFrame(image.frame_end_us, np.asarray(rgb)))
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", default="nvidia/Alpamayo-1.5-10B")
    parser.add_argument("--rollouts", default="benchmark/runs/alpamayo1_5")
    parser.add_argument("--frames-per-scene", type=int, default=5)
    parser.add_argument("--output", default="benchmark/weather_label_eval.json")
    args = parser.parse_args()

    hand_labels = yaml.safe_load(LABELS_FILE.read_text())["runtime"][
        "simulation_config"
    ]["scene_weather"]
    model = Alpamayo15Model.from_config(
        ModelConfig(model_type="alpamayo1_5", checkpoint_path=args.checkpoint),
        torch.device("cuda"),
        camera_ids=[CAMERA],
        context_length=1,
        output_frequency_hz=Alpamayo15Model.OUTPUT_FREQUENCY_HZ,
    )

    results, latencies_ms = [], []
    for scene_id, truth in hand_labels.items():
        asl = sorted(glob.glob(f"{args.rollouts}/rollouts/{scene_id}/*/rollout.asl"))
        if not asl:
            print(f"{scene_id}: no rollout in {args.rollouts}, skipped")
            continue
        frames = asyncio.run(_front_frames(asl[0]))
        picks = np.linspace(0, len(frames) - 1, args.frames_per_scene).astype(int)
        answers = []
        for i in picks:
            torch.cuda.synchronize()
            start = time.perf_counter()
            probabilities = model.classify(
                CAMERA, frames[i], WEATHER.question, WEATHER.labels
            )
            torch.cuda.synchronize()
            latencies_ms.append((time.perf_counter() - start) * 1e3)
            answers.append(
                {
                    "time_us": frames[i].timestamp_us,
                    "answer": max(probabilities, key=probabilities.__getitem__),
                    "probabilities": probabilities,
                }
            )
        majority = collections.Counter(a["answer"] for a in answers).most_common(1)[0][
            0
        ]
        results.append(
            {
                "scene_id": scene_id,
                "hand_label": truth,
                "majority": majority,
                "answers": answers,
            }
        )
        print(
            f"{scene_id[7:15]} truth={truth:5s} majority={majority:5s} "
            f"answers={[a['answer'] for a in answers]}"
        )

    summary = {
        "scene_accuracy": float(
            np.mean([r["majority"] == r["hand_label"] for r in results])
        ),
        "frame_accuracy": float(
            np.mean(
                [a["answer"] == r["hand_label"] for r in results for a in r["answers"]]
            )
        ),
        "latency_ms_median": float(np.median(latencies_ms)),
        "latency_ms_p95": float(np.percentile(latencies_ms, 95)),
    }
    print(json.dumps(summary, indent=2))
    Path(args.output).write_text(
        json.dumps({"summary": summary, "scenes": results}, indent=2)
    )


if __name__ == "__main__":
    main()
