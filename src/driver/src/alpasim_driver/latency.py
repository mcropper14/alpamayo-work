# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Per-inference latency and GPU memory recording for benchmarking.

The driver writes one JSON line per ``Drive`` call to ``latency.jsonl``.  Model
latency is measured inside the driver (CUDA-synchronized), so it stays
meaningful for drivers whose RPC returns before inference finishes.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from typing import IO, Any

import torch

LATENCY_LOG_NAME = "latency.jsonl"


def synchronized_now(device: torch.device) -> float:
    """Return ``perf_counter`` after all queued GPU work on ``device`` finished."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


@dataclass
class StageTimestamps:
    """Synchronized timestamps of model stages within one inference.

    Filled by forward hooks; the timestamps are ``perf_counter`` seconds.
    """

    vlm_first_start: float | None = None
    vlm_first_end: float | None = None
    vlm_last_end: float | None = None
    vlm_forward_count: int = 0
    vision_start: float | None = None
    vision_end: float | None = None
    # Per-inference counts reported by model components (e.g. visual cache hits).
    counters: dict[str, float] = field(default_factory=dict)

    def stage_durations_ms(self, start: float, end: float) -> dict[str, float]:
        """Split ``[start, end]`` into stages that sum to the full duration.

        Stages: input preprocessing up to the first VLM forward, vision encoder,
        the rest of the prefill forward, chain-of-thought decoding (one forward
        per generated token), and everything after the last VLM forward
        (diffusion trajectory sampling and postprocessing).
        """
        if self.vlm_first_start is None:
            return {}
        assert self.vlm_first_end is not None and self.vlm_last_end is not None
        vision_s = (
            self.vision_end - self.vision_start
            if self.vision_start is not None and self.vision_end is not None
            else 0.0
        )
        return {
            "preprocess_ms": (self.vlm_first_start - start) * 1e3,
            "vision_encoder_ms": vision_s * 1e3,
            "llm_prefill_ms": (self.vlm_first_end - self.vlm_first_start - vision_s)
            * 1e3,
            "cot_decode_ms": (self.vlm_last_end - self.vlm_first_end) * 1e3,
            "trajectory_ms": (end - self.vlm_last_end) * 1e3,
            "cot_decode_steps": float(self.vlm_forward_count - 1),
            **self.counters,
        }


class StageRecorder:
    """Receives forward-hook callbacks and timestamps the current inference.

    Valid for one inference at a time (driver batch size 1); call
    :meth:`reset` before each inference.
    """

    def __init__(self, device: torch.device) -> None:
        self._device = device
        self.current = StageTimestamps()

    def reset(self) -> None:
        self.current = StageTimestamps()

    def vlm_started(self) -> None:
        if self.current.vlm_first_start is None:
            self.current.vlm_first_start = synchronized_now(self._device)

    def vlm_finished(self) -> None:
        now = synchronized_now(self._device)
        if self.current.vlm_first_end is None:
            self.current.vlm_first_end = now
        self.current.vlm_last_end = now
        self.current.vlm_forward_count += 1

    def add_counters(self, counters: dict[str, float]) -> None:
        for name, value in counters.items():
            self.current.counters[name] = self.current.counters.get(name, 0.0) + value

    def vision_started(self) -> None:
        if self.current.vision_start is None:
            self.current.vision_start = synchronized_now(self._device)

    def vision_finished(self) -> None:
        if self.current.vision_end is None:
            self.current.vision_end = synchronized_now(self._device)


@dataclass
class LatencyLog:
    """Thread-safe JSONL writer for per-``Drive`` latency records."""

    path: str
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _file: IO[str] | None = None

    def write(self, record: dict[str, Any]) -> None:
        line = json.dumps(record)
        with self._lock:
            if self._file is None:
                self._file = open(self.path, "a", encoding="utf-8")
            self._file.write(line + "\n")
            self._file.flush()

    def close(self) -> None:
        with self._lock:
            if self._file is not None:
                self._file.close()
                self._file = None


def gpu_memory_gib(device: torch.device) -> dict[str, float]:
    """Peak allocated/reserved GPU memory since process start, in GiB."""
    if device.type != "cuda":
        return {}
    return {
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
    }
