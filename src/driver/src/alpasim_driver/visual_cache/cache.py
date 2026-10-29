# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Temporally persistent store of per-image vision-encoder outputs."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import torch


@dataclass
class CachedImage:
    """Vision-encoder output of one image.

    ``embeds`` holds the merged visual tokens of the image, ``deepstack`` the
    matching intermediate-layer features (one tensor per deepstack layer, same
    token count).  ``pixels`` is kept only for images the encoder actually ran
    on, and only when near-duplicate matching compares new images against them;
    entries that reuse another image's features have none, so every
    near-duplicate is measured against the image its features came from.
    """

    grid_thw: tuple[int, int, int]
    embeds: torch.Tensor
    deepstack: list[torch.Tensor]
    pixels: torch.Tensor | None
    # How many other images these features have been reused for.
    near_duplicate_uses: int = 0


class VisualFeatureCache:
    """LRU map from image fingerprint to its vision-encoder output.

    ``max_entries`` bounds memory.  It must cover the frames that are reused
    between inferences: cameras x context frames, times the number of rollouts
    whose inferences interleave on the driver.
    """

    def __init__(self, max_entries: int) -> None:
        if max_entries < 1:
            raise ValueError(f"max_entries must be at least 1, got {max_entries}.")
        self._max_entries = max_entries
        self._entries: OrderedDict[int, CachedImage] = OrderedDict()

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, key: int) -> CachedImage | None:
        entry = self._entries.get(key)
        if entry is not None:
            self._entries.move_to_end(key)
        return entry

    def put(self, key: int, entry: CachedImage) -> None:
        self._entries[key] = entry
        self._entries.move_to_end(key)
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)

    def entries_with_grid(self, grid_thw: tuple[int, int, int]) -> list[CachedImage]:
        """Encoded entries with stored pixels whose image has the given grid."""
        return [
            entry
            for entry in self._entries.values()
            if entry.grid_thw == grid_thw and entry.pixels is not None
        ]
