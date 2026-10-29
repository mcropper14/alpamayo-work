# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Cached replacement for Qwen3-VL's ``get_image_features``.

Alpamayo feeds every camera's last ``context_length`` frames to the VLM at
each inference, and ``generate`` repeats the images once per trajectory
sample.  With 4 cameras, 4 frames and 6 samples the vision encoder sees 96
images per inference of which only 4 are new.  Qwen3-VL encodes each image
independently (attention is restricted to the image's own patches), so
per-image outputs can be reused exactly.

Strategies:

* ``EXACT``: reuse features only for bit-identical images.  Lossless.
* ``NEAR_DUPLICATE``: additionally reuse the features of the most similar
  encoded image when the mean absolute difference of the normalized pixels is
  at most ``near_duplicate_threshold``.  Lossy, with the error bounded by the
  threshold against the image the features came from.  Each encoded image
  serves at most ``max_near_duplicate_uses`` other images, which forces a
  periodic refresh even in a static scene.
"""

from __future__ import annotations

import enum
from collections.abc import Callable
from dataclasses import dataclass

import torch

from .cache import CachedImage, VisualFeatureCache

_INT_VIEW_DTYPES = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}


class ReuseStrategy(str, enum.Enum):
    EXACT = "exact"
    NEAR_DUPLICATE = "near_duplicate"


@dataclass
class EncodeStats:
    """What one ``get_image_features`` call did with its images."""

    images: int = 0  # images requested, including per-sample repeats
    unique_images: int = 0
    exact_hits: int = 0  # unique images found in the cache
    near_duplicate_hits: int = 0  # unique images served by a similar image
    encoded: int = 0  # unique images run through the vision encoder

    def as_counters(self) -> dict[str, float]:
        return {f"vision_{name}": float(value) for name, value in vars(self).items()}


class ImageFingerprinter:
    """64-bit fingerprint of an image's exact pixel bits.

    A random-weighted sum over the raw bits with wrapping int64 arithmetic:
    deterministic regardless of reduction order, and two different images
    collide with probability about 2**-64.
    """

    def __init__(self, seed: int = 0) -> None:
        self._seed = seed
        self._weights: dict[tuple[int, torch.device], torch.Tensor] = {}

    def _weights_for(self, numel: int, device: torch.device) -> torch.Tensor:
        key = (numel, device)
        if key not in self._weights:
            generator = torch.Generator().manual_seed(self._seed + numel)
            self._weights[key] = torch.randint(
                -(2**62), 2**62, (numel,), generator=generator, dtype=torch.int64
            ).to(device)
        return self._weights[key]

    def __call__(self, images: list[torch.Tensor]) -> list[int]:
        sums = []
        for pixels in images:
            bits = pixels.contiguous().view(_INT_VIEW_DTYPES[pixels.element_size()])
            bits = bits.flatten().to(torch.int64)
            sums.append((bits * self._weights_for(bits.numel(), bits.device)).sum())
        return torch.stack(sums).tolist()


ImageFeatureFn = Callable[
    [torch.Tensor, torch.Tensor], tuple[tuple[torch.Tensor, ...], list[torch.Tensor]]
]


class CachedImageEncoder:
    """Serves ``get_image_features`` from a :class:`VisualFeatureCache`.

    Only images missing from the cache go through ``encode`` (the original
    ``get_image_features``), in one batch.
    """

    def __init__(
        self,
        encode: ImageFeatureFn,
        spatial_merge_size: int,
        cache: VisualFeatureCache,
        strategy: ReuseStrategy,
        near_duplicate_threshold: float,
        max_near_duplicate_uses: int,
    ) -> None:
        self._encode = encode
        self._tokens_per_patch_group = spatial_merge_size**2
        self._cache = cache
        self._strategy = strategy
        self._near_duplicate_threshold = near_duplicate_threshold
        self._max_near_duplicate_uses = max_near_duplicate_uses
        self._fingerprint = ImageFingerprinter()
        # Called with each call's stats as flat counters (e.g. by the latency log).
        self.on_call: Callable[[dict[str, float]], None] | None = None

    def get_image_features(
        self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor
    ) -> tuple[tuple[torch.Tensor, ...], list[torch.Tensor]]:
        """Drop-in for ``Qwen3VLModel.get_image_features``.

        ``pixel_values`` holds the patches of all images back to back; image i
        has ``prod(image_grid_thw[i])`` patches.
        """
        grids = [tuple(grid) for grid in image_grid_thw.tolist()]
        images = torch.split(pixel_values, [t * h * w for t, h, w in grids])
        keys = self._fingerprint(list(images))
        stats = EncodeStats(images=len(images))

        resolved: dict[int, CachedImage] = {}
        to_encode: list[int] = []  # index of the first image of each missing key
        pending_keys: set[int] = set()
        for index, key in enumerate(keys):
            if key in resolved or key in pending_keys:
                continue
            stats.unique_images += 1
            entry = self._cache.get(key)
            if entry is not None:
                stats.exact_hits += 1
                resolved[key] = entry
                continue
            if self._strategy is ReuseStrategy.NEAR_DUPLICATE:
                entry = self._near_duplicate(grids[index], images[index])
                if entry is not None:
                    stats.near_duplicate_hits += 1
                    resolved[key] = entry
                    self._cache.put(key, entry)
                    continue
            to_encode.append(index)
            pending_keys.add(key)

        if to_encode:
            stats.encoded = len(to_encode)
            for index, entry in zip(
                to_encode, self._encode_images(images, grids, to_encode), strict=True
            ):
                resolved[keys[index]] = entry
                self._cache.put(keys[index], entry)

        if self.on_call is not None:
            self.on_call(stats.as_counters())

        ordered = [resolved[key] for key in keys]
        image_embeds = tuple(entry.embeds for entry in ordered)
        deepstack = [
            torch.cat([entry.deepstack[layer] for entry in ordered])
            for layer in range(len(ordered[0].deepstack))
        ]
        return image_embeds, deepstack

    def _encode_images(
        self,
        images: tuple[torch.Tensor, ...],
        grids: list[tuple[int, int, int]],
        indices: list[int],
    ) -> list[CachedImage]:
        grid_thw = torch.tensor(
            [grids[i] for i in indices], dtype=torch.long, device=images[0].device
        )
        embeds, deepstack = self._encode(
            torch.cat([images[i] for i in indices]), grid_thw
        )
        token_counts = [embed.shape[0] for embed in embeds]
        deepstack_per_image = [torch.split(layer, token_counts) for layer in deepstack]
        keep_pixels = self._strategy is ReuseStrategy.NEAR_DUPLICATE
        # Clone so cached entries do not keep the whole batch tensors alive.
        return [
            CachedImage(
                grid_thw=grids[index],
                embeds=embeds[position].clone(),
                deepstack=[layer[position].clone() for layer in deepstack_per_image],
                pixels=images[index].clone() if keep_pixels else None,
            )
            for position, index in enumerate(indices)
        ]

    def _near_duplicate(
        self, grid: tuple[int, int, int], pixels: torch.Tensor
    ) -> CachedImage | None:
        """Reuse entry for the most similar encoded image within the threshold."""
        candidates = [
            entry
            for entry in self._cache.entries_with_grid(grid)
            if entry.near_duplicate_uses < self._max_near_duplicate_uses
        ]
        if not candidates:
            return None
        distances = torch.stack(
            [
                (entry.pixels - pixels).abs().float().mean()  # type: ignore[operator]
                for entry in candidates
            ]
        )
        best = int(distances.argmin())
        if distances[best].item() > self._near_duplicate_threshold:
            return None
        source = candidates[best]
        source.near_duplicate_uses += 1
        return CachedImage(
            grid_thw=grid,
            embeds=source.embeds,
            deepstack=source.deepstack,
            pixels=None,
        )
