# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Visual feature cache: reuse vision-encoder outputs across temporal frames."""

from __future__ import annotations

from typing import Any

from ..schema import VisualCacheConfig
from .cache import CachedImage, VisualFeatureCache
from .encoder import CachedImageEncoder, EncodeStats, ReuseStrategy

__all__ = [
    "CachedImage",
    "CachedImageEncoder",
    "EncodeStats",
    "ReuseStrategy",
    "VisualFeatureCache",
    "install_visual_cache",
]


def install_visual_cache(
    qwen3vl_model: Any, cfg: VisualCacheConfig
) -> CachedImageEncoder:
    """Route a ``Qwen3VLModel``'s image encoding through a feature cache.

    Replaces the instance's ``get_image_features``, which ``forward`` calls on
    the prefill pass; the model's code and outputs are otherwise unchanged.
    """
    encoder = CachedImageEncoder(
        encode=qwen3vl_model.get_image_features,
        spatial_merge_size=qwen3vl_model.visual.spatial_merge_size,
        cache=VisualFeatureCache(cfg.max_entries),
        strategy=ReuseStrategy(cfg.strategy),
        near_duplicate_threshold=cfg.near_duplicate_threshold,
        max_near_duplicate_uses=cfg.max_near_duplicate_uses,
    )
    qwen3vl_model.get_image_features = encoder.get_image_features
    return encoder
