# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import pytest
import torch

from ..visual_cache import (
    CachedImageEncoder,
    EncodeStats,
    ReuseStrategy,
    VisualFeatureCache,
)

GRID = (1, 4, 4)  # 16 patches -> 4 merged tokens with spatial_merge_size 2
PATCH_DIM = 6
HIDDEN = 3


class _PerImageEncoder:
    """Stands in for Qwen3-VL: encodes every image independently."""

    def __init__(self) -> None:
        self.weight = torch.randn(4 * PATCH_DIM, HIDDEN, generator=_gen(1))
        self.encoded_images: list[int] = []

    def __call__(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor):
        counts = grid_thw.prod(-1).tolist()
        self.encoded_images.append(len(counts))
        embeds = tuple(
            image.reshape(-1, 4 * PATCH_DIM) @ self.weight
            for image in torch.split(pixel_values, counts)
        )
        merged = torch.cat(embeds)
        return embeds, [merged * 2, merged * 3]


def _gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _frames(n: int, seed: int = 0) -> list[torch.Tensor]:
    return [torch.randn(16, PATCH_DIM, generator=_gen(seed + i)) for i in range(n)]


def _inputs(frames: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.cat(frames), torch.tensor([GRID] * len(frames))


def _encoder(
    strategy: ReuseStrategy = ReuseStrategy.EXACT, max_entries: int = 64
) -> tuple[CachedImageEncoder, _PerImageEncoder, list[EncodeStats]]:
    base = _PerImageEncoder()
    encoder = CachedImageEncoder(
        encode=base,
        spatial_merge_size=2,
        cache=VisualFeatureCache(max_entries),
        strategy=strategy,
        near_duplicate_threshold=0.05,
        max_near_duplicate_uses=2,
    )
    calls: list[EncodeStats] = []
    encoder.on_call = lambda counters: calls.append(
        EncodeStats(**{k.removeprefix("vision_"): int(v) for k, v in counters.items()})
    )
    return encoder, base, calls


def _assert_same_features(actual, expected) -> None:
    actual_embeds, actual_deepstack = actual
    expected_embeds, expected_deepstack = expected
    assert len(actual_embeds) == len(expected_embeds)
    for a, e in zip(actual_embeds, expected_embeds, strict=True):
        torch.testing.assert_close(a, e)
    for a, e in zip(actual_deepstack, expected_deepstack, strict=True):
        torch.testing.assert_close(a, e)


def test_exact_cache_matches_uncached_encoder_over_a_sliding_window() -> None:
    encoder, base, calls = _encoder()
    frames = _frames(5)
    # Step 1: frames 0-3; step 2: frames 1-4. Each repeated for 3 samples.
    for window in (frames[0:4], frames[1:5]):
        repeated = window * 3
        cached = encoder.get_image_features(*_inputs(repeated))
        _assert_same_features(
            cached, _PerImageEncoder.__call__(base, *_inputs(repeated))
        )

    assert calls == [
        EncodeStats(images=12, unique_images=4, exact_hits=0, encoded=4),
        EncodeStats(images=12, unique_images=4, exact_hits=3, encoded=1),
    ]


def test_only_missing_images_reach_the_encoder() -> None:
    encoder, base, _ = _encoder()
    frames = _frames(4)

    encoder.get_image_features(*_inputs(frames))
    encoder.get_image_features(*_inputs(frames))

    assert base.encoded_images == [4]


def test_lru_eviction_forces_re_encoding() -> None:
    encoder, base, _ = _encoder(max_entries=2)
    a, b, c = _frames(3)

    for frame in (a, b, c, a):
        encoder.get_image_features(*_inputs([frame]))

    assert base.encoded_images == [1, 1, 1, 1]


def test_near_duplicate_reuses_similar_frames_until_the_reuse_limit() -> None:
    encoder, base, calls = _encoder(ReuseStrategy.NEAR_DUPLICATE)
    (frame,) = _frames(1)
    drifting = [frame + 0.01 * step for step in range(4)]

    outputs = [encoder.get_image_features(*_inputs([f])) for f in drifting]

    # Encoded, reused twice (max_near_duplicate_uses=2), then encoded again.
    assert [stats.encoded for stats in calls] == [1, 0, 0, 1]
    assert [stats.near_duplicate_hits for stats in calls] == [0, 1, 1, 0]
    torch.testing.assert_close(outputs[2][0][0], outputs[0][0][0])
    assert base.encoded_images == [1, 1]


def test_near_duplicate_encodes_frames_beyond_the_threshold() -> None:
    encoder, _, calls = _encoder(ReuseStrategy.NEAR_DUPLICATE)
    (frame,) = _frames(1)

    encoder.get_image_features(*_inputs([frame]))
    encoder.get_image_features(*_inputs([frame + 1.0]))

    assert [stats.encoded for stats in calls] == [1, 1]


def test_cache_rejects_non_positive_size() -> None:
    with pytest.raises(ValueError, match="max_entries"):
        VisualFeatureCache(0)
