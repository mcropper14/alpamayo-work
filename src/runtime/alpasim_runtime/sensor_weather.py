# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

"""Weather effects on the LiDAR and radar data forwarded to the driver.

The renderer raycasts LiDAR against the scene reconstruction, and radar is
simulated from ground-truth traffic, so neither sees the weather the recorded
cameras show.  ``simulation_config.scene_weather`` names each scene's actual
weather, and these effects degrade both sensors to match:

- Attenuation: a return at range r survives with probability exp(-2 alpha r)
  (two-way Beer-Lambert extinction).
- Backscatter (LiDAR only), within a few degrees of horizontal and at ranges
  uniform in ``lidar_clutter_range_m``: single returns from droplets or flakes
  (``lidar_noise_points``, too sparse to cluster), and dense patches
  (``lidar_clutter_patches``) that the detector mistakes for obstacles.  Rain
  makes few patches; fog and snow make many.

The coefficients are illustrative orders of magnitude from the LiDAR weather
literature, not a calibration of a particular sensor.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from alpasim_utils.lidar_payload import LidarSpin, pack_lidar_spin, unpack_lidar_spin
from alpasim_utils.radar_payload import RadarScan

_BACKSCATTER_ELEVATION_DEG = 10.0
_POINTS_PER_PATCH = 40
_PATCH_RADIUS_M = 0.4


@dataclasses.dataclass(frozen=True)
class WeatherEffects:
    lidar_extinction_per_m: float
    lidar_noise_points: int  # per spin
    lidar_clutter_patches: int  # per spin
    lidar_clutter_range_m: tuple[float, float]
    radar_extinction_per_m: float


WEATHER_EFFECTS: dict[str, WeatherEffects] = {
    "clear": WeatherEffects(0.0, 0, 0, (0.0, 0.0), 0.0),
    "rain": WeatherEffects(0.01, 1_000, 2, (3.0, 15.0), 0.0005),
    "fog": WeatherEffects(0.06, 2_000, 40, (3.0, 12.0), 0.001),
    "snow": WeatherEffects(0.03, 2_000, 20, (3.0, 12.0), 0.0008),
}


def _survives(ranges: np.ndarray, extinction: float, rng: np.random.Generator):
    return rng.random(len(ranges)) < np.exp(-2.0 * extinction * ranges)


def degrade_lidar_payload(
    payload: bytes, effects: WeatherEffects, rng: np.random.Generator
) -> bytes:
    """Attenuate a packed LiDAR spin and add backscatter returns."""
    spin = unpack_lidar_spin(payload)
    degraded = degrade_lidar(spin, effects, rng)
    return pack_lidar_spin(
        degraded.tobytes(), len(degraded), spin.timestamp_us, spin.rig_to_sensor
    )


def degrade_lidar(
    spin: LidarSpin, effects: WeatherEffects, rng: np.random.Generator
) -> np.ndarray:
    """(N', 3) float32 sensor-frame points after weather effects."""
    ranges = np.linalg.norm(spin.xyz, axis=1)
    kept = spin.xyz[_survives(ranges, effects.lidar_extinction_per_m, rng)]

    noise = _backscatter_points(
        effects, effects.lidar_noise_points, rng, uniform_in_volume=True
    )
    patch_centers = _backscatter_points(
        effects, effects.lidar_clutter_patches, rng, uniform_in_volume=False
    )
    patches = np.repeat(patch_centers, _POINTS_PER_PATCH, axis=0) + rng.normal(
        0.0, _PATCH_RADIUS_M / 2, (len(patch_centers) * _POINTS_PER_PATCH, 3)
    )
    return np.concatenate([kept, noise, patches]).astype(np.float32)


def _backscatter_points(
    effects: WeatherEffects,
    n: int,
    rng: np.random.Generator,
    uniform_in_volume: bool,
) -> np.ndarray:
    """(n, 3) sensor-frame points in the backscatter band, near horizontal.

    Uniform in volume keeps single returns equally sparse at every range;
    uniform in range concentrates patches near the sensor.
    """
    near, far = effects.lidar_clutter_range_m
    if uniform_in_volume:
        r = np.cbrt(rng.uniform(near**3, far**3, n))
    else:
        r = rng.uniform(near, far, n)
    azimuth = rng.uniform(-np.pi, np.pi, n)
    elevation = np.deg2rad(
        rng.uniform(-_BACKSCATTER_ELEVATION_DEG, _BACKSCATTER_ELEVATION_DEG, n)
    )
    return np.column_stack(
        [
            r * np.cos(elevation) * np.cos(azimuth),
            r * np.cos(elevation) * np.sin(azimuth),
            r * np.sin(elevation),
        ]
    )


def degrade_radar(
    scan: RadarScan, effects: WeatherEffects, rng: np.random.Generator
) -> RadarScan:
    """Drop radar detections by two-way attenuation."""
    keep = _survives(
        np.linalg.norm(scan.xy, axis=1), effects.radar_extinction_per_m, rng
    )
    return dataclasses.replace(
        scan,
        xy=scan.xy[keep],
        range_rate=scan.range_rate[keep],
        line_of_sight=scan.line_of_sight[keep],
    )
