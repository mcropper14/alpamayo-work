# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

from __future__ import annotations

from pathlib import Path

import alpasim_wizard.setup_omegaconf  # noqa: F401
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra


def _compose_config(*overrides: str):
    config_dir = Path(__file__).parents[1] / "configs"
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        return compose(
            config_name="base_config.yaml",
            overrides=[
                "deploy=local",
                "topology=1gpu",
                "driver=alpamayo1_5",
                "wizard.log_dir=/tmp/alpasim-test",
                *overrides,
            ],
        )


def test_radar_preset_adds_radars_on_top_of_lidar_preset() -> None:
    cfg = _compose_config("+exp/sim=safety_validator_radar")

    simulation_config = cfg.runtime.simulation_config
    assert simulation_config.render_lidar_for_driver is True
    assert [radar.name for radar in simulation_config.radars] == [
        "front_long_range",
        "front_mid_range",
    ]
    assert cfg.driver.safety_validator.enabled is True


def test_weather_presets_combine_adaptive_safety_and_scene_weather() -> None:
    cfg = _compose_config(
        "+exp/sim=safety_validator_weather", "+exp/weather=dev_scenes"
    )

    simulation_config = cfg.runtime.simulation_config
    assert simulation_config.render_lidar_for_driver is True
    assert len(simulation_config.radars) == 2
    assert cfg.driver.weather.enabled is True
    assert (
        simulation_config.scene_weather["clipgt-5c1b7555-64fa-4ae4-98bf-4cf24e6de0a4"]
        == "rain"
    )


def test_scenario_labels_combine_with_weather_adaptive_safety() -> None:
    cfg = _compose_config("+exp/sim=safety_validator_weather", "+exp/labels=scenario")

    assert cfg.driver.weather.enabled is True
    assert cfg.driver.scenario.enabled is True
