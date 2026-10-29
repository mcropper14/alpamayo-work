# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation

import json
from pathlib import Path

import pytest

from ..models import alpamayo1_5_model


def _checkpoint(tmp_path: Path, vlm_name_or_path: str) -> Path:
    (tmp_path / "config.json").write_text(
        json.dumps({"vlm_name_or_path": vlm_name_or_path})
    )
    return tmp_path


@pytest.fixture(autouse=True)
def _plain_config(monkeypatch: pytest.MonkeyPatch) -> None:
    # The real config builds the VLM processor; only the resolved path matters.
    monkeypatch.setattr(alpamayo1_5_model, "Alpamayo1_5Config", lambda **kw: kw)


def test_bundled_vlm_folder_resolves_inside_the_checkpoint(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path, "vlm")
    (checkpoint / "vlm").mkdir()
    (checkpoint / "vlm" / "config.json").write_text("{}")

    config = alpamayo1_5_model.load_checkpoint_config(str(checkpoint), "sdpa")

    assert config["vlm_name_or_path"] == str(checkpoint / "vlm")


def test_hub_vlm_id_is_kept(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path, "nvidia/Cosmos-Reason2-8B")

    config = alpamayo1_5_model.load_checkpoint_config(str(checkpoint), "sdpa")

    assert config["vlm_name_or_path"] == "nvidia/Cosmos-Reason2-8B"


def test_attn_implementation_overrides_the_checkpoint(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path, "nvidia/Cosmos-Reason2-8B")

    config = alpamayo1_5_model.load_checkpoint_config(str(checkpoint), "sdpa")

    assert config["attn_implementation"] == "sdpa"
