"""Every crashed rollout in a replay: who the ego hit, from where, and what the
safety filter's counterfactual made of it per sensor.

Usage (repo root, after benchmark/replay_all.sh NAME):
    uv run python benchmark/crash_table.py validator_replay_v13
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import numpy as np
import polars as pl
from alpasim_utils.geometry import pose_from_grpc

sys.path.insert(0, str(Path(__file__).parent))
import replay_validator as rv  # noqa: E402

SENSORS = ("gt", "radar", "lidar", "fused")
SETS = [  # model, scene set, runs dir, arm, replay directory suffix
    ("Alpamayo 1.5", "braking car", Path("benchmark/runs_smoke"), "alpamayo1_5_leadbrake", ""),
    ("Alpamayo 1.5", "core", Path("benchmark/runs_core"), "alpamayo1_5", ""),
    ("Alpamayo 1.5", "dev", Path("benchmark/runs"), "alpamayo1_5", "_alpamayo1_5"),
    ("SimForge-D1.5", "dev", Path("benchmark/runs"), "simforge_d1_5", "_simforge_d1_5"),
    ("VaVAM", "dev", Path("benchmark/runs"), "vavam", "_vavam"),
]
SHORT = {
    "avoided": "avoided",
    "only hit from behind": "avoided*",
    "same actor": "same crash",
    "different actor": "other crash",
    "new collision": "new crash",
}


def _geometry(asl_path: Path, actor: str, collision_s: float) -> dict:
    """The other actor's place relative to the ego 1 s before the crash and at it."""
    log = asyncio.run(rv._read_log(asl_path))
    ego = log.actor_poses["EGO"]
    t0 = min(ego)
    out = {"label": log.labels[actor]}
    for name, dt_s in (("1s_before", -1.0), ("at_crash", 0.0)):
        t = min(ego, key=lambda u: abs(u - (t0 + (collision_s + dt_s) * 1e6)))
        poses = log.actor_poses[actor]
        ta = min(poses, key=lambda u: abs(u - t))
        rel = np.linalg.inv(pose_from_grpc(ego[t]).as_se3()) @ pose_from_grpc(poses[ta]).as_se3()
        out[name] = f"x={rel[0, 3]:+.1f} y={rel[1, 3]:+.1f}"
    times = sorted(ego)
    t = min(times, key=lambda u: abs(u - (t0 + collision_s * 1e6)))
    k = times.index(t)
    a, b = pose_from_grpc(ego[times[k - 1]]).as_se3(), pose_from_grpc(ego[t]).as_se3()
    out["ego_mps"] = round(float(np.linalg.norm(b[:2, 3] - a[:2, 3]) / ((t - times[k - 1]) * 1e-6)), 1)
    return out


def main() -> None:
    name = sys.argv[1]
    rows = []
    for model, scenes, runs_dir, arm, suffix in SETS:
        replay = runs_dir / f"{name}{suffix}"
        cf = {s: pl.read_csv(replay / f"counterfactual_{s}.csv") for s in SENSORS}
        asl = {p.parent.name: p for p in runs_dir.glob(f"{arm}*/rollouts/*/*/rollout.asl")}
        crashed = cf["fused"].filter(pl.col("logged_collision_s").is_not_null())
        for r in crashed.iter_rows(named=True):
            row = {
                "model": model, "scenes": scenes, "scene": r["scene"][7:15],
                "rollout": r["rollout"][:8], "eval": r["eval_collision"],
                "at_fault": r["at_fault"], "crash_s": round(r["logged_collision_s"], 1),
                **_geometry(asl[r["rollout"]], str(r["logged_actor"]), r["logged_collision_s"]),
            }
            for s in SENSORS:
                o = cf[s].filter(pl.col("rollout") == r["rollout"]).row(0, named=True)
                row[s] = SHORT[o["outcome"]]
            rows.append(row)
    table = pl.DataFrame(rows).sort("model", "scenes", "at_fault", "scene", descending=[False, False, True, False])
    out = Path(f"benchmark/crash_table_{name}.csv")
    table.write_csv(out)
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=250, fmt_str_lengths=20):
        print(table)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
