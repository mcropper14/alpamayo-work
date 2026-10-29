"""
visualise.py
────────────────────────────────────────────────────────────────────────────
Generates a self-contained interactive HTML viewer with Alpamayo camera
feeds side-by-side with the LiDAR bird's-eye view and physics analysis.

Camera data is loaded from the nvidia/PhysicalAI-Autonomous-Vehicles
HuggingFace dataset exactly the same way as LiDAR data — no AlpaSim
required.

Layout
──────
  ┌─────────────────────┬──────────────────────┬──────────────────┐
  │  FRONT WIDE 120°    │   BEV LiDAR          │  Physics status  │
  │  (primary camera)   │   point cloud        │  constraints     │
  │  + reasoning trace  │   detections         │  objects         │
  ├──────────┬──────────┤   ego corridor       │  metrics         │
  │ FRONT-L  │ FRONT-R  │                      │                  │
  ├──────────┴──────────┴──────────────────────┴──────────────────┤
  │  Timeline scrubber                                             │
  └────────────────────────────────────────────────────────────────┘

Usage — HuggingFace dataset (cameras loaded automatically)
──────────────────────────────────────────────────────────
  export HF_TOKEN=hf_...

  # Loads LiDAR + front 3 cameras from the dataset
  python visualise.py \\
    --clip-id 25cd4769-5dcf-4b53-a351-bf2c5deb6124 \\
    --output viewer.html

  # Specify chunk explicitly (skips clip_index lookup)
  python visualise.py \\
    --clip-id 25cd4769-5dcf-4b53-a351-bf2c5deb6124 \\
    --chunk 0 \\
    --output viewer.html

  # Only load specific cameras (faster — skips unrequested zips)
  python visualise.py \\
    --clip-id 25cd4769-... \\
    --cameras camera_front_wide_120fov camera_front_left_120fov \\
    --output viewer.html

  # Skip cameras entirely
  python visualise.py \\
    --clip-id 25cd4769-... \\
    --no-cameras \\
    --output viewer.html

Usage — local files (e.g. extracted from AlpaSim or dataset cache)
───────────────────────────────────────────────────────────────────
  python visualise.py \\
    --lidar-path path/to/lidar_top_360fov.parquet \\
    --camera-front-wide  path/to/camera_front_wide_120fov.mp4 \\
    --camera-front-left  path/to/camera_front_left_120fov.mp4 \\
    --camera-front-right path/to/camera_front_right_120fov.mp4 \\
    --output viewer.html

Usage — synthetic LiDAR (no download)
──────────────────────────────────────
  python visualise.py --scenario emergency_stop --output viewer.html

Requires
────────
  pip install opencv-python          (for camera frame extraction)
  pip install huggingface_hub hf_xet (for HuggingFace downloads)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from physical_ai_av_lidar import (
    LidarLoader, LidarPipeline, PipelineConfig,
    PreprocessConfig, DetectionConfig, PhysicsConfig,
    CAMERA_NAMES, CAMERA_LABELS,
)
from physical_ai_av_lidar.camera_loader import (
    CameraLoader, MultiCameraLoader,
    _blank_b64, _extract_clip_video,
)

# Cameras shown in the viewer [primary, secondary-left, secondary-right]
VIEWER_CAMERAS = [
    "camera_front_wide_120fov",
    "camera_front_left_120fov",
    "camera_front_right_120fov",
]

# Default: only load front wide (one zip download instead of three)
DEFAULT_CAMERAS = ["camera_front_wide_120fov"]

# ─────────────────────────────────────────────────────────────────────────────
# Reasoning trace
# ─────────────────────────────────────────────────────────────────────────────

def load_reasoning_trace(path: str) -> Dict[int, dict]:
    """
    Load AlpaSim chain-of-causation JSONL.
    Returns {spin_index: {"text": str, "action": str}}.
    """
    import json as _json
    traces = {}
    if not os.path.exists(path):
        return traces
    with open(path) as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj  = _json.loads(line)
                spin = int(obj.get("spin", obj.get("scan_index", line_num)))
                traces[spin] = {
                    "text":   obj.get("reasoning", obj.get("chain_of_causation", "")),
                    "action": obj.get("action",    obj.get("alpamayo_action", "")),
                }
            except (json.JSONDecodeError, ValueError):
                continue
    return traces


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline data collection
# ─────────────────────────────────────────────────────────────────────────────

def collect_frames(pipe: LidarPipeline, ego_speed_ms: float, max_spins: int) -> List[dict]:
    loader   = pipe._loader
    preproc  = pipe._preprocessor
    detector = pipe._detector
    tracker  = pipe._tracker
    reasoner = pipe._reasoner

    prev_speed = ego_speed_ms
    prev_ts    = None
    frames     = []

    for spin in loader.iter_spins(stop=max_spins):
        processed = preproc.process(spin)
        det_frame = detector.detect(processed)
        det_frame = tracker.update(det_frame)

        dt = 1.0 / 10.0
        if spin.ego_pose is not None and prev_ts is not None:
            dt = max((spin.timestamp_us - prev_ts) / 1e6, 0.01)

        report = reasoner.reason(
            detection_frame        = det_frame,
            ego_speed_ms           = ego_speed_ms,
            lateral_accel_ms2      = 0.0,
            longitudinal_accel_ms2 = 0.0,
            yaw_rate_rads          = 0.0,
            prev_ego_speed         = prev_speed,
            dt_s                   = dt,
        )

        # Subsample points for JSON
        xyz_obs = processed.xyz_obstacle
        if len(xyz_obs) > 3500:
            xyz_obs = xyz_obs[np.random.choice(len(xyz_obs), 3500, replace=False)]
        xyz_gnd = processed.xyz_ground
        if len(xyz_gnd) > 1500:
            xyz_gnd = xyz_gnd[np.random.choice(len(xyz_gnd), 1500, replace=False)]

        objects = []
        for obj in det_frame.objects:
            cx, cy, _ = obj.bbox.center.tolist()
            ex, ey, _ = obj.bbox.extent.tolist()
            ttc = obj.ttc_s
            objects.append({
                "id": obj.object_id, "cls": obj.cls.value,
                "cx": cx, "cy": cy, "ex": ex, "ey": ey,
                "range_m": round(obj.range_m, 2),
                "ttc_s":   round(ttc, 2) if ttc is not None and not math.isinf(ttc) else None,
                "conf":    round(obj.confidence, 3),
            })

        constraints = [{
            "name":     c.name,
            "status":   c.status.value,
            "measured": round(float(c.measured), 3),
            "limit":    round(float(c.limit), 3),
            "margin":   round(float(c.margin), 3),
        } for c in report.constraints]

        frames.append({
            "spin":            spin.scan_index,
            "timestamp_us":    spin.timestamp_us,
            "status":          report.overall_status,
            "action":          report.recommended_action.value,
            "nearest_m":       round(report.nearest_range_m, 2),
            "nearest_ttc":     round(report.nearest_ttc_s, 2) if not math.isinf(report.nearest_ttc_s) else 999,
            "n_objects":       len(det_frame.objects),
            "n_obstacle_pts":  len(processed.xyz_obstacle),
            "pts_obs":         xyz_obs[:, :2].tolist(),
            "pts_gnd":         xyz_gnd[:, :2].tolist(),
            "objects":         objects,
            "constraints":     constraints,
            "cam_front":       None,
            "cam_left":        None,
            "cam_right":       None,
            "reasoning":       None,
            "alpamayo_action": None,
        })

        prev_speed = ego_speed_ms
        prev_ts    = spin.timestamp_us

        sym = {"PASS": "✓", "WARN": "⚠", "CRITICAL": "✗"}.get(report.overall_status, "?")
        print(f"  Spin {spin.scan_index:3d} {sym}  "
              f"nearest {report.nearest_range_m:6.1f}m  "
              f"{report.recommended_action.value}")

    return frames


# ─────────────────────────────────────────────────────────────────────────────
# Camera injection
# ─────────────────────────────────────────────────────────────────────────────

def inject_cameras_from_hf(
    frames:    List[dict],
    clip_uuid: str,
    chunk_id:  Optional[int],
    cameras:   List[str],
    cache_dir: str,
    token:     Optional[str],
    width: int, height: int, quality: int,
) -> None:
    """
    Download camera zips from HuggingFace and inject base64 frames
    into the frames list in-place.
    """
    n = len(frames)

    # Collect LiDAR spin timestamps for frame alignment
    timestamps_us = [f["timestamp_us"] for f in frames]

    # Resolve chunk_id once up front (shared with LiDAR)
    from physical_ai_av_lidar.loader import find_chunk_for_clip
    if chunk_id is None:
        chunk_id = find_chunk_for_clip(clip_uuid, cache_dir=cache_dir, token=token)

    # Map camera names to frame slot keys
    cam_to_slot = {
        "camera_front_wide_120fov":  "cam_front",
        "camera_front_left_120fov":  "cam_left",
        "camera_front_right_120fov": "cam_right",
    }

    for cam in cameras:
        slot = cam_to_slot.get(cam)
        if slot is None:
            continue  # camera not shown in 3-slot viewer layout (yet)

        label = CAMERA_LABELS.get(cam, cam)
        print(f"  Camera {label}: downloading from HuggingFace...")

        try:
            loader = CameraLoader.from_hf(
                clip_uuid = clip_uuid,
                camera    = cam,
                chunk_id  = chunk_id,
                cache_dir = cache_dir,
                token     = token,
            )
            print(f"  Camera {label}: extracting {n} frames...")
            b64_frames = loader.extract_frames_b64(
                timestamps_us = timestamps_us,
                width         = width,
                height        = height,
                quality       = quality,
            )
            for i, b64 in enumerate(b64_frames):
                if i < n:
                    frames[i][slot] = b64
            print(f"  Camera {label}: OK")

        except FileNotFoundError as e:
            print(f"  Camera {label}: not available for this clip — {e}")
        except Exception as e:
            print(f"  Camera {label}: FAILED — {e}")


def inject_cameras_from_local(
    frames:      List[dict],
    cam_paths:   Dict[str, str],   # {camera_name: local_path}
    width: int, height: int, quality: int,
) -> None:
    """
    Inject frames from local mp4 files into the frames list in-place.
    """
    import glob
    n = len(frames)
    timestamps_us = [f["timestamp_us"] for f in frames]

    cam_to_slot = {
        "camera_front_wide_120fov":  "cam_front",
        "camera_front_left_120fov":  "cam_left",
        "camera_front_right_120fov": "cam_right",
    }

    for cam, path in cam_paths.items():
        slot = cam_to_slot.get(cam)
        if slot is None:
            continue
        label = CAMERA_LABELS.get(cam, cam)

        try:
            loader = CameraLoader.from_file(path, camera_name=cam)
            print(f"  Camera {label}: extracting {n} frames from {Path(path).name}...")
            b64_frames = loader.extract_frames_b64(
                timestamps_us = timestamps_us,
                width         = width,
                height        = height,
                quality       = quality,
            )
            for i, b64 in enumerate(b64_frames):
                if i < n:
                    frames[i][slot] = b64
            print(f"  Camera {label}: OK")
        except Exception as e:
            print(f"  Camera {label}: FAILED — {e}")


# ─────────────────────────────────────────────────────────────────────────────
# HTML template
# ─────────────────────────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Alpamayo · LiDAR Viewer</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:ital,wght@0,300;0,400;0,600;1,400&family=Barlow+Condensed:wght@400;600;700&display=swap');
:root{
  --bg:#070a0c;--s:#0c1014;--s2:#101519;--bd:#18232c;--dim:#233040;
  --tx:#5a7888;--hi:#b8cdd8;--cy:#00d4b8;--am:#f5a800;--rd:#f03050;
  --gr:#30e888;--bl:#3a7fff;
  --mono:'IBM Plex Mono',monospace;--disp:'Barlow Condensed',sans-serif;
}
*{box-sizing:border-box;margin:0;padding:0;}
body{
  background:var(--bg);color:var(--tx);font-family:var(--mono);font-size:11px;
  width:100vw;height:100vh;overflow:hidden;
  display:grid;
  grid-template-rows:42px 1fr 56px;
  grid-template-columns:340px 1fr 256px;
  grid-template-areas:"hdr hdr hdr""cam bev sid""tl tl tl";
}
body::after{content:'';position:fixed;inset:0;pointer-events:none;z-index:9999;
  background:repeating-linear-gradient(0deg,transparent 0,transparent 3px,
  rgba(0,212,184,.004) 3px,rgba(0,212,184,.004) 4px);}
/* HEADER */
header{grid-area:hdr;display:flex;align-items:center;justify-content:space-between;
  padding:0 16px;border-bottom:1px solid var(--bd);background:var(--s);}
.brand{display:flex;align-items:baseline;gap:10px;}
.brand h1{font-family:var(--disp);font-size:19px;font-weight:700;letter-spacing:.1em;
  color:var(--cy);text-shadow:0 0 28px rgba(0,212,184,.3);}
.brand sub{font-size:8px;letter-spacing:.2em;text-transform:uppercase;color:var(--dim);}
.hstats{display:flex;border:1px solid var(--bd);border-radius:3px;overflow:hidden;}
.hs{padding:4px 14px;border-right:1px solid var(--bd);display:flex;flex-direction:column;align-items:center;}
.hs:last-child{border-right:none;}
.hs .l{font-size:7px;letter-spacing:.15em;text-transform:uppercase;color:var(--dim);}
.hs .v{font-size:13px;font-weight:600;color:var(--hi);margin-top:1px;}
.hchips{display:flex;gap:6px;align-items:center;}
.chip{font-size:8px;padding:3px 8px;border-radius:2px;border:1px solid var(--dim);
  color:var(--tx);letter-spacing:.08em;}
.chip.on{border-color:var(--cy);color:var(--cy);}
/* CAMERA PANEL */
#cam{grid-area:cam;border-right:1px solid var(--bd);background:var(--s);
  display:flex;flex-direction:column;overflow:hidden;}
.cam-main{flex:1;position:relative;overflow:hidden;border-bottom:1px solid var(--bd);}
.cam-main img{width:100%;height:100%;object-fit:cover;display:block;}
.cam-row{display:flex;height:118px;}
.cam-slot{flex:1;position:relative;overflow:hidden;border-right:1px solid var(--bd);}
.cam-slot:last-child{border-right:none;}
.cam-slot img{width:100%;height:100%;object-fit:cover;display:block;}
.clabel{position:absolute;bottom:5px;left:6px;font-size:7px;letter-spacing:.14em;
  text-transform:uppercase;color:rgba(184,208,220,.55);
  background:rgba(7,10,12,.72);padding:2px 5px;border-radius:1px;pointer-events:none;}
.no-cam{width:100%;height:100%;display:flex;flex-direction:column;
  align-items:center;justify-content:center;background:var(--s2);gap:5px;}
.no-cam span{font-size:8px;color:var(--dim);letter-spacing:.1em;text-transform:uppercase;}
/* Reasoning overlay */
#rov{position:absolute;bottom:0;left:0;right:0;
  background:linear-gradient(transparent,rgba(7,10,12,.9));
  padding:20px 10px 8px;pointer-events:none;display:none;}
.rov-action{font-family:var(--disp);font-size:14px;font-weight:700;
  letter-spacing:.08em;text-transform:uppercase;margin-bottom:3px;}
.match{color:var(--gr);}.mismatch{color:var(--rd);}.unk{color:var(--am);}
.rov-text{font-size:8px;line-height:1.55;color:rgba(184,208,220,.72);
  font-style:italic;max-height:44px;overflow:hidden;}
/* BEV */
#bev-wrap{grid-area:bev;position:relative;overflow:hidden;background:var(--bg);}
canvas{width:100%;height:100%;display:block;}
.bov{position:absolute;top:9px;left:13px;font-size:7px;letter-spacing:.16em;
  text-transform:uppercase;color:var(--dim);}
.legend{position:absolute;bottom:9px;left:13px;display:flex;flex-wrap:wrap;gap:9px;}
.li{display:flex;align-items:center;gap:4px;font-size:7px;color:var(--tx);}
.ld{width:6px;height:6px;border-radius:50%;}
/* SIDEBAR */
aside{grid-area:sid;border-left:1px solid var(--bd);background:var(--s);
  overflow-y:auto;padding:11px;display:flex;flex-direction:column;gap:11px;
  scrollbar-width:thin;scrollbar-color:var(--dim) transparent;}
.stitle{font-size:7px;letter-spacing:.18em;text-transform:uppercase;color:var(--dim);margin-bottom:5px;}
.sbanner{padding:8px 11px;border-radius:3px;border:1px solid;
  display:flex;justify-content:space-between;align-items:center;}
.sbanner.PASS{border-color:var(--gr);background:rgba(48,232,136,.04);}
.sbanner.WARN{border-color:var(--am);background:rgba(245,168,0,.04);}
.sbanner.CRITICAL{border-color:var(--rd);background:rgba(240,48,80,.05);}
.sword{font-family:var(--disp);font-size:17px;font-weight:700;letter-spacing:.06em;}
.PASS .sword{color:var(--gr);}.WARN .sword{color:var(--am);}.CRITICAL .sword{color:var(--rd);}
.apill{font-size:7px;padding:3px 6px;border-radius:2px;border:1px solid var(--dim);
  color:var(--hi);letter-spacing:.06em;text-align:right;word-break:break-word;max-width:105px;}
.mets{display:grid;grid-template-columns:1fr 1fr;gap:5px;}
.met{background:var(--bg);border:1px solid var(--bd);border-radius:2px;padding:6px 8px;}
.met .l{font-size:6px;text-transform:uppercase;letter-spacing:.1em;color:var(--dim);}
.met .v{font-size:14px;font-weight:600;color:var(--hi);margin-top:2px;}
.cst{margin-bottom:8px;}
.csth{display:flex;justify-content:space-between;margin-bottom:2px;}
.cstn{font-size:8px;color:var(--hi);}
.cstv{font-size:7px;color:var(--tx);}
.cbar{height:3px;background:var(--bd);border-radius:2px;overflow:hidden;}
.cfill{height:100%;border-radius:2px;transition:width .1s;}
.PASS_f{background:var(--gr);}.WARN_f{background:var(--am);}.CRITICAL_f{background:var(--rd);}
.csts{font-size:6px;text-align:right;margin-top:2px;letter-spacing:.1em;}
.csts.PASS{color:var(--gr);}.csts.WARN{color:var(--am);}.csts.CRITICAL{color:var(--rd);}
.orow{display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid var(--bd);}
.ocls{color:var(--cy);font-size:8px;flex:1;}
.odst{color:var(--hi);font-size:8px;}
.ottc{color:var(--am);font-size:8px;min-width:34px;text-align:right;}
/* TIMELINE */
#tl{grid-area:tl;border-top:1px solid var(--bd);background:var(--s);
  padding:7px 14px;display:flex;flex-direction:column;gap:5px;}
.tlc{display:flex;align-items:center;gap:9px;}
button{background:var(--bd);border:1px solid var(--dim);color:var(--hi);
  font-family:var(--mono);font-size:10px;padding:3px 8px;border-radius:2px;cursor:pointer;}
button:hover{background:var(--dim);}
button.on{background:var(--cy);color:var(--bg);border-color:var(--cy);}
#si{font-size:10px;color:var(--hi);}
#ti{font-size:8px;color:var(--dim);}
.track{height:20px;background:var(--bg);border:1px solid var(--bd);border-radius:2px;
  display:flex;gap:1px;padding:2px;cursor:pointer;overflow:hidden;}
.tbar{flex:1;border-radius:1px;min-width:1px;opacity:.55;cursor:pointer;transition:opacity .07s;}
.tbar:hover{opacity:1;}
.tbar.PASS{background:var(--gr);}.tbar.WARN{background:var(--am);}.tbar.CRITICAL{background:var(--rd);}
.tbar.sel{opacity:1;outline:1px solid #fff;}
</style>
</head>
<body>
<header>
  <div class="brand"><h1>ALPAMAYO SAFETY ENFORCER</h1><sub>LiDAR · Camera · Physics Viewer</sub></div>
  <div class="hstats">
    <div class="hs"><span class="l">Clip</span><span class="v" id="hclip" style="font-size:9px">—</span></div>
    <div class="hs"><span class="l">Spins</span><span class="v" id="hspins">—</span></div>
    <div class="hs"><span class="l">Critical</span><span class="v" style="color:var(--rd)" id="hcrit">—</span></div>
    <div class="hs"><span class="l">Warn</span><span class="v" style="color:var(--am)" id="hwarn">—</span></div>
  </div>
  <div class="hchips">
    <span class="chip" id="chip-cam">NO CAMERAS</span>
    <span class="chip" id="chip-tr">NO TRACE</span>
    <span class="chip on">OFFLINE REPLAY</span>
  </div>
</header>

<div id="cam">
  <div class="cam-main">
    <div class="no-cam" id="noc">
      <span>No camera data</span>
      <span style="font-size:7px;color:var(--bd)">omit --no-cameras to load from HF</span>
    </div>
    <img id="if" style="display:none" alt="front">
    <div class="clabel">FRONT WIDE 120°</div>
    <div id="rov">
      <div class="rov-action unk" id="ra">—</div>
      <div class="rov-text" id="rt"></div>
    </div>
  </div>
  <div class="cam-row">
    <div class="cam-slot">
      <div class="no-cam"><span>—</span></div>
      <img id="il" style="display:none" alt="left">
      <div class="clabel">FRONT LEFT</div>
    </div>
    <div class="cam-slot">
      <div class="no-cam"><span>—</span></div>
      <img id="ir" style="display:none" alt="right">
      <div class="clabel">FRONT RIGHT</div>
    </div>
  </div>
</div>

<div id="bev-wrap">
  <canvas id="bev"></canvas>
  <div class="bov">BEV · EGO FRAME · X=FWD Y=LEFT</div>
  <div class="legend">
    <div class="li"><div class="ld" style="background:var(--bl)"></div>Ground</div>
    <div class="li"><div class="ld" style="background:var(--cy)"></div>Obstacle pts</div>
    <div class="li"><div class="ld" style="background:var(--rd)"></div>Vehicle</div>
    <div class="li"><div class="ld" style="background:var(--am)"></div>Pedestrian</div>
    <div class="li"><div class="ld" style="background:var(--gr)"></div>Cyclist</div>
    <div class="li"><div class="ld" style="background:#445"></div>Static</div>
  </div>
</div>

<aside>
  <div>
    <div class="stitle">Physics Status</div>
    <div class="sbanner PASS" id="sb">
      <span class="sword" id="sw">PASS</span>
      <span class="apill" id="ap">—</span>
    </div>
  </div>
  <div>
    <div class="stitle">Metrics</div>
    <div class="mets">
      <div class="met"><div class="l">Nearest</div><div class="v" id="mn">—</div></div>
      <div class="met"><div class="l">TTC</div><div class="v" id="mt">—</div></div>
      <div class="met"><div class="l">Objects</div><div class="v" id="mo">—</div></div>
      <div class="met"><div class="l">Obs Pts</div><div class="v" id="mp">—</div></div>
    </div>
  </div>
  <div><div class="stitle">Constraints</div><div id="clist"></div></div>
  <div><div class="stitle">Detections</div><div id="olist"></div></div>
</aside>

<div id="tl">
  <div class="tlc">
    <button id="bp">◀</button>
    <button id="bpl">▶</button>
    <button id="bn">▶▶</button>
    <span id="si">Spin 0</span>
    <span id="ti"></span>
  </div>
  <div class="track" id="track"></div>
</div>

<script>
const F=__FRAMES_JSON__;
const M=__META_JSON__;
const HAS_CAM=F.some(f=>f.cam_front!==null);
const HAS_TR=F.some(f=>f.reasoning!==null);
let cur=0,playing=false,tmr=null;

// Canvas
const cv=document.getElementById('bev'),ctx=cv.getContext('2d');
function rsz(){const w=document.getElementById('bev-wrap');cv.width=w.clientWidth;cv.height=w.clientHeight;bev(cur);}
window.addEventListener('resize',rsz);
const V={xMin:-8,xMax:55,yMin:-25,yMax:25};
function tc(x,y){return[cv.width*(-y-V.yMin)/(V.yMax-V.yMin),cv.height*(1-(x-V.xMin)/(V.xMax-V.xMin))];}
const CC={VEHICLE:'#f03050',PEDESTRIAN:'#f5a800',CYCLIST:'#30e888',STATIC:'#334455',UNKNOWN:'#2a3a4a'};
function bev(i){
  const f=F[i];ctx.clearRect(0,0,cv.width,cv.height);
  ctx.strokeStyle='#111a22';ctx.lineWidth=.5;
  for(let x=0;x<=60;x+=10){const[,cy]=tc(x,0);ctx.beginPath();ctx.moveTo(0,cy);ctx.lineTo(cv.width,cy);ctx.stroke();ctx.fillStyle='#1e2e3a';ctx.font='7px IBM Plex Mono,monospace';ctx.fillText(x+'m',4,cy-2);}
  for(let y=-24;y<=24;y+=4){const[cx]=tc(0,y);ctx.beginPath();ctx.moveTo(cx,0);ctx.lineTo(cx,cv.height);ctx.stroke();}
  const isCrit=f.constraints.some(c=>c.name==='PATH_CORRIDOR'&&c.status==='CRITICAL');
  const isWarn=f.constraints.some(c=>c.name==='PATH_CORRIDOR'&&c.status==='WARN');
  const col=isCrit?'rgba(240,48,80,':isWarn?'rgba(245,168,0,':'rgba(0,212,184,';
  const[cx1,cy1]=tc(0,2),[cx2,cy2]=tc(40,-2);
  const bx=Math.min(cx1,cx2),by=Math.min(cy1,cy2),bw=Math.abs(cx2-cx1),bh=Math.abs(cy2-cy1);
  ctx.fillStyle=col+'0.05)';ctx.strokeStyle=col+'0.22)';ctx.lineWidth=1;
  ctx.fillRect(bx,by,bw,bh);ctx.strokeRect(bx,by,bw,bh);
  ctx.fillStyle='rgba(58,127,255,0.18)';
  for(const[x,y] of f.pts_gnd){const[cx,cy]=tc(x,y);ctx.fillRect(cx-.5,cy-.5,1,1);}
  ctx.fillStyle='rgba(0,212,184,0.48)';
  for(const[x,y] of f.pts_obs){const[cx,cy]=tc(x,y);ctx.fillRect(cx-.8,cy-.8,1.6,1.6);}
  ctx.globalAlpha=.88;
  for(const o of f.objects){
    const c=CC[o.cls]||'#555',[cx,cy]=tc(o.cx,o.cy);
    const pw=cv.width*o.ey/(V.yMax-V.yMin),ph=cv.height*o.ex/(V.xMax-V.xMin);
    ctx.strokeStyle=c;ctx.lineWidth=1.5;ctx.strokeRect(cx-pw/2,cy-ph/2,pw,ph);
    ctx.fillStyle=c;ctx.font='7px IBM Plex Mono,monospace';
    ctx.fillText(o.ttc_s!==null?`${o.cls.slice(0,3)} ${o.range_m}m ${o.ttc_s}s`:`${o.cls.slice(0,3)} ${o.range_m}m`,cx-pw/2+2,cy-ph/2-3);
  }
  ctx.globalAlpha=1;
  const[ecx,ecy]=tc(0,0);
  const epw=cv.width*2.2/(V.yMax-V.yMin),eph=cv.height*4.6/(V.xMax-V.xMin);
  ctx.strokeStyle='#00d4b8';ctx.lineWidth=2;ctx.shadowColor='#00d4b8';ctx.shadowBlur=10;
  ctx.strokeRect(ecx-epw/2,ecy-eph/2,epw,eph);ctx.shadowBlur=0;
  ctx.beginPath();ctx.moveTo(ecx,ecy);ctx.lineTo(ecx,ecy-eph*.6);ctx.lineWidth=1.5;ctx.stroke();
}
// Sidebar
const SC={PASS:'var(--gr)',WARN:'var(--am)',CRITICAL:'var(--rd)'};
function side(i){
  const f=F[i];
  document.getElementById('sb').className='sbanner '+f.status;
  document.getElementById('sw').textContent=f.status;
  document.getElementById('ap').textContent=f.action.replace(/_/g,' ');
  document.getElementById('mn').textContent=f.nearest_m<999?f.nearest_m+'m':'—';
  document.getElementById('mt').textContent=f.nearest_ttc<999?f.nearest_ttc+'s':'∞';
  document.getElementById('mo').textContent=f.n_objects;
  document.getElementById('mp').textContent=f.n_obstacle_pts.toLocaleString();
  document.getElementById('clist').innerHTML=f.constraints.map(c=>{
    const pct=Math.min(100,Math.abs(c.measured/c.limit)*100).toFixed(1);
    return`<div class="cst"><div class="csth"><span class="cstn">${c.name.replace(/_/g,' ')}</span><span class="cstv" style="color:${SC[c.status]}">${c.measured.toFixed(2)}/${c.limit.toFixed(2)}</span></div><div class="cbar"><div class="cfill ${c.status}_f" style="width:${pct}%"></div></div><div class="csts ${c.status}">${c.status}</div></div>`;
  }).join('');
  document.getElementById('olist').innerHTML=f.objects.length===0
    ?'<div style="color:var(--dim);font-size:8px;padding:4px 0">No objects detected</div>'
    :f.objects.map(o=>`<div class="orow"><span class="ocls">${o.cls}</span><span class="odst">${o.range_m}m</span><span class="ottc">${o.ttc_s!==null?o.ttc_s+'s':'—'}</span></div>`).join('');
}
// Cameras
function cams(i){
  const f=F[i];
  const imgF=document.getElementById('if'),noc=document.getElementById('noc');
  if(f.cam_front){imgF.src='data:image/jpeg;base64,'+f.cam_front;imgF.style.display='block';noc.style.display='none';}
  else{imgF.style.display='none';noc.style.display='flex';}
  ['il','ir'].forEach((id,j)=>{
    const img=document.getElementById(id);
    const b64=j===0?f.cam_left:f.cam_right;
    const nc=img.previousElementSibling;
    if(b64){img.src='data:image/jpeg;base64,'+b64;img.style.display='block';if(nc)nc.style.display='none';}
    else{img.style.display='none';if(nc)nc.style.display='flex';}
  });
  const rov=document.getElementById('rov');
  if(f.reasoning){
    rov.style.display='block';
    const ra=document.getElementById('ra'),rt=document.getElementById('rt');
    const cls=f.status==='PASS'?'match':f.status==='CRITICAL'?'mismatch':'unk';
    ra.className='rov-action '+cls;
    ra.textContent=(f.alpamayo_action||'—').replace(/_/g,' ');
    rt.textContent=f.reasoning;
  }else{rov.style.display='none';}
}
// Timeline
function buildTL(){
  const t=document.getElementById('track');
  F.forEach((f,i)=>{const b=document.createElement('div');b.className='tbar '+f.status;b.title='Spin '+i+': '+f.status;b.addEventListener('click',()=>goto(i));t.appendChild(b);});
}
function markTL(i){document.querySelectorAll('.tbar').forEach((b,j)=>b.classList.toggle('sel',j===i));}
function goto(i){
  cur=Math.max(0,Math.min(F.length-1,i));
  bev(cur);side(cur);cams(cur);markTL(cur);
  document.getElementById('si').textContent='Spin '+cur+' / '+(F.length-1);
  document.getElementById('ti').textContent='t='+(F[cur].timestamp_us/1e6).toFixed(2)+'s';
}
document.getElementById('bp').onclick=()=>goto(cur-1);
document.getElementById('bn').onclick=()=>goto(cur+1);
document.addEventListener('keydown',e=>{
  if(e.key==='ArrowLeft')goto(cur-1);
  if(e.key==='ArrowRight')goto(cur+1);
  if(e.key===' ')play();
});
function play(){
  playing=!playing;const b=document.getElementById('bpl');
  b.textContent=playing?'⏸':'▶';b.classList.toggle('on',playing);
  if(playing)tick();else clearTimeout(tmr);
}
function tick(){if(!playing)return;tmr=setTimeout(()=>{if(cur<F.length-1){goto(cur+1);tick();}else play();},110);}
document.getElementById('bpl').onclick=play;
// Header
document.getElementById('hclip').textContent=M.clip_id||'synthetic';
document.getElementById('hspins').textContent=F.length;
document.getElementById('hcrit').textContent=F.filter(f=>f.status==='CRITICAL').length;
document.getElementById('hwarn').textContent=F.filter(f=>f.status==='WARN').length;
if(HAS_CAM){document.getElementById('chip-cam').textContent='CAMERAS ON';document.getElementById('chip-cam').classList.add('on');}
if(HAS_TR){document.getElementById('chip-tr').textContent='ALPAMAYO TRACE';document.getElementById('chip-tr').classList.add('on');}
buildTL();rsz();goto(0);
</script>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Alpamayo LiDAR + Camera Viewer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── LiDAR source ──────────────────────────────────────────────────────
    src = p.add_mutually_exclusive_group()
    src.add_argument("--scenario",   default="emergency_stop",
                     choices=["highway","urban","emergency_stop","lane_change"],
                     help="Synthetic LiDAR scenario (no download needed)")
    src.add_argument("--clip-id",    metavar="UUID",
                     help="HuggingFace dataset clip UUID — loads LiDAR AND cameras automatically")
    src.add_argument("--lidar-path", metavar="PATH",
                     help="Local LiDAR parquet file")

    p.add_argument("--ego-path",  default=None, metavar="PATH",
                   help="Local egomotion parquet (optional with --lidar-path)")
    p.add_argument("--chunk",     type=int, default=None, metavar="N",
                   help="Chunk index — resolved automatically from clip_index.parquet if omitted")
    p.add_argument("--cache-dir", default="./cache", metavar="DIR",
                   help="Local HuggingFace cache directory (default: ./cache)")
    p.add_argument("--hf-token",  default=None, metavar="TOKEN",
                   help="HuggingFace token (falls back to HF_TOKEN env var)")

    # ── Camera options ────────────────────────────────────────────────────
    cam_group = p.add_mutually_exclusive_group()
    cam_group.add_argument("--cameras", nargs="+", metavar="CAM",
                           default=DEFAULT_CAMERAS,
                           help=f"Which cameras to load from HuggingFace (default: front 3). "
                                f"Choices: {', '.join(CAMERA_NAMES)}")
    cam_group.add_argument("--no-cameras", action="store_true",
                           help="Skip camera loading entirely (LiDAR-only viewer)")

    # Local camera overrides (used when --lidar-path is specified)
    p.add_argument("--camera-front-wide",  default=None, metavar="PATH",
                   help="Local mp4 for front wide camera (used with --lidar-path)")
    p.add_argument("--camera-front-left",  default=None, metavar="PATH",
                   help="Local mp4 for front left camera")
    p.add_argument("--camera-front-right", default=None, metavar="PATH",
                   help="Local mp4 for front right camera")
    p.add_argument("--camera-dir",         default=None, metavar="DIR",
                   help="Directory containing camera_*.mp4 files (AlpaSim video/ folder)")

    # ── Reasoning trace ───────────────────────────────────────────────────
    p.add_argument("--reasoning", default=None, metavar="PATH",
                   help="AlpaSim chain-of-causation JSONL file")

    # ── Pipeline options ──────────────────────────────────────────────────
    p.add_argument("--ego-speed",    type=float, default=22.0, metavar="M/S")
    p.add_argument("--max-spins",    type=int,   default=200,  metavar="N")
    p.add_argument("--cam-width",    type=int,   default=960,  metavar="PX")
    p.add_argument("--cam-height",   type=int,   default=540,  metavar="PX")
    p.add_argument("--cam-quality",  type=int,   default=82,   metavar="1-95",
                   help="JPEG quality — lower = smaller HTML file")
    p.add_argument("--output",       default="viewer.html",    metavar="FILE")

    args = p.parse_args()

    token = args.hf_token or os.environ.get("HF_TOKEN")

    # ── LiDAR ─────────────────────────────────────────────────────────────
    if args.clip_id:
        print(f"Loading LiDAR for clip {args.clip_id} from HuggingFace...")
        loader  = LidarLoader.from_hf(
            clip_uuid = args.clip_id,
            chunk_id  = args.chunk,
            cache_dir = args.cache_dir,
            token     = token,
        )
        clip_id = args.clip_id
    elif args.lidar_path:
        print(f"Loading LiDAR from {args.lidar_path}")
        loader  = LidarLoader.from_file(args.lidar_path, args.ego_path)
        clip_id = Path(args.lidar_path).stem
    else:
        print(f"Generating synthetic '{args.scenario}'...")
        loader  = LidarLoader.synthetic(
            scenario = args.scenario,
            n_spins  = min(args.max_spins, 50),
        )
        clip_id = f"synthetic:{args.scenario}"

    # ── Run pipeline ──────────────────────────────────────────────────────
    pipe = LidarPipeline(PipelineConfig(
        preprocess = PreprocessConfig(),
        detection  = DetectionConfig(),
        physics    = PhysicsConfig(),
    ))
    pipe._loader = loader

    print(f"\nRunning LiDAR pipeline (ego_speed={args.ego_speed} m/s)...")
    frames = collect_frames(pipe, args.ego_speed, args.max_spins)
    n = len(frames)
    print(f"Collected {n} frames.\n")

    # ── Cameras ───────────────────────────────────────────────────────────
    w, h, q = args.cam_width, args.cam_height, args.cam_quality

    if args.no_cameras:
        print("Cameras: skipped (--no-cameras)")

    elif args.clip_id:
        # Primary path: load cameras from HuggingFace alongside LiDAR
        print("Loading cameras from HuggingFace dataset...")
        cameras_to_load = [c for c in args.cameras if c in VIEWER_CAMERAS]
        inject_cameras_from_hf(
            frames    = frames,
            clip_uuid = args.clip_id,
            chunk_id  = args.chunk,
            cameras   = cameras_to_load,
            cache_dir = args.cache_dir,
            token     = token,
            width=w, height=h, quality=q,
        )

    else:
        # Fallback: local files specified via --camera-* or --camera-dir
        local_cams: Dict[str, str] = {}

        if args.camera_front_wide:  local_cams["camera_front_wide_120fov"]  = args.camera_front_wide
        if args.camera_front_left:  local_cams["camera_front_left_120fov"]  = args.camera_front_left
        if args.camera_front_right: local_cams["camera_front_right_120fov"] = args.camera_front_right

        if args.camera_dir and os.path.isdir(args.camera_dir):
            import glob as _glob
            for fname in os.listdir(args.camera_dir):
                fpath = os.path.join(args.camera_dir, fname)
                low   = fname.lower().replace("-", "_")
                for cam in CAMERA_NAMES:
                    short = cam.replace("camera_","").replace("_120fov","").replace("_70fov","").replace("_30fov","")
                    if short in low or cam in low:
                        local_cams.setdefault(cam, fpath)
                        break

        if local_cams:
            print("Loading cameras from local files...")
            inject_cameras_from_local(frames, local_cams, w, h, q)
        else:
            print("No camera sources found — LiDAR-only viewer.")
            print("  With --clip-id: cameras load automatically from HuggingFace.")
            print("  With --lidar-path: use --camera-front-wide / --camera-dir.")

    # ── Reasoning trace ───────────────────────────────────────────────────
    if args.reasoning:
        print(f"\nLoading reasoning trace from {args.reasoning}...")
        traces = load_reasoning_trace(args.reasoning)
        for f in frames:
            t = traces.get(f["spin"])
            if t:
                f["reasoning"]        = t["text"]
                f["alpamayo_action"]  = t["action"]
        print(f"  {len(traces)} entries loaded.")

    # ── Write HTML ────────────────────────────────────────────────────────
    meta = {"clip_id": clip_id}
    html = HTML \
        .replace("__FRAMES_JSON__", json.dumps(frames)) \
        .replace("__META_JSON__",   json.dumps(meta))

    with open(args.output, "w", encoding="utf-8") as fh:
        fh.write(html)

    size_mb = os.path.getsize(args.output) / 1024 / 1024
    print(f"\nViewer saved → {args.output}  ({size_mb:.1f} MB)")
    print("Open in Chrome or Firefox.")
    print("Controls: ← → arrow keys · spacebar = play/pause")


if __name__ == "__main__":
    main()