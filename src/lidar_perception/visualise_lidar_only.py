"""
visualise.py
────────────────────────────────────────────────────────────────────────────
Runs the LiDAR pipeline on a clip (real or synthetic) and generates a
self-contained interactive HTML viewer you can open in any browser.

The viewer shows:
  • Bird's-eye view (BEV) point cloud for each spin
  • Detected objects with bounding boxes coloured by class
  • Ego path corridor overlay
  • Occupancy grid heatmap
  • Physics constraint status panel
  • Spin timeline scrubber

Usage
─────
  # Synthetic (no dataset needed)
  python visualise.py --scenario emergency_stop --output viewer.html

  # Real clip (requires HF_TOKEN and downloaded data)
  python visualise.py --clip-id <uuid> --chunk <int> --output viewer.html

  # From already-downloaded local parquet
  python visualise.py --lidar-path path/to/lidar.parquet --output viewer.html

Then open viewer.html in Chrome / Firefox.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd

# ── make the package importable when run from project root ───────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from physical_ai_av_lidar import (
    LidarLoader,
    LidarPipeline,
    PipelineConfig,
    PreprocessConfig,
    DetectionConfig,
    PhysicsConfig,
)
from physical_ai_av_lidar.physics_reasoning import ConstraintStatus


# ─────────────────────────────────────────────────────────────────────────────
# Data collection
# ─────────────────────────────────────────────────────────────────────────────

def collect_frames(pipe: LidarPipeline, ego_speed_ms: float, max_spins: int):
    """
    Run the pipeline and collect per-spin data for the viewer.
    Returns a list of frame dicts serialisable to JSON.
    """
    frames = []

    # We need access to intermediate results, so we step through manually
    loader     = pipe._loader
    preproc    = pipe._preprocessor
    detector   = pipe._detector
    tracker    = pipe._tracker
    reasoner   = pipe._reasoner

    prev_ego_speed = ego_speed_ms
    prev_ts        = None

    for spin in loader.iter_spins(stop=max_spins):
        # Preprocessing
        processed = preproc.process(spin)

        # Detection + tracking
        det_frame = detector.detect(processed)
        det_frame = tracker.update(det_frame)

        # Ego dynamics (approximate from ego pose if available)
        lateral_accel = 0.0
        long_accel    = 0.0
        yaw_rate      = 0.0
        dt            = 1.0 / 10.0

        if spin.ego_pose is not None and prev_ts is not None:
            dt = max((spin.timestamp_us - prev_ts) / 1e6, 0.01)

        # Physics
        report = reasoner.reason(
            detection_frame        = det_frame,
            ego_speed_ms           = ego_speed_ms,
            lateral_accel_ms2      = lateral_accel,
            longitudinal_accel_ms2 = long_accel,
            yaw_rate_rads          = yaw_rate,
            prev_ego_speed         = prev_ego_speed,
            dt_s                   = dt,
        )

        # ── Subsample point cloud for JSON size ──────────────────────────
        xyz_obs = processed.xyz_obstacle
        if len(xyz_obs) > 4000:
            idx     = np.random.choice(len(xyz_obs), 4000, replace=False)
            xyz_obs = xyz_obs[idx]

        xyz_gnd = processed.xyz_ground
        if len(xyz_gnd) > 2000:
            idx     = np.random.choice(len(xyz_gnd), 2000, replace=False)
            xyz_gnd = xyz_gnd[idx]

        # ── Detected objects ──────────────────────────────────────────────
        objects = []
        for obj in det_frame.objects:
            cx, cy, cz = obj.bbox.center.tolist()
            ex, ey, ez = obj.bbox.extent.tolist()
            objects.append({
                "id":       obj.object_id,
                "cls":      obj.cls.value,
                "cx": cx, "cy": cy,
                "ex": ex, "ey": ey,
                "range_m":  round(obj.range_m, 2),
                "ttc_s":    round(obj.ttc_s, 2) if obj.ttc_s is not None and not math.isinf(obj.ttc_s) else None,
                "conf":     round(obj.confidence, 3),
            })

        # ── Constraints ───────────────────────────────────────────────────
        constraints = []
        for c in report.constraints:
            constraints.append({
                "name":     c.name,
                "status":   c.status.value,
                "measured": round(float(c.measured), 3),
                "limit":    round(float(c.limit), 3),
                "margin":   round(float(c.margin), 3),
            })

        # ── Occupancy grid (downsample to 50×50 for JSON) ────────────────
        grid     = report.occupancy_grid
        grid_arr = grid.data
        # Downsample by taking max in 4×4 blocks
        h, w     = grid_arr.shape
        bh, bw   = max(1, h // 50), max(1, w // 50)
        oh, ow   = h // bh, w // bw
        occ_small = grid_arr[:oh * bh, :ow * bw].reshape(oh, bh, ow, bw).max(axis=(1, 3))

        frame = {
            "spin":        spin.scan_index,
            "timestamp_us": spin.timestamp_us,
            "status":      report.overall_status,
            "action":      report.recommended_action.value,
            "nearest_m":   round(report.nearest_range_m, 2),
            "nearest_ttc": round(report.nearest_ttc_s, 2) if not math.isinf(report.nearest_ttc_s) else 999,
            "n_objects":   len(det_frame.objects),
            "n_obstacle_pts": len(processed.xyz_obstacle),
            "pts_obs":     xyz_obs[:, :2].tolist(),   # x,y only (BEV)
            "pts_gnd":     xyz_gnd[:, :2].tolist(),
            "objects":     objects,
            "constraints": constraints,
            "occ_grid":    occ_small.tolist(),
            "grid_x_min":  grid.x_min,
            "grid_x_max":  grid.x_max,
            "grid_y_min":  grid.y_min,
            "grid_y_max":  grid.y_max,
        }
        frames.append(frame)

        prev_ego_speed = ego_speed_ms
        prev_ts        = spin.timestamp_us

        status_char = {"PASS": "✓", "WARN": "⚠", "CRITICAL": "✗"}.get(report.overall_status, "?")
        print(f"  Spin {spin.scan_index:3d} {status_char}  "
              f"nearest {report.nearest_range_m:6.1f}m  "
              f"{report.recommended_action.value}")

    return frames


# ─────────────────────────────────────────────────────────────────────────────
# HTML generation
# ─────────────────────────────────────────────────────────────────────────────

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LiDAR Pipeline Viewer</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;600&family=Syne:wght@700;800&display=swap');

:root {
  --bg:      #080b0d;
  --surface: #0d1114;
  --border:  #1a2128;
  --dim:     #2a3840;
  --text:    #6a8a98;
  --hi:      #c0d8e4;
  --cyan:    #00e5cc;
  --amber:   #ffb700;
  --red:     #ff3f50;
  --green:   #39ff8a;
  --blue:    #4488ff;
  --mono:    'IBM Plex Mono', monospace;
  --display: 'Syne', sans-serif;
}

* { box-sizing: border-box; margin: 0; padding: 0; }

body {
  background: var(--bg);
  color: var(--text);
  font-family: var(--mono);
  font-size: 11px;
  height: 100vh;
  display: grid;
  grid-template-rows: 48px 1fr 72px;
  grid-template-columns: 1fr 280px;
  grid-template-areas:
    "header  header"
    "main    sidebar"
    "timeline timeline";
  overflow: hidden;
}

/* scanlines */
body::after {
  content: '';
  position: fixed; inset: 0;
  background: repeating-linear-gradient(0deg, transparent 0, transparent 3px,
    rgba(0,229,204,.006) 3px, rgba(0,229,204,.006) 4px);
  pointer-events: none; z-index: 999;
}

/* ── HEADER ── */
header {
  grid-area: header;
  display: flex; align-items: center; justify-content: space-between;
  padding: 0 20px;
  border-bottom: 1px solid var(--border);
  background: var(--surface);
}

header h1 {
  font-family: var(--display);
  font-size: 15px; font-weight: 800;
  color: var(--cyan);
  letter-spacing: .04em;
  text-shadow: 0 0 20px rgba(0,229,204,.4);
}

.hdr-meta { display: flex; gap: 24px; }
.hdr-stat { display: flex; flex-direction: column; align-items: flex-end; }
.hdr-stat span:first-child { font-size: 9px; letter-spacing: .12em; text-transform: uppercase; color: var(--dim); }
.hdr-stat span:last-child  { font-size: 13px; font-weight: 600; color: var(--hi); }

/* ── MAIN CANVAS ── */
main {
  grid-area: main;
  position: relative;
  overflow: hidden;
}

canvas#bev {
  width: 100%; height: 100%;
  display: block;
}

.canvas-label {
  position: absolute; top: 12px; left: 16px;
  font-size: 9px; letter-spacing: .15em; text-transform: uppercase;
  color: var(--dim);
}

.legend {
  position: absolute; bottom: 12px; left: 16px;
  display: flex; gap: 14px;
}
.legend-item { display: flex; align-items: center; gap: 5px; font-size: 9px; color: var(--text); }
.legend-dot  { width: 8px; height: 8px; border-radius: 50%; }

/* ── SIDEBAR ── */
aside {
  grid-area: sidebar;
  border-left: 1px solid var(--border);
  overflow-y: auto;
  background: var(--surface);
  padding: 16px;
  display: flex; flex-direction: column; gap: 16px;
}

.panel-title {
  font-size: 9px; letter-spacing: .15em; text-transform: uppercase;
  color: var(--dim); margin-bottom: 8px;
}

/* Status badge */
.status-badge {
  display: flex; align-items: center; justify-content: space-between;
  padding: 10px 14px;
  border-radius: 4px;
  border: 1px solid var(--border);
}
.status-badge.PASS     { border-color: var(--green); background: rgba(57,255,138,.05); }
.status-badge.WARN     { border-color: var(--amber); background: rgba(255,183,0,.05); }
.status-badge.CRITICAL { border-color: var(--red);   background: rgba(255,63,80,.07); }

.status-label { font-size: 13px; font-weight: 600; }
.PASS     .status-label { color: var(--green); }
.WARN     .status-label { color: var(--amber); }
.CRITICAL .status-label { color: var(--red); }

.action-tag {
  font-size: 9px; padding: 3px 7px;
  border-radius: 2px; border: 1px solid var(--dim);
  color: var(--hi); letter-spacing: .08em;
}

/* Constraints */
.constraint {
  margin-bottom: 8px;
}
.constraint-header {
  display: flex; justify-content: space-between; align-items: center;
  margin-bottom: 3px;
}
.constraint-name { font-size: 10px; color: var(--hi); }
.constraint-vals { font-size: 9px; color: var(--text); }
.cbar-bg {
  height: 4px; border-radius: 2px;
  background: var(--border);
  overflow: hidden;
}
.cbar-fill {
  height: 100%; border-radius: 2px;
  transition: width .15s ease;
}
.PASS_bar     { background: var(--green); }
.WARN_bar     { background: var(--amber); }
.CRITICAL_bar { background: var(--red); }

/* Objects list */
.obj-row {
  display: flex; justify-content: space-between; align-items: center;
  padding: 5px 0; border-bottom: 1px solid var(--border);
  font-size: 10px;
}
.obj-cls   { color: var(--cyan); width: 80px; }
.obj-range { color: var(--hi); }
.obj-ttc   { color: var(--amber); }

/* Stats grid */
.stats-grid {
  display: grid; grid-template-columns: 1fr 1fr;
  gap: 8px;
}
.stat-box {
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 3px;
  padding: 8px 10px;
}
.stat-box .label { font-size: 8px; text-transform: uppercase; letter-spacing: .1em; color: var(--dim); }
.stat-box .value { font-size: 14px; font-weight: 600; color: var(--hi); margin-top: 2px; }

/* ── TIMELINE ── */
#timeline-area {
  grid-area: timeline;
  border-top: 1px solid var(--border);
  background: var(--surface);
  padding: 10px 20px;
  display: flex; flex-direction: column; gap: 6px;
}

.timeline-controls {
  display: flex; align-items: center; gap: 14px;
}

button {
  background: var(--border); border: 1px solid var(--dim);
  color: var(--hi); font-family: var(--mono); font-size: 10px;
  padding: 4px 10px; border-radius: 2px; cursor: pointer;
  transition: background .1s;
}
button:hover { background: var(--dim); }
button.active { background: var(--cyan); color: var(--bg); border-color: var(--cyan); }

#spin-label { font-size: 11px; color: var(--hi); min-width: 60px; }

.timeline-track {
  position: relative; height: 24px;
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 2px;
  cursor: pointer;
  overflow: hidden;
}

.timeline-bars {
  display: flex; height: 100%; gap: 1px;
  padding: 2px;
}

.tbar {
  flex: 1; border-radius: 1px; min-width: 2px;
  opacity: .7; transition: opacity .1s;
  cursor: pointer;
}
.tbar:hover { opacity: 1; }
.tbar.PASS     { background: var(--green); }
.tbar.WARN     { background: var(--amber); }
.tbar.CRITICAL { background: var(--red); }
.tbar.active   { opacity: 1; outline: 1px solid white; }

</style>
</head>
<body>

<header>
  <h1>LiDAR Pipeline Viewer</h1>
  <div class="hdr-meta">
    <div class="hdr-stat">
      <span>Clip</span>
      <span id="hdr-clip">—</span>
    </div>
    <div class="hdr-stat">
      <span>Spins</span>
      <span id="hdr-spins">—</span>
    </div>
    <div class="hdr-stat">
      <span>CRITICAL</span>
      <span id="hdr-crit" style="color:var(--red)">—</span>
    </div>
  </div>
</header>

<main>
  <canvas id="bev"></canvas>
  <div class="canvas-label">BIRD'S EYE VIEW · EGO FRAME</div>
  <div class="legend">
    <div class="legend-item"><div class="legend-dot" style="background:#4488ff"></div>Ground</div>
    <div class="legend-item"><div class="legend-dot" style="background:#00e5cc"></div>Obstacle pts</div>
    <div class="legend-item"><div class="legend-dot" style="background:#ff3f50"></div>Vehicle</div>
    <div class="legend-item"><div class="legend-dot" style="background:#ffb700"></div>Pedestrian</div>
    <div class="legend-item"><div class="legend-dot" style="background:#39ff8a"></div>Cyclist</div>
    <div class="legend-item"><div class="legend-dot" style="background:#888"></div>Static/Unknown</div>
  </div>
</main>

<aside>
  <!-- Status -->
  <div>
    <div class="panel-title">Status</div>
    <div class="status-badge PASS" id="status-badge">
      <span class="status-label" id="status-label">PASS</span>
      <span class="action-tag" id="action-tag">—</span>
    </div>
  </div>

  <!-- Stats -->
  <div>
    <div class="panel-title">Metrics</div>
    <div class="stats-grid">
      <div class="stat-box"><div class="label">Nearest</div><div class="value" id="stat-nearest">—</div></div>
      <div class="stat-box"><div class="label">TTC</div><div class="value" id="stat-ttc">—</div></div>
      <div class="stat-box"><div class="label">Objects</div><div class="value" id="stat-objs">—</div></div>
      <div class="stat-box"><div class="label">Pts (obs)</div><div class="value" id="stat-pts">—</div></div>
    </div>
  </div>

  <!-- Constraints -->
  <div>
    <div class="panel-title">Constraints</div>
    <div id="constraints-list"></div>
  </div>

  <!-- Objects -->
  <div>
    <div class="panel-title">Detected Objects</div>
    <div id="objects-list"></div>
  </div>
</aside>

<div id="timeline-area">
  <div class="timeline-controls">
    <button id="btn-prev">◀</button>
    <button id="btn-play">▶ Play</button>
    <button id="btn-next">▶</button>
    <span id="spin-label">Spin 0</span>
    <span style="color:var(--dim);font-size:10px" id="timestamp-label"></span>
  </div>
  <div class="timeline-track" id="timeline-track">
    <div class="timeline-bars" id="timeline-bars"></div>
  </div>
</div>

<script>
// ── Injected data ──────────────────────────────────────────────────────────
const FRAMES = __FRAMES_JSON__;
const META   = __META_JSON__;

// ── State ──────────────────────────────────────────────────────────────────
let currentSpin = 0;
let playing     = false;
let playTimer   = null;

// ── Canvas setup ───────────────────────────────────────────────────────────
const canvas = document.getElementById('bev');
const ctx    = canvas.getContext('2d');

function resizeCanvas() {
  const main = canvas.parentElement;
  canvas.width  = main.clientWidth;
  canvas.height = main.clientHeight;
  drawFrame(currentSpin);
}
window.addEventListener('resize', resizeCanvas);

// ── World → canvas transform ───────────────────────────────────────────────
// X = forward (up on screen), Y = left (right on screen in BEV)
// We show X in [-15, 60] and Y in [-25, 25]
const VIEW = { xMin: -8, xMax: 55, yMin: -25, yMax: 25 };

function toCanvas(wx, wy) {
  // wx = forward, wy = left
  // canvas: cx increases right, cy increases down
  // map wy → cx (left = right on screen in standard BEV)
  const cx = canvas.width  * (-wy - VIEW.yMin) / (VIEW.yMax - VIEW.yMin);
  const cy = canvas.height * (1 - (wx - VIEW.xMin) / (VIEW.xMax - VIEW.xMin));
  return [cx, cy];
}

// ── Draw ───────────────────────────────────────────────────────────────────
const CLS_COLORS = {
  VEHICLE:    '#ff3f50',
  PEDESTRIAN: '#ffb700',
  CYCLIST:    '#39ff8a',
  STATIC:     '#555',
  UNKNOWN:    '#446',
};

function drawFrame(spinIdx) {
  const f = FRAMES[spinIdx];
  if (!f) return;

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Background grid
  drawGrid();

  // Corridor
  drawCorridor();

  // Ground points
  ctx.fillStyle = 'rgba(68,136,255,0.25)';
  for (const [x, y] of f.pts_gnd) {
    const [cx, cy] = toCanvas(x, y);
    ctx.fillRect(cx - 0.5, cy - 0.5, 1, 1);
  }

  // Obstacle points
  ctx.fillStyle = 'rgba(0,229,204,0.55)';
  for (const [x, y] of f.pts_obs) {
    const [cx, cy] = toCanvas(x, y);
    ctx.fillRect(cx - 0.8, cy - 0.8, 1.6, 1.6);
  }

  // Detected object bounding boxes
  for (const obj of f.objects) {
    const color = CLS_COLORS[obj.cls] || '#888';
    const [cx, cy] = toCanvas(obj.cx, obj.cy);
    const pw = canvas.width  * obj.ey / (VIEW.yMax - VIEW.yMin);
    const ph = canvas.height * obj.ex / (VIEW.xMax - VIEW.xMin);

    ctx.strokeStyle = color;
    ctx.lineWidth   = 1.5;
    ctx.globalAlpha = 0.85;
    ctx.strokeRect(cx - pw / 2, cy - ph / 2, pw, ph);

    // Label
    ctx.globalAlpha = 1;
    ctx.fillStyle   = color;
    ctx.font        = '9px IBM Plex Mono, monospace';
    const label = obj.ttc_s !== null
      ? `${obj.cls.slice(0,3)} ${obj.range_m}m ${obj.ttc_s}s`
      : `${obj.cls.slice(0,3)} ${obj.range_m}m`;
    ctx.fillText(label, cx - pw / 2 + 2, cy - ph / 2 - 3);
  }

  ctx.globalAlpha = 1;

  // Ego vehicle marker
  drawEgo();
}

function drawGrid() {
  ctx.strokeStyle = '#1a2128';
  ctx.lineWidth   = 0.5;
  // X grid lines (range rings at 10m intervals)
  for (let x = 0; x <= 60; x += 10) {
    const [, cy] = toCanvas(x, 0);
    ctx.beginPath();
    ctx.moveTo(0, cy); ctx.lineTo(canvas.width, cy);
    ctx.stroke();
    ctx.fillStyle = '#2a3840';
    ctx.font      = '8px IBM Plex Mono, monospace';
    ctx.fillText(`${x}m`, 4, cy - 2);
  }
  // Y grid lines (lane markers at 3.5m intervals)
  for (let y = -21; y <= 21; y += 3.5) {
    const [cx] = toCanvas(0, y);
    ctx.beginPath();
    ctx.moveTo(cx, 0); ctx.lineTo(cx, canvas.height);
    ctx.stroke();
  }
}

function drawCorridor() {
  // Corridor: x in [0,40], y in [-2,2]
  const [cx1, cy1] = toCanvas(0,   2);
  const [cx2, cy2] = toCanvas(40, -2);
  const w = Math.abs(cx2 - cx1);
  const h = Math.abs(cy2 - cy1);
  const x = Math.min(cx1, cx2);
  const y = Math.min(cy1, cy2);

  ctx.fillStyle   = 'rgba(0,229,204,0.04)';
  ctx.strokeStyle = 'rgba(0,229,204,0.18)';
  ctx.lineWidth   = 1;
  ctx.fillRect(x, y, w, h);
  ctx.strokeRect(x, y, w, h);
}

function drawEgo() {
  const [cx, cy] = toCanvas(0, 0);
  // Car outline (~4.5m × 2m)
  const pw = canvas.width  * 2.0  / (VIEW.yMax - VIEW.yMin);
  const ph = canvas.height * 4.5  / (VIEW.xMax - VIEW.xMin);
  ctx.strokeStyle = '#00e5cc';
  ctx.lineWidth   = 2;
  ctx.shadowColor = '#00e5cc';
  ctx.shadowBlur  = 8;
  ctx.strokeRect(cx - pw / 2, cy - ph / 2, pw, ph);
  ctx.shadowBlur  = 0;

  // Direction arrow
  ctx.strokeStyle = '#00e5cc';
  ctx.lineWidth   = 1.5;
  ctx.beginPath();
  ctx.moveTo(cx, cy);
  ctx.lineTo(cx, cy - ph * 0.6);
  ctx.stroke();
}

// ── Sidebar update ─────────────────────────────────────────────────────────
const STATUS_COLORS = { PASS: 'var(--green)', WARN: 'var(--amber)', CRITICAL: 'var(--red)' };

function updateSidebar(spinIdx) {
  const f = FRAMES[spinIdx];
  if (!f) return;

  // Status badge
  const badge = document.getElementById('status-badge');
  badge.className = `status-badge ${f.status}`;
  document.getElementById('status-label').textContent = f.status;
  document.getElementById('action-tag').textContent   = f.action.replace(/_/g, ' ');

  // Stats
  document.getElementById('stat-nearest').textContent = f.nearest_m < 999 ? `${f.nearest_m}m` : '—';
  document.getElementById('stat-ttc').textContent     = f.nearest_ttc < 999 ? `${f.nearest_ttc}s` : '∞';
  document.getElementById('stat-objs').textContent    = f.n_objects;
  document.getElementById('stat-pts').textContent     = f.n_obstacle_pts.toLocaleString();

  // Constraints
  const cl = document.getElementById('constraints-list');
  cl.innerHTML = '';
  for (const c of f.constraints) {
    const pct = Math.min(100, Math.abs(c.measured / c.limit) * 100);
    cl.innerHTML += `
      <div class="constraint">
        <div class="constraint-header">
          <span class="constraint-name">${c.name.replace(/_/g,' ')}</span>
          <span class="constraint-vals" style="color:${STATUS_COLORS[c.status]}">${c.status}</span>
        </div>
        <div class="constraint-vals" style="margin-bottom:3px">
          ${c.measured.toFixed(2)} / ${c.limit.toFixed(2)}
        </div>
        <div class="cbar-bg">
          <div class="cbar-fill ${c.status}_bar" style="width:${pct.toFixed(1)}%"></div>
        </div>
      </div>`;
  }

  // Objects
  const ol = document.getElementById('objects-list');
  if (f.objects.length === 0) {
    ol.innerHTML = '<div style="color:var(--dim);font-size:10px">No objects detected</div>';
  } else {
    ol.innerHTML = f.objects.map(o => `
      <div class="obj-row">
        <span class="obj-cls">${o.cls}</span>
        <span class="obj-range">${o.range_m}m</span>
        <span class="obj-ttc">${o.ttc_s !== null ? o.ttc_s + 's' : '—'}</span>
      </div>`).join('');
  }
}

// ── Timeline ───────────────────────────────────────────────────────────────
function buildTimeline() {
  const bars = document.getElementById('timeline-bars');
  bars.innerHTML = '';
  for (let i = 0; i < FRAMES.length; i++) {
    const b = document.createElement('div');
    b.className   = `tbar ${FRAMES[i].status}`;
    b.dataset.idx = i;
    b.title       = `Spin ${i}: ${FRAMES[i].status}`;
    b.addEventListener('click', () => goTo(i));
    bars.appendChild(b);
  }
}

function highlightTimeline(idx) {
  document.querySelectorAll('.tbar').forEach((b, i) => {
    b.classList.toggle('active', i === idx);
  });
}

// ── Navigation ─────────────────────────────────────────────────────────────
function goTo(idx) {
  currentSpin = Math.max(0, Math.min(FRAMES.length - 1, idx));
  drawFrame(currentSpin);
  updateSidebar(currentSpin);
  highlightTimeline(currentSpin);
  document.getElementById('spin-label').textContent =
    `Spin ${currentSpin} / ${FRAMES.length - 1}`;
  document.getElementById('timestamp-label').textContent =
    `t = ${(FRAMES[currentSpin].timestamp_us / 1e6).toFixed(2)}s`;
}

document.getElementById('btn-prev').addEventListener('click', () => goTo(currentSpin - 1));
document.getElementById('btn-next').addEventListener('click', () => goTo(currentSpin + 1));

document.addEventListener('keydown', e => {
  if (e.key === 'ArrowLeft')  goTo(currentSpin - 1);
  if (e.key === 'ArrowRight') goTo(currentSpin + 1);
  if (e.key === ' ') togglePlay();
});

// ── Playback ───────────────────────────────────────────────────────────────
function togglePlay() {
  playing = !playing;
  const btn = document.getElementById('btn-play');
  btn.textContent = playing ? '⏸ Pause' : '▶ Play';
  btn.classList.toggle('active', playing);
  if (playing) scheduleNext();
  else clearTimeout(playTimer);
}

function scheduleNext() {
  if (!playing) return;
  playTimer = setTimeout(() => {
    if (currentSpin < FRAMES.length - 1) {
      goTo(currentSpin + 1);
      scheduleNext();
    } else {
      togglePlay();
    }
  }, 120);
}

document.getElementById('btn-play').addEventListener('click', togglePlay);

// ── Header meta ────────────────────────────────────────────────────────────
document.getElementById('hdr-clip').textContent   = META.clip_id || 'synthetic';
document.getElementById('hdr-spins').textContent  = FRAMES.length;
document.getElementById('hdr-crit').textContent   =
  FRAMES.filter(f => f.status === 'CRITICAL').length;

// ── Init ───────────────────────────────────────────────────────────────────
buildTimeline();
resizeCanvas();
goTo(0);
</script>
</body>
</html>
"""


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LiDAR pipeline visualiser")
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--scenario",   default="emergency_stop",
                     choices=["highway", "urban", "emergency_stop", "lane_change"],
                     help="Synthetic scenario (default: emergency_stop)")
    src.add_argument("--clip-id",    help="Real dataset clip UUID")
    src.add_argument("--lidar-path", help="Local lidar parquet path")

    parser.add_argument("--chunk",      type=int,   default=None)
    parser.add_argument("--ego-path",   default=None)
    parser.add_argument("--cache-dir",  default="./cache")
    parser.add_argument("--ego-speed",  type=float, default=22.0)
    parser.add_argument("--max-spins",  type=int,   default=200)
    parser.add_argument("--output",     default="viewer.html")
    args = parser.parse_args()

    # ── Load ──────────────────────────────────────────────────────────────
    if args.clip_id:
        print(f"Loading clip {args.clip_id} from HuggingFace...")
        loader = LidarLoader.from_hf(
            clip_uuid=args.clip_id,
            chunk_id=args.chunk,
            cache_dir=args.cache_dir,
        )
        clip_id = args.clip_id
    elif args.lidar_path:
        print(f"Loading from {args.lidar_path}...")
        loader  = LidarLoader.from_file(args.lidar_path, args.ego_path)
        clip_id = os.path.basename(args.lidar_path)
    else:
        print(f"Generating synthetic '{args.scenario}' scenario...")
        loader  = LidarLoader.synthetic(scenario=args.scenario, n_spins=min(args.max_spins, 50))
        clip_id = f"synthetic:{args.scenario}"

    # ── Build pipeline ────────────────────────────────────────────────────
    pipe = LidarPipeline(PipelineConfig(
        preprocess = PreprocessConfig(),
        detection  = DetectionConfig(),
        physics    = PhysicsConfig(),
    ))
    pipe._loader = loader

    # ── Collect frames ────────────────────────────────────────────────────
    print(f"Running pipeline (ego_speed={args.ego_speed} m/s)...")
    frames = collect_frames(pipe, args.ego_speed, args.max_spins)
    print(f"Collected {len(frames)} frames.")

    meta = {"clip_id": clip_id}

    # ── Generate HTML ─────────────────────────────────────────────────────
    html = HTML_TEMPLATE \
        .replace("__FRAMES_JSON__", json.dumps(frames)) \
        .replace("__META_JSON__",   json.dumps(meta))

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(html)

    size_kb = os.path.getsize(args.output) / 1024
    print(f"\nViewer saved → {args.output}  ({size_kb:.0f} KB)")
    print("Open in Chrome or Firefox. Use ← → arrow keys or spacebar to play.")


if __name__ == "__main__":
    main()