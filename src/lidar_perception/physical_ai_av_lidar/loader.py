"""
loader.py
─────────────────────────────────────────────────────────────────────────────
Loads nvidia/PhysicalAI-Autonomous-Vehicles LiDAR parquet files.

Supports three access patterns:
  1. Local path  – parquet already downloaded to disk
  2. HuggingFace – streams from HF hub via huggingface_hub (requires auth)
  3. Synthetic   – generates realistic fake spins for unit-testing without
                   dataset access (useful for CI and pipeline development)

Usage
─────
  from physical_ai_av_lidar.loader import LidarLoader

  # From local file
  loader = LidarLoader.from_file("path/to/<clip_uuid>.lidar_top360_fov.parquet")

  # From HuggingFace (requires HF_TOKEN env var or huggingface-cli login)
  # chunk_id is resolved automatically if omitted
  loader = LidarLoader.from_hf(clip_uuid="<uuid>")
  # or supply chunk_id directly to skip the metadata lookup
  loader = LidarLoader.from_hf(clip_uuid="<uuid>", chunk_id=4)

  # Synthetic (no dataset access needed)
  loader = LidarLoader.synthetic(n_spins=10, points_per_spin=30_000)

  for spin in loader.iter_spins():
      process(spin)   # spin is a SpinFrame namedtuple
"""

from __future__ import annotations

import os
import io
import math
import random
import struct
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np
import pandas as pd

from .schema import (
    LIDAR_COLS,
    RANGE_MIN_M,
    RANGE_MAX_M,
    SPINS_PER_CLIP,
    LIDAR_HZ,
    ClipMetadata,
    LidarExtrinsics,
)


# ── Data containers ───────────────────────────────────────────────────────────

@dataclass
class SpinFrame:
    """
    One full 360° LiDAR spin.

    Attributes
    ----------
    scan_index : int
        Sequential spin number within the clip (0–199).
    timestamp_us : int
        System wall-clock timestamp of the spin centre, microseconds.
    points : np.ndarray, shape (N, 6)
        Columns: [x, y, z, intensity, ring, return_type]
        All in rig-frame coordinates (metres, normalised intensity).
    ego_pose : Optional[np.ndarray], shape (7,)
        [x, y, z, qx, qy, qz, qw] at spin timestamp; None if not loaded.
    """
    scan_index: int
    timestamp_us: int
    points: np.ndarray          # (N, 6) float32
    ego_pose: Optional[np.ndarray] = None  # (7,) float64

    @property
    def xyz(self) -> np.ndarray:
        """Point positions (N, 3)."""
        return self.points[:, :3]

    @property
    def intensity(self) -> np.ndarray:
        """Normalised intensity (N,)."""
        return self.points[:, 3]

    @property
    def ring(self) -> np.ndarray:
        """Beam / ring indices (N,), int."""
        return self.points[:, 4].astype(np.int32)

    @property
    def return_type(self) -> np.ndarray:
        """Return type flags (N,): 0=first, 1=second."""
        return self.points[:, 5].astype(np.uint8)

    @property
    def n_points(self) -> int:
        return len(self.points)


# ── Loader ────────────────────────────────────────────────────────────────────

class LidarLoader:
    """
    Loads and iterates over LiDAR spin frames from a parquet clip file.

    Parameters
    ----------
    df : pd.DataFrame
        Full LiDAR parquet (all spins concatenated).
    metadata : ClipMetadata
        Clip-level metadata (may be partially populated).
    ego_df : Optional[pd.DataFrame]
        Ego-motion parquet for the same clip; used to attach pose per spin.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        metadata: Optional[ClipMetadata] = None,
        ego_df: Optional[pd.DataFrame] = None,
    ):
        self._df = df
        self.metadata = metadata or ClipMetadata()
        self._ego_df = ego_df
        self._scan_groups = None  # lazy

    # ── Constructors ──────────────────────────────────────────────────────────

    @classmethod
    def from_file(
        cls,
        lidar_path: str,
        ego_path: Optional[str] = None,
        metadata: Optional[ClipMetadata] = None,
    ) -> "LidarLoader":
        """
        Load from local parquet file(s).

        Handles two on-disk formats:
          1. Flat columns  - x, y, z, intensity, ring, scan_index, ...
             (synthetic data and older dataset versions)
          2. Draco-encoded - reference_timestamp + draco_encoded_pointcloud
             (nvidia/PhysicalAI-Autonomous-Vehicles real data)
             Each row is one spin; the point cloud is Draco-compressed binary.
             Requires: pip install DracoPy
        """
        raw = pd.read_parquet(lidar_path)

        if "draco_encoded_pointcloud" in raw.columns:
            df = _decode_draco_parquet(raw)
        else:
            _validate_lidar_df(raw)
            df = raw

        ego_df = None
        if ego_path:
            ego_raw = pd.read_parquet(ego_path)
            ego_df  = _normalise_ego_df(ego_raw)

        return cls(df, metadata, ego_df)

    @classmethod
    def from_hf(
        cls,
        clip_uuid: str,
        chunk_id: Optional[int] = None,
        cache_dir: Optional[str] = None,
        load_ego: bool = True,
        token: Optional[str] = None,
    ) -> "LidarLoader":
        """
        Download and load a LiDAR clip from HuggingFace hub.

        The dataset stores LiDAR data inside per-chunk zip archives.  This
        method downloads the correct zip, extracts only the parquet for the
        requested clip_uuid, then loads it.  The extracted parquet is cached
        so subsequent calls for the same clip are instant.

        Parameters
        ----------
        clip_uuid : str
            UUID of the clip to load, e.g. "3f2a8c1d-...".
        chunk_id : Optional[int]
            Index of the zip chunk that contains this clip (0-based).
            If None, chunk_id is resolved automatically via the dataset
            metadata parquet using find_chunk_for_clip().
        cache_dir : Optional[str]
            Directory for downloaded / extracted files.
            Defaults to <tempdir>/physai_av_cache.
        load_ego : bool
            Whether to also download and load the ego-motion parquet.

        Requires
        --------
          pip install huggingface_hub
          HF_TOKEN env var OR `huggingface-cli login`
          Agreement to the NVIDIA AV Dataset license on HuggingFace
        """
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as e:
            raise ImportError(
                "huggingface_hub is required for HF access. "
                "Install with: pip install huggingface_hub"
            ) from e

        import zipfile
        import tempfile

        repo_id   = "nvidia/PhysicalAI-Autonomous-Vehicles"
        token     = token or os.environ.get("HF_TOKEN")
        cache_dir = cache_dir or os.path.join(
            tempfile.gettempdir(), "physai_av_cache"
        )
        os.makedirs(cache_dir, exist_ok=True)

        # ── Resolve chunk_id if not supplied ─────────────────────────────
        if chunk_id is None:
            chunk_id = find_chunk_for_clip(clip_uuid, cache_dir=cache_dir, token=token)

        chunk_str = f"chunk_{chunk_id:04d}"

        # ── Download lidar zip for this chunk ─────────────────────────────
        # Dataset path: lidar/lidar_top_360fov/lidar_top_360fov.chunk_XXXX.zip
        lidar_zip_hf = f"lidar/lidar_top_360fov/lidar_top_360fov.{chunk_str}.zip"
        lidar_zip_local = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=lidar_zip_hf,
            cache_dir=cache_dir,
            token=token,
        )

        # ── Extract just the parquet for this clip_id ─────────────────────
        # Inside zip: {clip_id}.lidar_top_360fov.parquet
        lidar_parquet_local = _extract_clip_parquet(
            zip_path=lidar_zip_local,
            clip_uuid=clip_uuid,
            suffix=".lidar_top_360fov.parquet",
            dest_dir=cache_dir,
        )

        # ── Repeat for ego motion (optional) ─────────────────────────────
        # Dataset path: labels/egomotion/egomotion.chunk_XXXX.zip
        # Inside zip:   {clip_id}.egomotion.parquet
        ego_parquet_local = None
        if load_ego:
            try:
                ego_zip_hf = f"labels/egomotion/egomotion.{chunk_str}.zip"
                ego_zip_local = hf_hub_download(
                    repo_id=repo_id,
                    repo_type="dataset",
                    filename=ego_zip_hf,
                    cache_dir=cache_dir,
                    token=token,
                )
                ego_parquet_local = _extract_clip_parquet(
                    zip_path=ego_zip_local,
                    clip_uuid=clip_uuid,
                    suffix=".egomotion.parquet",
                    dest_dir=cache_dir,
                )
            except Exception as exc:
                import warnings
                warnings.warn(
                    f"Could not load ego motion for clip {clip_uuid}: {exc}",
                    stacklevel=2,
                )

        meta = ClipMetadata(clip_id=clip_uuid)
        return cls.from_file(lidar_parquet_local, ego_parquet_local, meta)

    @classmethod
    def synthetic(
        cls,
        n_spins: int = 20,
        points_per_spin: int = 30_000,
        n_rings: int = 128,
        seed: int = 42,
        scenario: str = "highway",
    ) -> "LidarLoader":
        """
        Generate synthetic LiDAR spins for testing and development.

        Scenarios: 'highway', 'urban', 'emergency_stop', 'lane_change'
        """
        rng = np.random.default_rng(seed)
        rows = []

        # Scenario-dependent obstacle placement
        scene_cfg = _SYNTHETIC_SCENES.get(scenario, _SYNTHETIC_SCENES["highway"])
        obstacles = _generate_obstacles(rng, scene_cfg)

        t0_us = 1_700_000_000_000_000  # arbitrary start timestamp (μs)
        dt_us = int(1_000_000 / LIDAR_HZ)

        for spin_idx in range(n_spins):
            ts = t0_us + spin_idx * dt_us
            pts = _synthetic_spin(
                rng, points_per_spin, n_rings, obstacles, scene_cfg
            )
            for pt in pts:
                rows.append({
                    "scan_index": spin_idx,
                    "timestamp": ts,
                    "sensor_timestamp": ts + rng.integers(-500, 500),
                    "x": pt[0],
                    "y": pt[1],
                    "z": pt[2],
                    "intensity": pt[3],
                    "ring": int(pt[4]),
                    "range": float(np.linalg.norm(pt[:3])),
                    "return_type": int(pt[5]),
                })

        df = pd.DataFrame(rows)
        # Ego motion: straight road with slight curvature for lane-change
        ego_rows = []
        for spin_idx in range(n_spins):
            ts = t0_us + spin_idx * dt_us
            speed = scene_cfg["ego_speed"]
            ego_x = speed * spin_idx / LIDAR_HZ
            ego_y = 0.0
            if scenario == "lane_change" and spin_idx > n_spins // 2:
                ego_y = (spin_idx - n_spins // 2) * 0.15
            ego_rows.append({
                "timestamp": ts,
                "x": ego_x, "y": ego_y, "z": 0.0,
                "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0,
                "vx": speed, "vy": 0.0, "vz": 0.0,
                "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
            })
        ego_df = pd.DataFrame(ego_rows)

        meta = ClipMetadata(
            clip_id=f"synthetic_{scenario}_{seed}",
            country="XX",
            platform_class="hyperion_8",
        )
        return cls(df, meta, ego_df)

    # ── Iteration ─────────────────────────────────────────────────────────────

    def _build_groups(self):
        if self._scan_groups is None:
            self._scan_groups = dict(tuple(self._df.groupby("scan_index")))

    def __len__(self) -> int:
        self._build_groups()
        return len(self._scan_groups)

    def iter_spins(
        self,
        start: int = 0,
        stop: Optional[int] = None,
        first_return_only: bool = True,
    ) -> Iterator[SpinFrame]:
        """
        Yield SpinFrame objects one at a time.

        Parameters
        ----------
        start : int
            First scan_index to yield (inclusive).
        stop : Optional[int]
            Last scan_index to yield (exclusive). None = all.
        first_return_only : bool
            If True, drop dual-return second echoes (return_type == 1).
        """
        self._build_groups()
        indices = sorted(self._scan_groups.keys())
        stop = stop or len(indices)

        for idx in indices[start:stop]:
            grp = self._scan_groups[idx]
            if first_return_only and "return_type" in grp.columns:
                grp = grp[grp["return_type"] == 0]

            pts = grp[["x", "y", "z", "intensity", "ring", "return_type"]].values.astype(
                np.float32
            )
            ts = int(grp["timestamp"].iloc[0])

            ego_pose = None
            if self._ego_df is not None:
                ego_pose = _interpolate_ego_pose(self._ego_df, ts)

            yield SpinFrame(
                scan_index=int(idx),
                timestamp_us=ts,
                points=pts,
                ego_pose=ego_pose,
            )

    def get_spin(self, scan_index: int) -> SpinFrame:
        """Return a single spin by index."""
        self._build_groups()
        if scan_index not in self._scan_groups:
            raise KeyError(f"scan_index {scan_index} not found in clip")
        grp = self._scan_groups[scan_index]
        pts = grp[["x", "y", "z", "intensity", "ring", "return_type"]].values.astype(
            np.float32
        )
        ts = int(grp["timestamp"].iloc[0])
        ego_pose = None
        if self._ego_df is not None:
            ego_pose = _interpolate_ego_pose(self._ego_df, ts)
        return SpinFrame(int(scan_index), ts, pts, ego_pose)


# ── Public helpers ────────────────────────────────────────────────────────────

def find_chunk_for_clip(
    clip_uuid: str,
    cache_dir: Optional[str] = None,
    token: Optional[str] = None,
) -> int:
    """
    Find which chunk number contains a given clip_id.

    The dataset metadata does not include a chunk_id column — clip_id is the
    index with only country/month/hour_of_day/platform_class as columns.
    Chunk membership is determined by scanning the calibration
    sensor_extrinsics chunk parquets, which have one row per clip and are
    small enough to download cheaply.  Results are cached in a local JSON
    file so the scan only runs once per machine.

    Parameters
    ----------
    clip_uuid : str
        clip_id exactly as it appears in the dataset index, e.g.
        "25cd4769-5dcf-4b53-a351-bf2c5deb6124".
    cache_dir : Optional[str]
        Directory for cached files. Defaults to <tempdir>/physai_av_cache.
    token : Optional[str]
        HuggingFace token. Falls back to HF_TOKEN env var if None.

    Returns
    -------
    int
        chunk number (0-based) whose zip contains this clip.

    Raises
    ------
    ValueError
        If the clip_id is not found in any scanned chunk.
    ImportError
        If huggingface_hub is not installed.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise ImportError(
            "huggingface_hub is required. Install with: pip install huggingface_hub"
        ) from e

    import json
    import tempfile

    token     = token or os.environ.get("HF_TOKEN")
    cache_dir = cache_dir or os.path.join(
        tempfile.gettempdir(), "physai_av_cache"
    )
    os.makedirs(cache_dir, exist_ok=True)

    repo_id   = "nvidia/PhysicalAI-Autonomous-Vehicles"
    index_path = os.path.join(cache_dir, "clip_chunk_index.json")

    # ── Fast path: local cache hit ────────────────────────────────────────
    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
        if clip_uuid in index:
            return int(index[clip_uuid])

    # ── Download clip_index.parquet (root-level, one file for all clips) ─
    # Columns: clip_is_valid, chunk, split — index: clip_id
    # This is the authoritative chunk mapping for the entire dataset.
    print("Downloading clip_index.parquet...")
    index_parquet = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename="clip_index.parquet",
        cache_dir=cache_dir,
        token=token,
    )

    clip_df = pd.read_parquet(index_parquet)

    # Persist a lightweight JSON cache for instant future lookups
    index = {
        str(cid): int(row["chunk"])
        for cid, row in clip_df.iterrows()
    }
    with open(index_path, "w") as f:
        json.dump(index, f)

    if clip_uuid not in index:
        raise ValueError(
            f"clip_id '{clip_uuid}' not found in clip_index.parquet.\n"
            f"Total clips in index: {len(index)}\n"
            "Possible reasons:\n"
            "  - The clip_id is incorrect\n"
            "  - Clips with a 'clipgt-' prefix are ground-truth annotation "
            "clips and do not exist as raw sensor data in this dataset\n"
            "  - clip_is_valid may be False for this clip\n"
            "Check the dataset README for the correct clip ID format."
        )

    return index[clip_uuid]


def _parse_chunk_number(hf_path: str) -> Optional[int]:
    """Extract the integer chunk number from a path like foo.chunk_0042.parquet."""
    import re
    m = re.search(r"chunk_(\d+)", hf_path)
    return int(m.group(1)) if m else None


# ── Private helpers ───────────────────────────────────────────────────────────

def _extract_clip_parquet(
    zip_path: str,
    clip_uuid: str,
    suffix: str,
    dest_dir: str,
) -> str:
    """
    Extract one parquet file for clip_uuid from a zip archive.

    Returns the local path to the extracted parquet.
    Skips extraction if the file already exists (cache hit).

    Parameters
    ----------
    zip_path : str
        Local path to the downloaded zip archive.
    clip_uuid : str
        UUID used to identify the correct entry inside the zip.
    suffix : str
        Expected filename suffix, e.g. ".lidar_top360_fov.parquet".
    dest_dir : str
        Directory to extract into.

    Raises
    ------
    FileNotFoundError
        If no entry matching clip_uuid is found inside the zip.
    """
    import zipfile

    # Fast path — already extracted from a previous run
    direct = os.path.join(dest_dir, f"{clip_uuid}{suffix}")
    if os.path.exists(direct):
        return direct

    with zipfile.ZipFile(zip_path, "r") as zf:
        entries  = zf.namelist()
        matching = [e for e in entries if clip_uuid in e and e.endswith(".parquet")]

        if not matching:
            raise FileNotFoundError(
                f"No parquet entry matching clip_uuid '{clip_uuid}' found in:\n"
                f"  {zip_path}\n"
                f"First few entries: {entries[:8]}"
            )

        # Extract to dest_dir, preserving any subdirectory inside the zip
        zf.extract(matching[0], path=dest_dir)
        extracted = os.path.join(dest_dir, matching[0])

        # Flatten to dest_dir/<clip_uuid><suffix> for a consistent cache path
        if extracted != direct:
            os.replace(extracted, direct)

    return direct


def _decode_draco_parquet(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Decode a Draco-encoded LiDAR parquet into a flat x/y/z/intensity/ring/... DataFrame.

    The nvidia/PhysicalAI-Autonomous-Vehicles dataset stores each spin as one
    row with columns:
        reference_timestamp       int64   microseconds
        draco_encoded_pointcloud  bytes   Draco-compressed point cloud

    The Draco payload decodes to N points with attributes:
        0: x          float32  metres (rig frame)
        1: y          float32
        2: z          float32
        3: intensity  float32  raw sensor units (normalised below)
        4: ring       int32    beam index
        5: return_type int32   0=first, 1=second

    Requires: pip install DracoPy
    """
    try:
        import DracoPy
    except ImportError as e:
        raise ImportError(
            "DracoPy is required to decode the NVIDIA dataset point clouds.\n"
            "Install with: pip install DracoPy"
        ) from e

    # Resolve column names robustly — itertuples mangles names with spaces/dots
    ts_col = next(
        (c for c in raw.columns if "timestamp" in c.lower()),
        raw.columns[0],
    )
    draco_col = next(
        (c for c in raw.columns if "draco" in c.lower() or "pointcloud" in c.lower()),
        None,
    )
    if draco_col is None:
        raise ValueError(
            f"Cannot find Draco column in LiDAR parquet.\n"
            f"Columns found: {list(raw.columns)}\n"
            f"Expected a column containing 'draco' or 'pointcloud'."
        )

    rows = []
    for spin_idx in range(len(raw)):
        ts   = int(raw[ts_col].iloc[spin_idx])
        blob = raw[draco_col].iloc[spin_idx]

        # DracoPy returns a PointCloud object with .points (N,3) and
        # .generic_attributes list for extra attribute arrays
        try:
            pc = DracoPy.decode(bytes(blob))
        except Exception as exc:
            raise RuntimeError(
                f"Draco decode failed for spin {spin_idx} "
                f"(timestamp {ts}): {exc}"
            ) from exc

        pts = np.asarray(pc.points, dtype=np.float32)   # (N, 3) x y z

        # Extra attributes beyond xyz are stored in generic_attributes
        n_pts = len(pts)
        attrs = getattr(pc, "generic_attributes", [])

        if len(attrs) >= 3:
            intensity   = np.asarray(attrs[0], dtype=np.float32).reshape(-1)
            ring        = np.asarray(attrs[1], dtype=np.float32).reshape(-1)
            return_type = np.asarray(attrs[2], dtype=np.float32).reshape(-1)
        elif len(attrs) == 2:
            intensity   = np.asarray(attrs[0], dtype=np.float32).reshape(-1)
            ring        = np.asarray(attrs[1], dtype=np.float32).reshape(-1)
            return_type = np.zeros(n_pts, dtype=np.float32)
        elif len(attrs) == 1:
            intensity   = np.asarray(attrs[0], dtype=np.float32).reshape(-1)
            ring        = np.zeros(n_pts, dtype=np.float32)
            return_type = np.zeros(n_pts, dtype=np.float32)
        else:
            intensity   = np.zeros(n_pts, dtype=np.float32)
            ring        = np.zeros(n_pts, dtype=np.float32)
            return_type = np.zeros(n_pts, dtype=np.float32)

        # Normalise intensity to [0, 1] if it looks like raw sensor units
        if intensity.max() > 1.0:
            intensity = intensity / intensity.max()

        chunk = pd.DataFrame({
            "scan_index":   spin_idx,
            "timestamp":    ts,
            "x":            pts[:, 0],
            "y":            pts[:, 1],
            "z":            pts[:, 2],
            "intensity":    intensity,
            "ring":         ring.astype(np.int32),
            "return_type":  return_type.astype(np.int32),
            "range":        np.linalg.norm(pts, axis=1).astype(np.float32),
        })
        rows.append(chunk)

    return pd.concat(rows, ignore_index=True)


def _normalise_ego_df(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise an egomotion parquet to the standard ego_df schema:
        timestamp, x, y, z, qx, qy, qz, qw, vx, vy, vz, roll, pitch, yaw

    The dataset egomotion parquet uses:
        reference_timestamp  int64  microseconds
        draco_encoded_*      may not exist — egomotion is usually flat
        tx, ty, tz           translation  (alternative to x/y/z)
        qx, qy, qz, qw       quaternion   (same naming)
        vx, vy, vz           velocity     (same naming, may be absent)
    """
    col = set(raw.columns)

    # Rename reference_timestamp -> timestamp if needed
    if "reference_timestamp" in col and "timestamp" not in col:
        raw = raw.rename(columns={"reference_timestamp": "timestamp"})

    # Rename tx/ty/tz -> x/y/z if needed
    if "tx" in col and "x" not in col:
        raw = raw.rename(columns={"tx": "x", "ty": "y", "tz": "z"})

    # Fill missing velocity / attitude columns with zeros
    for c in ("vx", "vy", "vz", "roll", "pitch", "yaw"):
        if c not in raw.columns:
            raw[c] = 0.0

    # Ensure timestamp is sorted for interpolation
    raw = raw.sort_values("timestamp").reset_index(drop=True)

    required = {"timestamp", "x", "y", "z", "qx", "qy", "qz", "qw"}
    missing  = required - set(raw.columns)
    if missing:
        raise ValueError(
            f"Egomotion parquet is missing required columns: {missing}\n"
            f"Found: {list(raw.columns)}"
        )

    return raw


def _validate_lidar_df(df: pd.DataFrame) -> None:
    required = {"scan_index", "x", "y", "z", "intensity"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"LiDAR parquet is missing required columns: {missing}\n"
            f"Found: {list(df.columns)}"
        )


def _interpolate_ego_pose(ego_df: pd.DataFrame, ts_us: int) -> np.ndarray:
    """Linear interpolation of ego pose at an arbitrary timestamp."""
    ts_col = ego_df["timestamp"].values
    idx = np.searchsorted(ts_col, ts_us)
    if idx == 0:
        row = ego_df.iloc[0]
    elif idx >= len(ego_df):
        row = ego_df.iloc[-1]
    else:
        t0, t1 = ts_col[idx - 1], ts_col[idx]
        alpha = (ts_us - t0) / max(t1 - t0, 1)
        r0 = ego_df.iloc[idx - 1][["x", "y", "z", "qx", "qy", "qz", "qw"]].values
        r1 = ego_df.iloc[idx][["x", "y", "z", "qx", "qy", "qz", "qw"]].values
        return (r0 + alpha * (r1 - r0)).astype(np.float64)
    return row[["x", "y", "z", "qx", "qy", "qz", "qw"]].values.astype(np.float64)


# ── Synthetic scene generation ────────────────────────────────────────────────

_SYNTHETIC_SCENES = {
    "highway": {
        "ego_speed": 28.0,
        "obstacle_count": 3,
        "obstacle_ranges": (20, 80),
        "lane_width": 3.6,
        "ground_z": -1.8,
    },
    "urban": {
        "ego_speed": 8.0,
        "obstacle_count": 8,
        "obstacle_ranges": (5, 35),
        "lane_width": 3.2,
        "ground_z": -1.8,
    },
    "emergency_stop": {
        "ego_speed": 22.0,
        "obstacle_count": 2,
        "obstacle_ranges": (6, 12),
        "lane_width": 3.6,
        "ground_z": -1.8,
    },
    "lane_change": {
        "ego_speed": 18.0,
        "obstacle_count": 4,
        "obstacle_ranges": (25, 60),
        "lane_width": 3.6,
        "ground_z": -1.8,
    },
}


def _generate_obstacles(rng, cfg):
    n = cfg["obstacle_count"]
    r_min, r_max = cfg["obstacle_ranges"]
    obstacles = []
    for _ in range(n):
        r = rng.uniform(r_min, r_max)
        az = rng.uniform(-0.4, 0.4)  # radians from forward
        obstacles.append({
            "x": r * math.cos(az),
            "y": r * math.sin(az),
            "z_base": cfg["ground_z"] + rng.uniform(0, 0.1),
            "height": rng.uniform(1.4, 2.1),
            "width":  rng.uniform(1.8, 2.5),
        })
    return obstacles


def _synthetic_spin(rng, n_pts, n_rings, obstacles, cfg) -> list:
    """Generate a single spin of synthetic LiDAR points."""
    pts = []
    ground_z = cfg["ground_z"]

    # Ground plane points (majority)
    n_ground = int(n_pts * 0.65)
    ranges_g = rng.uniform(1.0, 80.0, n_ground)
    azimuths_g = rng.uniform(-math.pi, math.pi, n_ground)
    pts_g_x = ranges_g * np.cos(azimuths_g)
    pts_g_y = ranges_g * np.sin(azimuths_g)
    pts_g_z = np.full(n_ground, ground_z) + rng.normal(0, 0.02, n_ground)
    intens_g = rng.uniform(0.05, 0.35, n_ground)
    rings_g = rng.integers(0, n_rings // 4, n_ground)
    for i in range(n_ground):
        pts.append([pts_g_x[i], pts_g_y[i], pts_g_z[i],
                    float(intens_g[i]), float(rings_g[i]), 0.0])

    # Obstacle points
    for obs in obstacles:
        n_obs = int(n_pts * 0.05)
        for _ in range(n_obs):
            ox = obs["x"] + rng.uniform(-obs["width"] / 2, obs["width"] / 2)
            oy = obs["y"] + rng.uniform(-obs["width"] / 4, obs["width"] / 4)
            oz = obs["z_base"] + rng.uniform(0, obs["height"])
            ring = int(rng.uniform(n_rings // 4, n_rings - 1))
            pts.append([ox, oy, oz, float(rng.uniform(0.4, 0.9)), float(ring), 0.0])

    # Fill remainder with sky / no-return noise (very sparse)
    while len(pts) < n_pts:
        r = rng.uniform(0.5, RANGE_MAX_M)
        az = rng.uniform(-math.pi, math.pi)
        el = rng.uniform(-0.2, 0.4)
        x_ = r * math.cos(el) * math.cos(az)
        y_ = r * math.cos(el) * math.sin(az)
        z_ = r * math.sin(el)
        ring = int(rng.integers(0, n_rings))
        pts.append([x_, y_, z_, float(rng.uniform(0.0, 0.15)), float(ring), 0.0])

    return pts[:n_pts]