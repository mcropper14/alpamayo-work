"""
camera_loader.py
────────────────────────────────────────────────────────────────────────────
Loads camera video files from the nvidia/PhysicalAI-Autonomous-Vehicles
dataset on HuggingFace, mirroring the LidarLoader interface.

Dataset camera structure
────────────────────────
  camera/
  ├── camera_front_wide_120fov/
  │   └── camera_front_wide_120fov.chunk_XXXX.zip
  │       └── {clip_id}.camera_front_wide_120fov.mp4
  ├── camera_front_tele_30fov/
  │   └── camera_front_tele_30fov.chunk_XXXX.zip
  │       └── {clip_id}.camera_front_tele_30fov.mp4
  └── ...

The dataset runs cameras at ~12 Hz (not the same as LiDAR's 10 Hz).
Frame extraction is aligned to LiDAR spin timestamps when ego motion
timestamps are available, otherwise evenly sampled.

Usage
─────
  from physical_ai_av_lidar.camera_loader import CameraLoader, CAMERA_NAMES

  # Load front wide camera for a clip
  cam = CameraLoader.from_hf(
      clip_uuid = "25cd4769-5dcf-4b53-a351-bf2c5deb6124",
      camera    = "camera_front_wide_120fov",
      chunk_id  = 0,            # optional — resolved from clip_index if omitted
      cache_dir = "./cache",
  )

  # Extract frames aligned to LiDAR spin timestamps (microseconds)
  frames_b64 = cam.extract_frames_b64(
      timestamps_us = [1700000000000, 1700000100000, ...],
      width=320, height=180, quality=55,
  )

  # Or just extract N evenly-spaced frames
  frames_b64 = cam.extract_frames_b64(n_frames=200, width=320, height=180)

Requires
────────
  pip install opencv-python huggingface_hub
"""

from __future__ import annotations

import base64
import os
import tempfile
import warnings
import zipfile
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .loader import _extract_clip_parquet, find_chunk_for_clip

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

REPO_ID = "nvidia/PhysicalAI-Autonomous-Vehicles"

# All camera sensor names in the dataset
CAMERA_NAMES = [
    "camera_front_wide_120fov",
    "camera_front_tele_30fov",
    "camera_front_left_120fov",
    "camera_front_right_120fov",
    "camera_rear_left_70fov",
    "camera_rear_right_70fov",
    "camera_rear_tele_30fov",
]

# Human-readable display labels
CAMERA_LABELS = {
    "camera_front_wide_120fov":  "FRONT WIDE 120°",
    "camera_front_tele_30fov":   "FRONT TELE 30°",
    "camera_front_left_120fov":  "FRONT LEFT",
    "camera_front_right_120fov": "FRONT RIGHT",
    "camera_rear_left_70fov":    "REAR LEFT 70°",
    "camera_rear_right_70fov":   "REAR RIGHT 70°",
    "camera_rear_tele_30fov":    "REAR TELE 30°",
}

# Approximate camera frame rate in the dataset
CAMERA_HZ = 12.0


# ─────────────────────────────────────────────────────────────────────────────
# CameraLoader
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CameraLoader:
    """
    Provides access to one camera's video for a single clip.

    Attributes
    ----------
    video_path : str
        Local path to the extracted mp4 file.
    clip_id : str
        Dataset clip UUID.
    camera_name : str
        Sensor name e.g. "camera_front_wide_120fov".
    """
    video_path:  str
    clip_id:     str
    camera_name: str

    # ── Constructors ──────────────────────────────────────────────────────

    @classmethod
    def from_hf(
        cls,
        clip_uuid:   str,
        camera:      str = "camera_front_wide_120fov",
        chunk_id:    Optional[int] = None,
        cache_dir:   Optional[str] = None,
        token:       Optional[str] = None,
    ) -> "CameraLoader":
        """
        Download and cache a camera mp4 from HuggingFace.

        Parameters
        ----------
        clip_uuid : str
            Dataset clip ID (bare UUID, no clipgt- prefix).
        camera : str
            Camera sensor name. One of CAMERA_NAMES.
        chunk_id : Optional[int]
            Zip chunk index. Resolved automatically from clip_index.parquet
            if not supplied.
        cache_dir : Optional[str]
            Local cache directory. Defaults to <tempdir>/physai_av_cache.
        token : Optional[str]
            HuggingFace token. Falls back to HF_TOKEN env var.

        Requires
        ────────
          pip install huggingface_hub
          HF_TOKEN env var or huggingface-cli login
          NVIDIA AV Dataset license accepted on HuggingFace
        """
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ImportError(
                "huggingface_hub required. Install with: pip install huggingface_hub"
            ) from exc

        if camera not in CAMERA_NAMES:
            raise ValueError(
                f"Unknown camera '{camera}'. Valid options:\n"
                + "\n".join(f"  {c}" for c in CAMERA_NAMES)
            )

        token     = token or os.environ.get("HF_TOKEN")
        cache_dir = cache_dir or os.path.join(
            tempfile.gettempdir(), "physai_av_cache"
        )
        os.makedirs(cache_dir, exist_ok=True)

        # Resolve chunk_id if not provided
        if chunk_id is None:
            chunk_id = find_chunk_for_clip(clip_uuid, cache_dir=cache_dir, token=token)

        chunk_str = f"chunk_{chunk_id:04d}"

        # Dataset path: camera/<camera_name>/<camera_name>.chunk_XXXX.zip
        zip_hf_path = f"camera/{camera}/{camera}.{chunk_str}.zip"

        print(f"  Downloading {camera} zip for chunk {chunk_id}...")
        zip_local = hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=zip_hf_path,
            cache_dir=cache_dir,
            token=token,
        )

        # Extract just the mp4 for this clip
        # Inside zip: {clip_id}.{camera_name}.mp4
        video_local = _extract_clip_video(
            zip_path   = zip_local,
            clip_uuid  = clip_uuid,
            camera     = camera,
            dest_dir   = cache_dir,
        )

        return cls(
            video_path  = video_local,
            clip_id     = clip_uuid,
            camera_name = camera,
        )

    @classmethod
    def from_file(cls, video_path: str, camera_name: str = "camera_front_wide_120fov") -> "CameraLoader":
        """Load from a local mp4 file."""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        clip_id = os.path.splitext(os.path.basename(video_path))[0].split(".")[0]
        return cls(video_path=video_path, clip_id=clip_id, camera_name=camera_name)

    # ── Frame extraction ──────────────────────────────────────────────────

    def extract_frames_b64(
        self,
        timestamps_us: Optional[List[int]] = None,
        n_frames:      Optional[int] = None,
        width:         int  = 320,
        height:        int  = 180,
        quality:       int  = 55,
    ) -> List[str]:
        """
        Extract frames from the video as base64-encoded JPEG strings.

        Supply either:
          timestamps_us  – list of microsecond timestamps to align to
                           (use LiDAR spin timestamps for sync)
          n_frames       – extract this many evenly-spaced frames

        Returns list of base64 JPEG strings, one per requested frame.

        Requires: pip install opencv-python
        """
        try:
            import cv2
        except ImportError as exc:
            raise ImportError(
                "opencv-python required for frame extraction.\n"
                "Install with: pip install opencv-python"
            ) from exc

        cap   = cv2.VideoCapture(self.video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps   = cap.get(cv2.CAP_PROP_FPS) or CAMERA_HZ

        if total <= 0:
            cap.release()
            n = n_frames or (len(timestamps_us) if timestamps_us else 1)
            return [_blank_b64(width, height)] * n

        # Determine which frame indices to extract
        if timestamps_us is not None:
            # Convert timestamps to frame indices
            # Assumes video starts at timestamps_us[0]
            t0  = timestamps_us[0]
            indices = []
            for ts in timestamps_us:
                dt_s    = (ts - t0) / 1e6
                frame_i = int(dt_s * fps)
                frame_i = max(0, min(total - 1, frame_i))
                indices.append(frame_i)
        elif n_frames is not None:
            indices = np.linspace(0, total - 1, n_frames, dtype=int).tolist()
        else:
            raise ValueError("Supply either timestamps_us or n_frames.")

        frames   = []
        last_b64 = _blank_b64(width, height)

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                frames.append(last_b64)
                continue
            frame    = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            ok, buf  = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if ok:
                last_b64 = base64.b64encode(buf.tobytes()).decode("ascii")
            frames.append(last_b64)

        cap.release()
        return frames

    def __repr__(self) -> str:
        return (f"CameraLoader(camera={self.camera_name!r}, "
                f"clip={self.clip_id!r}, path={self.video_path!r})")


# ─────────────────────────────────────────────────────────────────────────────
# Multi-camera convenience loader
# ─────────────────────────────────────────────────────────────────────────────

class MultiCameraLoader:
    """
    Loads multiple cameras for a single clip in one call.

    Usage
    ─────
      ml = MultiCameraLoader.from_hf(
          clip_uuid = "25cd4769-...",
          cameras   = ["camera_front_wide_120fov",
                       "camera_front_left_120fov",
                       "camera_front_right_120fov"],
          chunk_id  = 0,
      )

      # Extract frames for all cameras aligned to LiDAR spin timestamps
      frames = ml.extract_all_b64(
          timestamps_us = spin_timestamps,
          width=320, height=180, quality=55,
      )
      # frames = {
      #   "camera_front_wide_120fov":  [b64, b64, ...],
      #   "camera_front_left_120fov":  [b64, b64, ...],
      #   "camera_front_right_120fov": [b64, b64, ...],
      # }
    """

    def __init__(self, loaders: dict):
        self._loaders = loaders  # {camera_name: CameraLoader}

    @classmethod
    def from_hf(
        cls,
        clip_uuid:  str,
        cameras:    Optional[List[str]] = None,
        chunk_id:   Optional[int] = None,
        cache_dir:  Optional[str] = None,
        token:      Optional[str] = None,
    ) -> "MultiCameraLoader":
        """
        Download multiple camera zips for a clip.

        Parameters
        ----------
        cameras : Optional[List[str]]
            Which cameras to load. Defaults to the three front-facing cameras.
            Pass CAMERA_NAMES to load all seven.
        """
        if cameras is None:
            cameras = [
                "camera_front_wide_120fov",
                "camera_front_left_120fov",
                "camera_front_right_120fov",
            ]

        token     = token or os.environ.get("HF_TOKEN")
        cache_dir = cache_dir or os.path.join(
            tempfile.gettempdir(), "physai_av_cache"
        )

        # Resolve chunk_id once, reuse for all cameras
        if chunk_id is None:
            chunk_id = find_chunk_for_clip(clip_uuid, cache_dir=cache_dir, token=token)

        loaders = {}
        for cam in cameras:
            try:
                loaders[cam] = CameraLoader.from_hf(
                    clip_uuid = clip_uuid,
                    camera    = cam,
                    chunk_id  = chunk_id,
                    cache_dir = cache_dir,
                    token     = token,
                )
            except Exception as exc:
                warnings.warn(
                    f"Could not load {cam} for clip {clip_uuid}: {exc}",
                    stacklevel=2,
                )

        return cls(loaders)

    def extract_all_b64(
        self,
        timestamps_us: Optional[List[int]] = None,
        n_frames:      Optional[int] = None,
        width:  int = 320,
        height: int = 180,
        quality: int = 55,
    ) -> dict:
        """
        Extract frames from all loaded cameras.
        Returns {camera_name: [b64_frame, ...]} dict.
        """
        result = {}
        for cam_name, loader in self._loaders.items():
            try:
                result[cam_name] = loader.extract_frames_b64(
                    timestamps_us = timestamps_us,
                    n_frames      = n_frames,
                    width         = width,
                    height        = height,
                    quality       = quality,
                )
            except Exception as exc:
                warnings.warn(f"Frame extraction failed for {cam_name}: {exc}", stacklevel=2)
                n = n_frames or (len(timestamps_us) if timestamps_us else 0)
                result[cam_name] = [_blank_b64(width, height)] * n
        return result

    @property
    def available_cameras(self) -> List[str]:
        return list(self._loaders.keys())

    def __repr__(self) -> str:
        return f"MultiCameraLoader(cameras={self.available_cameras})"


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_clip_video(
    zip_path:  str,
    clip_uuid: str,
    camera:    str,
    dest_dir:  str,
) -> str:
    """
    Extract one mp4 file for clip_uuid from a camera zip archive.
    Returns local path to extracted mp4. Caches to avoid re-extraction.
    """
    # Expected filename inside zip: {clip_id}.{camera_name}.mp4
    direct = os.path.join(dest_dir, f"{clip_uuid}.{camera}.mp4")
    if os.path.exists(direct):
        return direct

    with zipfile.ZipFile(zip_path, "r") as zf:
        entries  = zf.namelist()
        matching = [
            e for e in entries
            if clip_uuid in e and (e.endswith(".mp4") or e.endswith(".avi"))
        ]

        if not matching:
            raise FileNotFoundError(
                f"No video entry for clip '{clip_uuid}' in:\n"
                f"  {zip_path}\n"
                f"First few entries: {entries[:6]}\n"
                f"Check that this camera is available for the clip using "
                f"the sensor_presence.parquet metadata."
            )

        zf.extract(matching[0], path=dest_dir)
        extracted = os.path.join(dest_dir, matching[0])

        if extracted != direct:
            os.replace(extracted, direct)

    return direct


def _blank_b64(width: int, height: int) -> str:
    """Return a dark placeholder frame as base64 JPEG."""
    try:
        import cv2
        blank   = np.full((height, width, 3), 18, dtype=np.uint8)
        ok, buf = cv2.imencode(".jpg", blank, [cv2.IMWRITE_JPEG_QUALITY, 40])
        return base64.b64encode(buf.tobytes()).decode("ascii") if ok else ""
    except ImportError:
        return ""