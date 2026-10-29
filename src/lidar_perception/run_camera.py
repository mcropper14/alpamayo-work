"""
run_camera.py
─────────────
Download and extract a single camera mp4 from HuggingFace.
Run this to verify camera loading works before running visualise.py.

Usage
─────
  set HF_TOKEN=hf_...
  python run_camera.py --clip-id <uuid>

  # Specify chunk explicitly to skip clip_index lookup
  python run_camera.py --clip-id <uuid> --chunk 588

  # Different camera
  python run_camera.py --clip-id <uuid> --camera camera_front_tele_30fov
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from physical_ai_av_lidar.camera_loader import CameraLoader, CAMERA_NAMES

def main():
    p = argparse.ArgumentParser(description="Download a single camera mp4 from HuggingFace")
    p.add_argument("--clip-id",   required=True, metavar="UUID")
    p.add_argument("--camera",    default="camera_front_wide_120fov",
                   choices=CAMERA_NAMES, metavar="CAM")
    p.add_argument("--chunk",     type=int, default=None, metavar="N")
    p.add_argument("--cache-dir", default="./cache", metavar="DIR")
    p.add_argument("--hf-token",  default=None, metavar="TOKEN")
    args = p.parse_args()

    token = args.hf_token or os.environ.get("HF_TOKEN")

    print(f"Clip:    {args.clip_id}")
    print(f"Camera:  {args.camera}")
    print(f"Cache:   {args.cache_dir}")
    print()

    loader = CameraLoader.from_hf(
        clip_uuid = args.clip_id,
        camera    = args.camera,
        chunk_id  = args.chunk,
        cache_dir = args.cache_dir,
        token     = token,
    )

    print(f"\nSuccess!")
    print(f"Video saved to: {loader.video_path}")

    # Quick sanity check — extract 3 frames
    print("\nExtracting 3 test frames...")
    frames = loader.extract_frames_b64(n_frames=3, width=320, height=180)
    print(f"Got {len(frames)} frames, first frame size: {len(frames[0])} bytes (base64)")
    print("\nCamera loader is working. You can now run visualise.py.")

if __name__ == "__main__":
    main()