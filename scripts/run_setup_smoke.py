#!/usr/bin/env python3
"""Run lightweight CUDA, video, detection, and tracking smoke tests."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import requests
import torch
from fastapi import FastAPI
from ultralytics import YOLO


WORKSPACE = Path("/home/ahmet/workspace")
MODELS = Path("/home/ahmet/models")
STAMP = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
RUN = WORKSPACE / "runs" / f"setup_smoke_{STAMP}"
STAGING = WORKSPACE / "staging"
MODEL = MODELS / "yolo11n.pt"
IMAGE = STAGING / "ultralytics_bus.jpg"
VIDEO = STAGING / "synthetic_bus_pan.mp4"
MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt"
IMAGE_URL = "https://ultralytics.com/images/bus.jpg"


def download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        temporary = target.with_suffix(target.suffix + ".part")
        with requests.get(url, stream=True, timeout=120) as response:
            response.raise_for_status()
            with temporary.open("wb") as stream:
                for chunk in response.iter_content(1024 * 1024):
                    stream.write(chunk)
        temporary.replace(target)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_video(image_path: Path, target: Path) -> tuple[int, float]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"OpenCV could not read {image_path}")
    image = cv2.resize(image, (640, 360))
    fps = 15.0
    frames = 45
    writer = cv2.VideoWriter(
        str(target), cv2.VideoWriter_fourcc(*"mp4v"), fps, (640, 360)
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV VideoWriter failed")
    for index in range(frames):
        shift = int(8 * np.sin(index / 8))
        frame = np.roll(image, shift, axis=1)
        writer.write(frame)
    writer.release()
    capture = cv2.VideoCapture(str(target))
    read_frames = 0
    while True:
        ok, _ = capture.read()
        if not ok:
            break
        read_frames += 1
    capture.release()
    if read_frames != frames:
        raise RuntimeError(f"OpenCV read {read_frames}/{frames} frames")
    return frames, fps


def main() -> int:
    RUN.mkdir(parents=True, exist_ok=False)
    download(IMAGE_URL, IMAGE)
    download(MODEL_URL, MODEL)
    frames, source_fps = create_video(IMAGE, VIDEO)
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(VIDEO)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    (RUN / "ffprobe.json").write_text(probe.stdout)
    subprocess.run(["ffmpeg", "-version"], check=True, stdout=subprocess.PIPE)

    torch.cuda.reset_peak_memory_stats()
    model = YOLO(str(MODEL))
    predict_start = time.perf_counter()
    predictions = model.predict(
        source=str(IMAGE),
        device=0,
        imgsz=640,
        batch=1,
        half=True,
        project=str(RUN),
        name="image_detection",
        save=True,
        verbose=False,
    )
    predict_seconds = time.perf_counter() - predict_start
    detections = sum(len(result.boxes) for result in predictions)
    if detections < 1:
        raise RuntimeError("YOLO image smoke returned no detections")

    track_start = time.perf_counter()
    tracked_frames = 0
    tracked_detections = 0
    tracking_results = model.track(
        source=str(VIDEO),
        tracker="bytetrack.yaml",
        device=0,
        imgsz=640,
        batch=1,
        half=True,
        stream=True,
        project=str(RUN),
        name="video_tracking",
        save=True,
        verbose=False,
    )
    for result in tracking_results:
        tracked_frames += 1
        tracked_detections += len(result.boxes)
    torch.cuda.synchronize()
    track_seconds = time.perf_counter() - track_start
    if tracked_frames != frames or tracked_detections < 1:
        raise RuntimeError(
            f"Tracking smoke incomplete: frames={tracked_frames}, detections={tracked_detections}"
        )

    table_path = RUN / "parquet_smoke.parquet"
    pd.DataFrame(
        [{"frame_id": 0, "timestamp_ms": 0, "detections": detections}]
    ).to_parquet(table_path, index=False)
    if pq.read_table(table_path).num_rows != 1:
        raise RuntimeError("Parquet roundtrip failed")

    from SoccerNet.Downloader import SoccerNetDownloader

    _ = SoccerNetDownloader(LocalDirectory="/mnt/c/football_data/datasets/soccernet")
    _ = FastAPI(title="football setup smoke")
    summary = {
        "timestamp": dt.datetime.now().astimezone().isoformat(),
        "status": "PASS",
        "model": str(MODEL),
        "model_sha256": sha256(MODEL),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "peak_vram_bytes": torch.cuda.max_memory_allocated(),
        "image_detections": detections,
        "image_runtime_seconds": predict_seconds,
        "video_frames": tracked_frames,
        "video_source_fps": source_fps,
        "video_runtime_seconds": track_seconds,
        "processing_fps": tracked_frames / track_seconds,
        "tracked_detections": tracked_detections,
        "source_video": str(VIDEO),
        "run_directory": str(RUN),
    }
    (RUN / "smoke_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
