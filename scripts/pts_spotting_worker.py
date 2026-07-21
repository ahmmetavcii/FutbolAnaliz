"""PTS-baseline (E2E-Spot) out-of-process spotting worker.

Runs inside ``sn-pts-baseline`` so ai-dev torch is untouched. Emits JSON
spotting candidates; never invents events when the model predicts background.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


V2_LABELS = sorted(
    [
        "Penalty",
        "Kick-off",
        "Goal",
        "Substitution",
        "Offside",
        "Shots on target",
        "Shots off target",
        "Clearance",
        "Ball out of play",
        "Throw-in",
        "Foul",
        "Indirect free-kick",
        "Direct free-kick",
        "Corner",
        "Yellow card",
        "Red card",
        "Yellow->red card",
    ]
)

IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-frames", type=int, default=0, help="0 = full clip_len batches")
    parser.add_argument("--score-threshold", type=float, default=0.35)
    return parser.parse_args()


def _load_frames(video: Path, count: int) -> tuple[np.ndarray, list[int], float]:
    capture = cv2.VideoCapture(str(video))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
    frames: list[np.ndarray] = []
    frame_ids: list[int] = []
    index = 0
    while len(frames) < count:
        ok, image = capture.read()
        if not ok:
            break
        resized = cv2.resize(image, (398, 224))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
        frames.append(rgb.transpose(2, 0, 1))
        frame_ids.append(index)
        index += 1
    capture.release()
    if not frames:
        return np.zeros((0, 3, 224, 398), dtype=np.float32), [], fps
    return np.stack(frames).astype(np.float32), frame_ids, fps


def main() -> int:
    args = parse_args()
    repo = Path(args.repo_root)
    ckpt_dir = Path(args.checkpoint_dir)
    sys.path.insert(0, str(repo))

    from train_e2e import E2EModel
    from util.dataset import load_classes

    cfg = json.loads((ckpt_dir / "config.json").read_text(encoding="utf-8"))
    class_file = Path(args.output).with_suffix(".classes.txt")
    class_file.write_text("\n".join(V2_LABELS), encoding="utf-8")
    classes = load_classes(str(class_file))
    id_to_name = {value: key for key, value in classes.items()}

    device = args.device if torch.cuda.is_available() else "cpu"
    model = E2EModel(
        len(classes) + 1,
        cfg["feature_arch"],
        cfg["temporal_arch"],
        clip_len=cfg["clip_len"],
        modality=cfg["modality"],
        device=device,
    )
    state = torch.load(ckpt_dir / "checkpoint_088.pt", map_location="cpu")
    model.load(state)

    clip_len = int(cfg["clip_len"])
    max_frames = int(args.max_frames) if args.max_frames > 0 else clip_len
    array, frame_ids, fps = _load_frames(Path(args.video), max_frames)
    candidates: list[dict] = []
    status = "EMPTY_INPUT"
    if array.shape[0] > 0:
        # Pad to clip_len if needed (short clips).
        if array.shape[0] < clip_len:
            pad = np.repeat(array[-1:], clip_len - array.shape[0], axis=0)
            array = np.concatenate([array, pad], axis=0)
            frame_ids = frame_ids + [frame_ids[-1]] * (clip_len - len(frame_ids))
        seq = torch.from_numpy(array[:clip_len]).float()
        _pred_cls, pred_scores = model.predict(seq, use_amp=False)
        scores_t = pred_scores[0]
        if hasattr(scores_t, "detach"):
            scores = scores_t.detach().cpu().numpy()
        else:
            scores = np.asarray(scores_t)
        for index in range(min(len(frame_ids), scores.shape[0])):
            class_id = int(np.argmax(scores[index, 1:]) + 1)  # skip background=0
            score = float(scores[index, class_id])
            if score < float(args.score_threshold):
                continue
            name = id_to_name.get(class_id, str(class_id))
            frame_id = int(frame_ids[index])
            candidates.append(
                {
                    "event_type": name,
                    "timestamp": frame_id / max(fps, 1e-6),
                    "frame_id": frame_id,
                    "confidence": score,
                    "source_model": "PTS-baseline/E2E-Spot",
                    "temporal_window": [frame_id, frame_id],
                    "class_id": class_id,
                }
            )
        status = "PASS"

    payload = {
        "status": status,
        "source_model": "PTS-baseline/E2E-Spot",
        "checkpoint": str(ckpt_dir / "checkpoint_088.pt"),
        "device": device,
        "fps": fps,
        "frames_processed": len(frame_ids),
        "candidates": candidates,
    }
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
