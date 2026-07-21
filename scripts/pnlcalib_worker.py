"""Out-of-process PnLCalib keypoint extraction worker.

Runs inside an isolated environment that has PnLCalib's dependencies
(torch, shapely, opencv). It does NOT import football_analytics.

For each requested frame it emits the detected ground-plane keypoint
correspondences: image points in pixels and pitch points in canonical
0..105 / 0..68 metre coordinates. Homography fitting and every validity
gate stay in the main pipeline, so nothing here can bypass validation.

Usage (executed by the calibration stage via subprocess):
    python pnlcalib_worker.py \
        --pnlcalib-root /path/to/pnlcalib \
        --weights-kp SV_kp.pth --weights-line SV_lines.pth \
        --video input.mp4 --frame-ids 0,5,10 \
        --output result.json [--device cuda:0]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pnlcalib-root", required=True)
    parser.add_argument("--weights-kp", required=True)
    parser.add_argument("--weights-line", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--frame-ids", required=True, help="comma separated frame ids")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--kp-threshold", type=float, default=0.3434)
    parser.add_argument("--line-threshold", type=float, default=0.7867)
    parser.add_argument("--min-keypoints", type=int, default=6)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pnl_root = Path(args.pnlcalib_root)
    sys.path.insert(0, str(pnl_root))

    import cv2
    import torch
    import yaml
    import torchvision.transforms as T
    import torchvision.transforms.functional as f
    from PIL import Image

    from model.cls_hrnet import get_cls_net
    from model.cls_hrnet_l import get_cls_net as get_cls_net_l
    from utils.utils_calib import FramebyFrameCalib
    from utils.utils_heatmap import (
        complete_keypoints,
        coords_to_dict,
        get_keypoints_from_heatmap_batch_maxpool,
        get_keypoints_from_heatmap_batch_maxpool_l,
    )

    device = args.device if torch.cuda.is_available() else "cpu"

    cfg = yaml.safe_load((pnl_root / "config" / "hrnetv2_w48.yaml").read_text())
    cfg_l = yaml.safe_load((pnl_root / "config" / "hrnetv2_w48_l.yaml").read_text())

    model = get_cls_net(cfg)
    model.load_state_dict(torch.load(args.weights_kp, map_location=device))
    model.to(device)
    model.eval()

    model_l = get_cls_net_l(cfg_l)
    model_l.load_state_dict(torch.load(args.weights_line, map_location=device))
    model_l.to(device)
    model_l.eval()

    resize = T.Resize((540, 960))

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"cannot open video: {args.video}", file=sys.stderr)
        return 2
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    calib = FramebyFrameCalib(iwidth=width, iheight=height, denormalize=True)

    frame_ids = sorted({int(v) for v in args.frame_ids.split(",") if v.strip()})
    frames_out = []
    started = time.time()
    for frame_id in frame_ids:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, frame_bgr = cap.read()
        if not ok:
            frames_out.append(
                {"frame_id": frame_id, "status": "read_failed", "keypoints": 0}
            )
            continue

        tensor = f.to_tensor(
            Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        ).float().unsqueeze(0)
        if tensor.size(-1) != 960:
            tensor = resize(tensor)
        tensor = tensor.to(device)
        _, _, h, w = tensor.size()

        with torch.no_grad():
            heatmaps = model(tensor)
            heatmaps_l = model_l(tensor)

        kp_coords = get_keypoints_from_heatmap_batch_maxpool(heatmaps[:, :-1, :, :])
        line_coords = get_keypoints_from_heatmap_batch_maxpool_l(
            heatmaps_l[:, :-1, :, :]
        )
        kp_dict = coords_to_dict(kp_coords, threshold=args.kp_threshold)
        lines_dict = coords_to_dict(line_coords, threshold=args.line_threshold)
        kp_dict, lines_dict = complete_keypoints(
            kp_dict[0], lines_dict[0], w=w, h=h, normalize=True
        )

        calib.update(kp_dict, lines_dict)
        ground = calib.subsets["ground_plane"]
        image_points = []
        pitch_points = []
        keypoint_ids = []
        for kp_id, entry in sorted(ground.items()):
            # world coords in utils_calib are centred; shift back to 0..105/0..68
            pitch_x = float(entry["xw"]) + 52.5
            pitch_y = float(entry["yw"]) + 34.0
            if not (0.0 <= pitch_x <= 105.0 and 0.0 <= pitch_y <= 68.0):
                continue
            image_points.append([float(entry["xi"]), float(entry["yi"])])
            pitch_points.append([pitch_x, pitch_y])
            keypoint_ids.append(int(kp_id))

        status = "ok" if len(image_points) >= args.min_keypoints else "too_few_keypoints"
        frames_out.append(
            {
                "frame_id": frame_id,
                "status": status,
                "keypoints": len(image_points),
                "keypoint_ids": keypoint_ids,
                "image_points": image_points,
                "pitch_points": pitch_points,
            }
        )
    cap.release()

    payload = {
        "worker": "pnlcalib_worker",
        "video": args.video,
        "image_width": width,
        "image_height": height,
        "device": device,
        "kp_threshold": args.kp_threshold,
        "line_threshold": args.line_threshold,
        "elapsed_seconds": round(time.time() - started, 3),
        "frames": frames_out,
    }
    Path(args.output).write_text(json.dumps(payload), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
