"""Interactive manual pitch calibration tool.

Creates an operator-verified calibration JSON for the MVP-2 pipeline by
matching clicked pixel positions of known pitch-line intersections to
canonical 105x68 metre pitch coordinates.

Interactive mode (default, requires a display):
    python scripts/create_manual_calibration.py \
        --video /path/to/match.mp4 --frame-id 100 \
        --output configs/calibration/manual_match.json

    Left-click a verified pitch line intersection, then type its pitch
    coordinates in metres (x y) in the terminal. Keys in the window:
    u = undo last point, s = save and validate, q = abort.

Non-interactive mode (pre-verified points):
    python scripts/create_manual_calibration.py \
        --video ... --frame-id 100 --points-file points.json --output out.json

    points.json: [{"image": [px, py], "pitch": [x_m, y_m]}, ...]

The tool refuses to save a calibration that fails any validation gate:
fewer than four points, collinear point triples, degenerate homography,
excessive reprojection error, or insufficient visible pitch coverage.
The output is labelled DEMO_MANUAL_FALLBACK and records provenance.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0
LABEL = "DEMO_MANUAL_FALLBACK"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--video", required=True, help="source video path")
    parser.add_argument("--frame-id", type=int, required=True,
                        help="frame to annotate (pick a main_wide frame)")
    parser.add_argument("--output", required=True, help="output calibration JSON path")
    parser.add_argument("--points-file", default=None,
                        help="optional JSON with pre-verified point pairs")
    parser.add_argument("--orientation", default="left_to_right",
                        choices=["left_to_right", "right_to_left"])
    parser.add_argument("--pitch-length-m", type=float, default=PITCH_LENGTH_M)
    parser.add_argument("--pitch-width-m", type=float, default=PITCH_WIDTH_M)
    parser.add_argument("--max-reprojection-error", type=float, default=8.0,
                        help="maximum mean reprojection error in metres")
    parser.add_argument("--min-coverage", type=float, default=0.15,
                        help="minimum visible pitch coverage (convex hull ratio)")
    parser.add_argument("--min-points", type=int, default=4)
    parser.add_argument("--save-frame", default=None,
                        help="optionally save the annotated source frame PNG here")
    return parser.parse_args()


def read_frame(video: Path, frame_id: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {video}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"cannot read frame {frame_id} from {video}")
    return frame


def collinear_triples(points: np.ndarray, tolerance: float) -> list[tuple[int, int, int]]:
    bad: list[tuple[int, int, int]] = []
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a, b, c = points[i], points[j], points[k]
                area = abs(
                    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
                ) / 2.0
                if area < tolerance:
                    bad.append((i, j, k))
    return bad


def validate_and_fit(
    image_points: np.ndarray,
    pitch_points: np.ndarray,
    args: argparse.Namespace,
) -> dict:
    errors: list[str] = []
    if len(image_points) < args.min_points:
        errors.append(
            f"need at least {args.min_points} points, got {len(image_points)}"
        )
    if np.any(pitch_points[:, 0] < 0) or np.any(pitch_points[:, 0] > args.pitch_length_m) \
            or np.any(pitch_points[:, 1] < 0) or np.any(pitch_points[:, 1] > args.pitch_width_m):
        errors.append("pitch coordinates outside canonical pitch bounds")
    if len(image_points) >= 3:
        # The whole set must never be collinear (degenerate homography).
        image_hull = cv2.convexHull(image_points.astype(np.float32))
        pitch_hull = cv2.convexHull(pitch_points.astype(np.float32))
        if cv2.contourArea(image_hull) < 100.0:
            errors.append("clicked pixel points are (near-)collinear as a set")
        if cv2.contourArea(pitch_hull) < 2.0:
            errors.append("pitch points are (near-)collinear as a set")
        # With exactly the minimum four points, no three may be collinear,
        # otherwise the homography is under-determined.
        if len(image_points) == 4:
            if collinear_triples(image_points, 50.0):
                errors.append(
                    "with only four points, no three pixel points may be collinear"
                )
            if collinear_triples(pitch_points, 1.0):
                errors.append(
                    "with only four points, no three pitch points may be collinear"
                )
    if errors:
        return {"valid": False, "errors": errors}

    homography, _ = cv2.findHomography(
        image_points.astype(np.float64), pitch_points.astype(np.float64), method=0
    )
    if homography is None or not np.all(np.isfinite(homography)) \
            or abs(float(np.linalg.det(homography))) < 1e-12:
        return {"valid": False, "errors": ["degenerate homography from these points"]}
    homography = homography / homography[2, 2]

    projected = cv2.perspectiveTransform(
        image_points.reshape(1, -1, 2).astype(np.float64), homography
    )[0]
    per_point = np.linalg.norm(projected - pitch_points, axis=1)
    reprojection_error = float(np.mean(per_point))
    hull = cv2.convexHull(pitch_points.astype(np.float32))
    coverage = float(
        np.clip(
            cv2.contourArea(hull) / (args.pitch_length_m * args.pitch_width_m),
            0.0,
            1.0,
        )
    )
    if reprojection_error > args.max_reprojection_error:
        errors.append(
            f"mean reprojection error {reprojection_error:.3f} m exceeds "
            f"{args.max_reprojection_error:.3f} m"
        )
    if coverage < args.min_coverage:
        errors.append(
            f"visible pitch coverage {coverage:.3f} below minimum {args.min_coverage:.3f}"
        )
    if errors:
        return {
            "valid": False,
            "errors": errors,
            "reprojection_error": reprojection_error,
            "coverage": coverage,
        }
    return {
        "valid": True,
        "homography": homography.tolist(),
        "reprojection_error": reprojection_error,
        "per_point_error_m": [round(float(v), 4) for v in per_point],
        "coverage": coverage,
    }


def collect_points_interactive(frame: np.ndarray) -> list[dict]:
    points: list[dict] = []
    window = "manual calibration (u=undo, s=save, q=quit)"
    display = frame.copy()

    def redraw() -> None:
        nonlocal display
        display = frame.copy()
        for index, item in enumerate(points):
            px, py = int(item["image"][0]), int(item["image"][1])
            cv2.drawMarker(display, (px, py), (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
            cv2.putText(
                display,
                f"{index}:({item['pitch'][0]:.1f},{item['pitch'][1]:.1f})",
                (px + 8, py - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
            )
        cv2.imshow(window, display)

    def on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
        del flags, param
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        print(f"\nClicked pixel ({x}, {y}).")
        raw = input(
            "Pitch coordinates in metres as 'x y' "
            f"(0..{PITCH_LENGTH_M} 0..{PITCH_WIDTH_M}), empty to discard: "
        ).strip()
        if not raw:
            print("discarded.")
            return
        try:
            pitch_x, pitch_y = (float(v) for v in raw.split())
        except ValueError:
            print("could not parse two numbers; point discarded.")
            return
        points.append({"image": [float(x), float(y)], "pitch": [pitch_x, pitch_y]})
        redraw()

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)
    redraw()
    while True:
        key = cv2.waitKey(50) & 0xFF
        if key == ord("u") and points:
            removed = points.pop()
            print(f"removed point {removed}")
            redraw()
        elif key == ord("s"):
            break
        elif key == ord("q"):
            cv2.destroyAllWindows()
            raise SystemExit("aborted by user; no calibration written")
    cv2.destroyAllWindows()
    return points


def main() -> int:
    args = parse_args()
    video = Path(args.video)
    frame = read_frame(video, args.frame_id)

    if args.points_file:
        raw = json.loads(Path(args.points_file).read_text(encoding="utf-8"))
        points = [
            {"image": [float(p["image"][0]), float(p["image"][1])],
             "pitch": [float(p["pitch"][0]), float(p["pitch"][1])]}
            for p in raw
        ]
        print(f"loaded {len(points)} pre-verified points from {args.points_file}")
    else:
        points = collect_points_interactive(frame)

    image_points = np.asarray([p["image"] for p in points], dtype=np.float64)
    pitch_points = np.asarray([p["pitch"] for p in points], dtype=np.float64)
    fit = validate_and_fit(image_points, pitch_points, args)
    if not fit["valid"]:
        print("CALIBRATION REJECTED:", file=sys.stderr)
        for error in fit["errors"]:
            print(f"  - {error}", file=sys.stderr)
        return 1

    if args.save_frame:
        annotated = frame.copy()
        for index, p in enumerate(points):
            px, py = int(p["image"][0]), int(p["image"][1])
            cv2.drawMarker(annotated, (px, py), (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
            cv2.putText(annotated, str(index), (px + 8, py - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        Path(args.save_frame).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(args.save_frame, annotated)

    video_sha = hashlib.sha256()
    with video.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            video_sha.update(chunk)

    payload = {
        "schema_version": "1.0.0",
        "label": LABEL,
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "source": {
            "video": str(video),
            "video_sha256": video_sha.hexdigest(),
            "frame_id": args.frame_id,
            "frame_width": int(frame.shape[1]),
            "frame_height": int(frame.shape[0]),
        },
        "calibration": {
            "label": LABEL,
            "orientation": args.orientation,
            "pitch_length_m": args.pitch_length_m,
            "pitch_width_m": args.pitch_width_m,
            "image_points": image_points.tolist(),
            "pitch_points": pitch_points.tolist(),
            "homography": fit["homography"],
            "reprojection_error": fit["reprojection_error"],
            "per_point_error_m": fit["per_point_error_m"],
            "visible_pitch_coverage": fit["coverage"],
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"saved {output}\n"
        f"  points={len(points)} reprojection_error={fit['reprojection_error']:.3f} m "
        f"coverage={fit['coverage']:.3f} label={LABEL}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
