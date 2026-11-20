"""Analyze a constant-frame-rate side-view squat video."""
import argparse
import json
from math import isfinite
from pathlib import Path

from kinematics import measure
from squat import SquatTracker

CONNECTIONS = (
    (11, 12), (11, 23), (12, 24), (23, 24),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
)
LABELS = {
    "knee_flexion_deg": "Knee flexion",
    "hip_flexion_proxy_deg": "Hip flexion proxy",
    "ankle_dorsiflexion_proxy_deg": "Ankle dorsiflexion proxy",
    "torso_inclination_deg": "Torso inclination",
}


def draw(cv2, frame, points, metrics, phase, tracker, side):
    pixels = {i: tuple(round(v) for v in point) for i, point in points.items()}
    for start, end in CONNECTIONS:
        if start in pixels and end in pixels:
            cv2.line(frame, pixels[start], pixels[end], (55, 230, 170), 2, cv2.LINE_AA)
    for index in {i for edge in CONNECTIONS for i in edge} & pixels.keys():
        cv2.circle(frame, pixels[index], 4, (40, 170, 250), -1, cv2.LINE_AA)
    status = phase
    if tracker.baseline is None and phase != "untracked":
        status = "stand still to calibrate"
    lines = [f"Reps {len(tracker.reps)} | {status} | {side}"]
    for key, label in LABELS.items():
        value = f"{metrics[key]:.1f} deg" if key in metrics else "--"
        lines.append(f"{label}: {value}")
    scale = min(0.65, frame.shape[1] / 510, frame.shape[0] / 205)
    line_height = max(1, round(34 * scale))
    for index, line in enumerate(lines, 1):
        location = (6, index * line_height)
        cv2.putText(frame, line, location, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, location, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)


def analyze(args):
    import cv2

    from pose import PoseStream, open_landmarker

    tracker = SquatTracker()
    capture = cv2.VideoCapture(str(args.video))
    writer = None
    frame_count = tracked = 0
    annotated = args.output / "annotated.mp4"
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open video {args.video}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not isfinite(fps) or fps <= 0:
            raise ValueError("Video needs a valid constant frame rate.")
        with open_landmarker(args.model) as landmarker:
            stream = PoseStream(landmarker)
            last_ms = -1
            while True:
                success, frame = capture.read()
                if not success:
                    break
                if writer is None:
                    height, width = frame.shape[:2]
                    writer = cv2.VideoWriter(str(annotated), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
                    if not writer.isOpened():
                        raise ValueError("OpenCV could not open the MP4 writer.")
                time_ms = max(last_ms + 1, round(frame_count * 1000 / fps))
                last_ms = time_ms
                points = stream.read(frame, time_ms)
                metrics = measure(points, args.side)
                phase = tracker.update(time_ms / 1000, metrics)
                tracked += phase != "untracked"
                draw(cv2, frame, points, metrics, phase, tracker, args.side)
                writer.write(frame)
                frame_count += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    if not frame_count:
        raise ValueError("Video contains no decodable frames.")
    return {
        "schema_version": 1,
        "movement": "squat",
        "variant": args.variant,
        "view": "side",
        "measured_side": args.side,
        "coordinate_space": "image_2d",
        "fps": fps,
        "frames": frame_count,
        "tracked_knee_frames": tracked,
        "tracked_fraction": tracked / frame_count,
        "standing_baseline_deg": tracker.baseline,
        "rep_count": len(tracker.reps),
        "discarded_attempts": tracker.discarded + bool(tracker.frames),
        "reps": tracker.reps,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--side", choices=("left", "right"), required=True, help="Athlete's camera-facing side")
    parser.add_argument("--variant", choices=("bodyweight", "back", "front"), default="bodyweight")
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    args = parser.parse_args()
    for path in (args.video, args.model):
        if not path.is_file():
            parser.error(f"File not found: {path}")
    try:
        args.output.mkdir(parents=True)
        summary = analyze(args)
        (args.output / "summary.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"Analysis failed: {error}\n")
    print(f"{summary['rep_count']} reps | {summary['tracked_fraction']:.0%} knee tracking | {args.output}")


if __name__ == "__main__":
    main()
