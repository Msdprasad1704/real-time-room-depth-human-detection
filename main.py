from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

from src.pipeline import DepthHumanPipeline


ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT_DIR / "outputs"
SCREENSHOT_DIR = OUTPUT_DIR / "screenshots"
VIDEO_DIR = OUTPUT_DIR / "videos"
REPORT_DIR = OUTPUT_DIR / "reports"
for folder in (OUTPUT_DIR, SCREENSHOT_DIR, VIDEO_DIR, REPORT_DIR):
    folder.mkdir(exist_ok=True, parents=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Real-Time Room Depth Estimation and Human Detection Using YOLOv8",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--webcam", action="store_true", help="Use the webcam as input.")
    parser.add_argument("--input", type=str, default=None, help="Path to an input image file.")
    parser.add_argument("--conf", type=float, default=0.45, help="YOLO confidence threshold for person detection.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Use CUDA if available; otherwise CPU.",
    )
    return parser


def save_frame(frame: np.ndarray, label: str = "capture", folder: Path | None = None) -> Path:
    output_folder = folder or OUTPUT_DIR
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_path = output_folder / f"{label}_{timestamp}.png"
    success = cv2.imwrite(str(output_path), frame)
    if not success:
        raise RuntimeError(f"Could not save frame to {output_path!s}.")
    return output_path


def save_screenshot(frame: np.ndarray, label: str = "person_monitor") -> Path:
    return save_frame(frame, label=label, folder=SCREENSHOT_DIR)


def open_camera() -> cv2.VideoCapture:
    """Try a few standard OpenCV camera backends so the webcam opens reliably on Windows and other machines."""
    backends: list[int] = []
    if sys.platform.startswith("win"):
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    else:
        backends = [cv2.CAP_ANY]

    for backend in backends:
        for index in range(0, 6):
            camera = cv2.VideoCapture(index, backend)
            if camera.isOpened():
                return camera
            camera.release()

    for index in range(0, 6):
        camera = cv2.VideoCapture(index)
        if camera.isOpened():
            return camera
        camera.release()

    raise RuntimeError(
        "Unable to open a webcam. Please connect a camera or use --input with an image file instead."
    )


def export_csv_report(detections: list[dict], label: str = "occupancy_report") -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    report_path = REPORT_DIR / f"{label}_{timestamp}.csv"
    with report_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Timestamp", "Person ID", "Relative depth", "Depth zone", "Confidence"])
        for detection in detections:
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
                detection.get("person_id", "N/A"),
                f"{float(detection.get('relative_depth', 0.0)):.6f}",
                detection.get("depth_zone", "UNKNOWN"),
                f"{float(detection.get('confidence', 0.0)):.6f}",
            ])
    return report_path


def create_video_writer(frame_shape: tuple[int, int] | tuple[int, int, int], fps: float) -> tuple[cv2.VideoWriter, Path]:
    if len(frame_shape) == 3:
        frame_height, frame_width = frame_shape[0], frame_shape[1]
    else:
        frame_width, frame_height = frame_shape

    if frame_width <= 0 or frame_height <= 0:
        raise ValueError(f"Invalid frame size for recording: {frame_shape!r}.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    candidates: list[tuple[Path, int]] = [
        (VIDEO_DIR / f"room_depth_{timestamp}.avi", cv2.VideoWriter_fourcc(*"MJPG")),
        (VIDEO_DIR / f"room_depth_{timestamp}.mp4", cv2.VideoWriter_fourcc(*"mp4v")),
    ]

    for video_path, fourcc in candidates:
        writer = cv2.VideoWriter(str(video_path), fourcc, fps, (frame_width, frame_height))
        if writer.isOpened():
            return writer, video_path
        writer.release()

    raise RuntimeError(
        "Could not start video recorder. OpenCV could not open a compatible Windows video writer. "
        "Tried AVI/MJPG and MP4/mp4v."
    )


def toggle_recording(
    video_writer: cv2.VideoWriter | None,
    frame_shape: tuple[int, int] | tuple[int, int, int],
    fps: float,
) -> tuple[cv2.VideoWriter | None, Path | None]:
    if video_writer is not None:
        video_writer.release()
        return None, None

    writer, video_path = create_video_writer(frame_shape, fps)
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(
            "VideoWriter failed to initialize. Recording was not started. "
            f"Frame size: {frame_shape}."
        )
    return writer, video_path


def run_webcam(device: str, conf_threshold: float) -> int:
    try:
        pipeline = DepthHumanPipeline(device=device, conf_threshold=conf_threshold)
    except (FileNotFoundError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    try:
        camera = open_camera()
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    fps = 20.0
    video_writer = None
    video_path = None
    print("Webcam mode started. Controls: Q=Quit | S=Save screenshot | R=Record | H=Heatmap | D=Depth info | C=CSV report")

    while True:
        ret, frame = camera.read()
        if not ret:
            print("Error: Failed to read frames from webcam.", file=sys.stderr)
            break

        processed_frame, detections, _ = pipeline.process(frame)
        if video_writer is not None:
            video_writer.write(processed_frame)

        cv2.imshow("AI ROOM DEPTH & HUMAN MONITORING", processed_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            save_path = save_screenshot(processed_frame, label="webcam")
            print(f"Saved screenshot to: {save_path}")
        elif key == ord("r"):
            try:
                frame_size = (processed_frame.shape[1], processed_frame.shape[0])
                video_writer, video_path = toggle_recording(video_writer, frame_size, fps)
            except (RuntimeError, ValueError) as exc:
                print(f"Error: {exc}", file=sys.stderr)
            else:
                status = "started" if video_writer is not None else "stopped"
                print(f"Video recording {status}." if video_path is not None else "Video recording stopped.")
        elif key == ord("h"):
            heatmap_state = pipeline.toggle_heatmap()
            print(f"Heatmap {'ON' if heatmap_state else 'OFF'}")
        elif key == ord("d"):
            depth_state = pipeline.toggle_depth_info()
            print(f"Depth labels {'ON' if depth_state else 'OFF'}")
        elif key == ord("c"):
            report_path = export_csv_report(detections, label="report")
            print(f"Saved CSV report to: {report_path}")

    if video_writer is not None:
        video_writer.release()
    camera.release()
    cv2.destroyAllWindows()
    return 0


def run_image(input_path: str, device: str, conf_threshold: float) -> int:
    try:
        pipeline = DepthHumanPipeline(device=device, conf_threshold=conf_threshold)
    except (FileNotFoundError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    image = cv2.imread(input_path)
    if image is None:
        print(f"Error: Unable to read image from '{input_path}'.", file=sys.stderr)
        return 1

    processed_frame, detections, _ = pipeline.process(image)
    cv2.imshow("AI ROOM DEPTH & HUMAN MONITORING", processed_frame)
    print("Image mode. Press S to save screenshot, H to toggle heatmap, D to toggle depth labels, C to export CSV, or any key to close.")

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord("q") or key == 27:
            break
        if key == ord("s"):
            save_path = save_screenshot(processed_frame, label="image")
            print(f"Saved screenshot to: {save_path}")
        elif key == ord("h"):
            heatmap_state = pipeline.toggle_heatmap()
            print(f"Heatmap {'ON' if heatmap_state else 'OFF'}")
            processed_frame, detections, _ = pipeline.process(image)
            cv2.imshow("AI ROOM DEPTH & HUMAN MONITORING", processed_frame)
        elif key == ord("d"):
            depth_state = pipeline.toggle_depth_info()
            print(f"Depth labels {'ON' if depth_state else 'OFF'}")
            processed_frame, detections, _ = pipeline.process(image)
            cv2.imshow("AI ROOM DEPTH & HUMAN MONITORING", processed_frame)
        elif key == ord("c"):
            report_path = export_csv_report(detections, label="image_report")
            print(f"Saved CSV report to: {report_path}")

    cv2.destroyAllWindows()
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    if args.webcam and args.input:
        parser.error("Please provide only one input mode: --webcam or --input.")

    if args.webcam:
        return run_webcam(device=device, conf_threshold=args.conf)

    if args.input:
        return run_image(args.input, device=device, conf_threshold=args.conf)

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
