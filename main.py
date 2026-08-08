"""
Traffic Vehicle Counter
------------------------
Detects vehicles with YOLOv8, tracks them across frames with a centroid
tracker, and counts how many cross a virtual line (split by direction).

Usage:
    python main.py --source traffic.mp4 --line-y 400 --output annotated.mp4
    python main.py --source 0   # live webcam
"""

import argparse
import cv2
from ultralytics import YOLO

from tracker import CentroidTracker

VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle"}


def parse_args():
    parser = argparse.ArgumentParser(description="Count vehicles crossing a line in a video.")
    parser.add_argument("--source", required=True, help="Video file path or camera index (e.g. 0)")
    parser.add_argument("--line-y", type=int, default=None, help="Y-coordinate of the counting line")
    parser.add_argument("--conf", type=float, default=0.35, help="Detection confidence threshold")
    parser.add_argument("--output", default=None, help="Path to save the annotated video")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLOv8 weights to use")
    return parser.parse_args()


def main():
    args = parse_args()
    source = int(args.source) if str(args.source).isdigit() else args.source

    model = YOLO(args.model)
    tracker = CentroidTracker(max_disappeared=15, max_distance=75)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {args.source}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    line_y = args.line_y or height // 2

    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    prev_side = {}   # object_id -> "above" / "below" of the line, for crossing detection
    count_in, count_out = 0, 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)[0]
        detections = []
        for box in results.boxes:
            cls_name = model.names[int(box.cls[0])]
            conf = float(box.conf[0])
            if cls_name in VEHICLE_CLASSES and conf >= args.conf:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append((x1, y1, x2, y2))

        tracked = tracker.update(detections)

        for object_id, (centroid, bbox) in tracked.items():
            cx, cy = centroid
            side = "above" if cy < line_y else "below"

            if object_id in prev_side and prev_side[object_id] != side:
                if side == "below":
                    count_in += 1
                else:
                    count_out += 1

            prev_side[object_id] = side

            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(frame, f"ID {object_id}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)

        cv2.line(frame, (0, line_y), (width, line_y), (255, 0, 0), 2)
        cv2.putText(frame, f"IN: {count_in}  OUT: {count_out}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        if writer:
            writer.write(frame)

        cv2.imshow("Traffic Vehicle Counter", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    print(f"Final counts -> IN: {count_in}, OUT: {count_out}")


if __name__ == "__main__":
    main()
