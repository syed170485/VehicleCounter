# Traffic Vehicle Counter

Real-time vehicle detection, tracking, and directional counting from traffic camera or dashcam video, using YOLOv8 for detection and a lightweight centroid tracker for identity persistence across frames.

## Problem

Traffic planners and smart-city systems need per-lane vehicle counts (and direction of travel) without expensive inductive-loop sensors. This project turns any fixed camera feed into a vehicle counter.

## Approach

1. **Detection** — YOLOv8n (Ultralytics, COCO-pretrained) detects `car`, `truck`, `bus`, and `motorcycle` in each frame.
2. **Tracking** — A centroid-distance tracker assigns a persistent ID to each detected vehicle across frames, so the same car isn't counted twice.
3. **Counting** — A virtual line is drawn across the frame. When a tracked centroid crosses the line, it increments the count for its direction (in / out).
4. **Output** — Annotated video with bounding boxes, track IDs, the counting line, and a running tally overlay.

## Tech Stack

- Python 3.10+
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) (object detection)
- OpenCV (video I/O, drawing, and the tracker's frame handling)
- NumPy

## Project Structure

```
01-traffic-vehicle-counter/
├── main.py            # entry point: runs detection + tracking + counting on a video
├── tracker.py          # CentroidTracker class
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python main.py --source path/to/traffic.mp4 --line-y 400 --output annotated.mp4
```

Use `--source 0` to run on a live webcam feed instead of a file.

| Flag | Description | Default |
|---|---|---|
| `--source` | Video file path or camera index | required |
| `--line-y` | Y-coordinate (pixels) of the counting line | frame midpoint |
| `--conf` | Detection confidence threshold | 0.35 |
| `--output` | Path to save annotated video (omit to skip saving) | none |

## Results (fill in after running on your own footage)

| Metric | Value |
|---|---|
| Vehicles counted (in) | — |
| Vehicles counted (out) | — |
| Avg. FPS on test hardware | — |

## Possible Extensions

- Per-class counts (cars vs. trucks vs. motorcycles)
- Multi-line / multi-lane counting
- Speed estimation from frame-to-frame displacement + camera calibration
- Export counts to CSV/time-series for traffic-pattern dashboards
