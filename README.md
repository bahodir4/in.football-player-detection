# Football Player Detection & Analysis

A computer vision pipeline that detects and tracks players, referees, and the ball in football match footage, assigns players to teams by jersey color, and tracks ball possession throughout the video.

## Demo

<video src="data/output/08fd33_4_output.mp4" controls width="100%"></video>

## What it does

- Detects players, goalkeepers, referees, and the ball using a fine-tuned YOLO model
- Tracks every object across frames with ByteTrack, keeping consistent IDs
- Separates players into two teams using K-Means clustering on jersey colors
- Determines which player has the ball each frame and calculates team ball-control percentages
- Interpolates missing ball positions to smooth out detection gaps
- Annotates and exports the final video with ellipses, player IDs, ball triangles, and a ball-control overlay

## Project structure

```
├── data/
│   ├── input/          # source video(s)
│   └── output/         # annotated output video
├── models/
│   ├── best.pt         # custom-trained football detection model
│   └── yolov8x.pt      # base YOLOv8x weights
├── notebooks/
│   └── color_assignment.ipynb
├── src/
│   ├── app.py                          # main entry point
│   ├── trackers/tracker.py             # detection + tracking + annotation
│   ├── team_assigner/team_assigner.py  # jersey-color team clustering
│   ├── player_ball_assigner/           # ball possession assignment
│   └── utils/                          # bbox helpers, video I/O
├── stubs/                              # cached track pkl files
└── requirements.txt
```

## Setup

Python 3.11 is required.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place your input video in `data/input/` and update the filename in `src/app.py` if needed.

## Run

```bash
cd src
python app.py
```

Tracking results are cached to `stubs/track_stubs.pkl` on the first run. Subsequent runs load from the stub instantly — delete the file to re-run detection.

## Model

`models/best.pt` is a YOLOv5x model fine-tuned on a football player dataset from Roboflow Universe. The dataset contains annotated classes for `player`, `goalkeeper`, `referee`, and `ball`.

- Dataset: [Football Player Detection — Roboflow Universe](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc)
- Base architecture: YOLOv5x
- Inference library: [Ultralytics](https://github.com/ultralytics/ultralytics) (`ultralytics==8.4.72`)

## Dependencies

| Package | Version |
|---|---|
| ultralytics | 8.4.72 |
| supervision | latest |
| opencv-python | 4.13.0.92 |
| scikit-learn | 1.9.0 |
| pandas | 3.0.3 |
| roboflow | 1.3.10 |
