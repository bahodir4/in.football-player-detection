# Football Player Detection & Analysis

A computer vision pipeline that detects and tracks players, referees, and the ball in football match footage, compensates for camera movement, transforms positions to real-world pitch coordinates, assigns players to teams by jersey color, and tracks ball possession throughout the video.

## Demo

<video src="data/output/08fd33_4_output.mp4" controls width="100%"></video>

## What it does

- Detects players, goalkeepers, referees, and the ball using a fine-tuned YOLOv5x model
- Tracks every object across frames with ByteTrack, keeping consistent IDs
- Estimates camera movement per frame using Lucas-Kanade optical flow and adjusts all tracked positions accordingly
- Transforms pixel positions to real-world pitch coordinates (meters) using perspective transform
- Estimates player speed (km/h) and total distance covered (m) from the transformed positions
- Separates players into two teams using K-Means clustering on jersey colors
- Determines which player has the ball each frame and calculates team ball-control percentages
- Interpolates missing ball positions to smooth out detection gaps
- Annotates and exports the final video with ellipses, player IDs, speed/distance labels, ball triangles, camera movement overlay, and ball-control stats

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
│   ├── trackers/                       # detection, ByteTrack tracking, annotation
│   ├── camera_movement_estimator/      # Lucas-Kanade optical flow camera compensation
│   ├── view_transformer/               # perspective transform to real-world pitch coords
│   ├── speed_and_distance_estimator/   # per-player speed (km/h) and distance (m)
│   ├── team_assigner/                  # K-Means jersey-color team clustering
│   ├── player_ball_assigner/           # ball possession assignment per frame
│   └── utils/                          # bbox helpers, video I/O
├── stubs/                              # cached pkl files (tracks, camera movement)
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

Tracking and camera movement results are cached to `stubs/` on the first run. Subsequent runs load from the stubs instantly — delete the pkl files to re-run from scratch.

## Model

`models/best.pt` is a YOLOv5x model fine-tuned on a football player dataset from Roboflow Universe. The dataset contains annotated classes for `player`, `goalkeeper`, `referee`, and `ball`.

- Dataset: [Football Player Detection — Roboflow Universe](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc)
- Base architecture: YOLOv5x — 151 layers, 97.2M parameters, 246 GFLOPs
- Inference library: [Ultralytics](https://github.com/ultralytics/ultralytics) (`ultralytics==8.4.72`)
- Training: 100 epochs, ~2 hours on Tesla T4 (Google Colab), 612 train / 38 val images
- Download weights: [best.pt — Google Drive](https://drive.google.com/file/d/1vrJbNv1xRtnlAkU6L8aadKYewpFgr4l_/view?usp=share_link)

### Validation results (best.pt)

| Class | Precision | Recall | mAP@50 | mAP@50-95 |
|---|---|---|---|---|
| **all** | 0.882 | 0.829 | 0.845 | 0.586 |
| ball | 0.776 | 0.486 | 0.465 | 0.201 |
| goalkeeper | 0.891 | 0.909 | 0.977 | 0.734 |
| player | 0.967 | 0.983 | 0.985 | 0.764 |
| referee | 0.893 | 0.940 | 0.952 | 0.643 |

Inference speed: **16.7 ms/image** on T4 GPU. Ball detection scores are lower due to the small number of annotated ball instances (35) and its small size in-frame.

## Dependencies

| Package | Version |
|---|---|
| ultralytics | 8.4.72 |
| supervision | latest |
| opencv-python | 4.13.0.92 |
| scikit-learn | 1.9.0 |
| pandas | 3.0.3 |
| roboflow | 1.3.10 |
