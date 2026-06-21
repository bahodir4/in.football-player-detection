from ultralytics import YOLO
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

model = YOLO(BASE_DIR / "models/yolov8x.pt")

result = model.predict(
    source=str(BASE_DIR / "data/input/08fd33_4.mp4"),
    save=True,
    project=str(BASE_DIR / "data"),
    name='output',
)

print(result[0])  # print results

for box in result[0].boxes:
    print(box)