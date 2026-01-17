import cv2
import torch
import numpy as np
from ultralytics import YOLO
from torchvision import transforms, models
import torch.nn as nn

# =========================
# CONFIG
# =========================
VIDEO_PATH = "detection/test/traffic_test.mp4"  
OUTPUT_PATH = "detection/output_yolo_cnn.mp4"

YOLO_MODEL_PATH = "detection/runs/car_detection4/weights/best.pt"
CNN_MODEL_PATH = "classifier/weights/cnn_car_type.pt"

IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAMES = ['Hatchback', 'MPV', 'SUV', 'Sedan', 'Truck']

print("Using device:", DEVICE)

# =========================
# LOAD YOLO
# =========================
yolo = YOLO(YOLO_MODEL_PATH)

# =========================
# LOAD CNN (ResNet18)
# =========================
cnn = models.resnet18(weights=None)
cnn.fc = nn.Linear(cnn.fc.in_features, len(CLASS_NAMES))
cnn.load_state_dict(torch.load(CNN_MODEL_PATH, map_location=DEVICE))
cnn = cnn.to(DEVICE)
cnn.eval()

# =========================
# TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# =========================
# VIDEO IO
# =========================
cap = cv2.VideoCapture(VIDEO_PATH)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

# =========================
# PROCESS VIDEO
# =========================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = yolo(frame, conf=0.25, verbose=False)[0]

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Crop car
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        # CNN classify
        img_tensor = transform(crop).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = cnn(img_tensor)
            pred = torch.argmax(outputs, dim=1).item()
            car_type = CLASS_NAMES[pred]

        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"CAR | {car_type}"
        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    out.write(frame)

cap.release()
out.release()

print("Inference selesai. Video disimpan di:", OUTPUT_PATH)
