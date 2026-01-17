import cv2
import torch
from ultralytics import YOLO
from torchvision import transforms, models
import torch.nn as nn

# =========================
# CONFIG
# =========================
IMAGE_PATH = "detection/test/sedan.jpg"
OUTPUT_PATH = "detection/output_image_test.jpg"

YOLO_MODEL_PATH = "detection/runs/car_detection4/weights/best.pt"
CNN_MODEL_PATH = "classifier/weights/cnn_car_type.pt"

IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAMES = ['Hatchback', 'MPV', 'SUV', 'Sedan', 'Truck']

print("Using device:", DEVICE)

# =========================
# LOAD MODELS
# =========================
yolo = YOLO("yolov8n.pt")

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
# LOAD IMAGE
# =========================
image = cv2.imread(IMAGE_PATH)
assert image is not None, "Image not found!"

# =========================
# YOLO DETECTION
# =========================
results = yolo(image, conf=0.25, verbose=False)[0]

for box in results.boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])

    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        continue

    img_tensor = transform(crop).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = cnn(img_tensor)
        pred = torch.argmax(outputs, dim=1).item()
        car_type = CLASS_NAMES[pred]

    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        image,
        f"CAR | {car_type}",
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

# =========================
# SAVE RESULT
# =========================
cv2.imwrite(OUTPUT_PATH, image)
print("Inference selesai. Hasil disimpan di:", OUTPUT_PATH)
