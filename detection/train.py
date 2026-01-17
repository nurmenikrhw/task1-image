print("=== TRAIN.PY STARTED ===")

from ultralytics import YOLO
import torch

print("=== ULTRALYTICS IMPORTED ===")

def main():
    print("=== ENTER MAIN ===")

    print(torch.version.cuda)          # versi CUDA yang digunakan PyTorch
    print(torch.cuda.is_available())   # harus True
    print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== USING DEVICE: {device} ===")

    model = YOLO("yolov8n.pt")  # atau yolov8n.yaml
    print("=== MODEL LOADED ===")

    model.train(
        data="../dataset/data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        project="runs",
        name="car_detection",
        device=device  # penting supaya pakai GPU
    )

if __name__ == "__main__":
    main()
