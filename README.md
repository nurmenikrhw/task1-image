# Car Detection & Classification System

This project implements a car detection and classification system using deep learning.
The system detects cars from images or videos using YOLO, then classifies the detected cars
into several categories using a CNN-based classifier.

## Features
- Multi-car detection using YOLO
- Car type classification (Hatchback, MPV, SUV, Sedan, Truck)
- Supports image and video inference
- Designed for Indonesian road vehicle types

## Model Architecture
- Object Detection: YOLO (CNN-based)
- Classification: CNN (PyTorch)

## Project Structure

```bash
task1_image/
│
├── classifier/
│ ├── build_dataset_from_yolo.py
│ ├── train_cnn.py
│ ├── train_vit.py
│ ├── eval_cnn.py
│ └── eval_vit.py
│
├── detection/
│ ├── train.py
│ ├── infer_yolo_cnn_image.py
│ └── infer_yolo_cnn_video.py
│
├── .gitignore
└── README.md
```

