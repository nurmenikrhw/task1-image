import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import os

# =========================
# CONFIG
# =========================
DATASET_DIR = "classifier/dataset"
SAVE_PATH = "classifier/weights/vit_car_type.pt"
BATCH_SIZE = 16
EPOCHS = 5
IMG_SIZE = 224

CLASS_NAMES = ['Hatchback', 'MPV', 'SUV', 'Sedan', 'Truck']
NUM_CLASSES = len(CLASS_NAMES)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# =========================
# TRANSFORMS
# =========================
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# =========================
# DATASET
# =========================
dataset = datasets.ImageFolder(DATASET_DIR, transform=train_transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

print("Classes:", dataset.classes)

# =========================
# MODEL (ViT)
# =========================
model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)

# Replace classification head
model.heads.head = nn.Linear(
    model.heads.head.in_features,
    NUM_CLASSES
)

model = model.to(DEVICE)

# =========================
# TRAINING SETUP
# =========================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# =========================
# TRAIN LOOP
# =========================
for epoch in range(EPOCHS):
    model.train()
    all_preds = []
    all_labels = []

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for images, labels in pbar:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        pbar.set_postfix(loss=loss.item())

    acc = accuracy_score(all_labels, all_preds)
    print(f"Epoch {epoch+1} | Accuracy: {acc:.4f}")

# =========================
# SAVE MODEL
# =========================
os.makedirs("classifier/weights", exist_ok=True)
torch.save(model.state_dict(), SAVE_PATH)
print("ViT training completed. Model saved to:", SAVE_PATH)
