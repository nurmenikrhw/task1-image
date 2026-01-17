import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# =====================
# CONFIG
# =====================
DATASET_DIR = "classifier/dataset"
BATCH_SIZE = 32
EPOCHS = 15
IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", DEVICE)

# =====================
# TRANSFORMS
# =====================
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
])

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# =====================
# DATASET
# =====================
full_dataset = datasets.ImageFolder(DATASET_DIR, transform=train_tf)
class_names = full_dataset.classes
num_classes = len(class_names)

print("Classes:", class_names)

# Split train / val
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_ds, val_ds = torch.utils.data.random_split(
    full_dataset, [train_size, val_size]
)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

# =====================
# MODEL (CNN - ResNet18)
# =====================
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Replace classifier head
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(DEVICE)

# =====================
# LOSS & OPTIMIZER
# =====================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# =====================
# TRAINING LOOP
# =====================
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # =====================
    # VALIDATION
    # =====================
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)

    print(f"Epoch {epoch+1} | Loss: {running_loss:.4f} | Val Acc: {acc:.4f}")

# =====================
# SAVE MODEL
# =====================
os.makedirs("classifier/weights", exist_ok=True)
torch.save(model.state_dict(), "classifier/weights/cnn_car_type.pt")

print("CNN Car Type Classifier training completed!")
