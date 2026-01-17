import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

# =====================
# CONFIG
# =====================
DATA_DIR = "classifier/dataset"
MODEL_PATH = "classifier/weights/vit_car_type.pt"
CLASS_NAMES = ['Hatchback', 'MPV', 'SUV', 'Sedan', 'Truck']
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =====================
# DATA
# =====================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# =====================
# MODEL (ViT)
# =====================
model = models.vit_b_16(weights=None)
model.heads.head = nn.Linear(model.heads.head.in_features, len(CLASS_NAMES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# =====================
# EVALUATION
# =====================
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in dataloader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Accuracy
acc = accuracy_score(all_labels, all_preds)
print(f"ViT Accuracy: {acc:.4f}")

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)

os.makedirs("classifier/results", exist_ok=True)

plt.figure(figsize=(7, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix – ViT Classifier")
plt.tight_layout()
plt.savefig("classifier/results/vit_confusion_matrix.png")
plt.show()
