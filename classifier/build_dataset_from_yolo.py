import os
import cv2

# ==============================
# PATH CONFIG
# ==============================
YOLO_IMAGE_DIR = "dataset/train/images"
YOLO_LABEL_DIR = "dataset/train/labels"
OUTPUT_DIR = "classifier/dataset"

# ==============================
# YOLO CLASS NAMES
# ==============================
YOLO_CLASSES = [
    'BAK TERBUKA_DAIHATSU_GRANDMAX', 'BAK TERBUKA_MITSUBISHI_COLT',
    'BAK TERBUKA_SUZUKI_CARRY', 'BUS_ISUZU_ELF',
    'JEEP_DAIHATSU_ROCKY', 'JEEP_DAIHATSU_TERRIOS', 'JEEP_HONDA_CRV',
    'JEEP_HONDA_HRV', 'JEEP_MITSUBISHI_PAJERO', 'JEEP_MITSUBISHI_XFORCE',
    'JEEP_NISSAN_JUKE', 'JEEP_NISSAN_XTRAIL', 'JEEP_SUZUKI_IGNIS',
    'JEEP_TOYOTA_FORTUNER', 'JEEP_TOYOTA_RAIZE', 'JEEP_TOYOTA_RUSH',
    'JEEP_TOYOTA_YARIS', 'MINIBUS_DAIHATSU_AYLA',
    'MINIBUS_DAIHATSU_GRANDMAX', 'MINIBUS_DAIHATSU_LUXIO',
    'MINIBUS_DAIHATSU_SIGRA', 'MINIBUS_DAIHATSU_XENIA',
    'MINIBUS_HONDA_BRIO', 'MINIBUS_HONDA_BRV', 'MINIBUS_HONDA_FREED',
    'MINIBUS_HONDA_JAZZ', 'MINIBUS_HONDA_MOBILLIO',
    'MINIBUS_ISUZU_PANTHER', 'MINIBUS_MITSUBISHI_T120',
    'MINIBUS_MITSUBISHI_XPANDER', 'MINIBUS_NISSAN_GRAND LIVINA',
    'MINIBUS_NISSAN_MARCH', 'MINIBUS_SUZUKI_APV',
    'MINIBUS_SUZUKI_CARRY', 'MINIBUS_SUZUKI_ERTIGA',
    'MINIBUS_SUZUKI_KARIMUN', 'MINIBUS_SUZUKI_SWIFT',
    'MINIBUS_SUZUKI_XL7', 'MINIBUS_TOYOTA_AGYA',
    'MINIBUS_TOYOTA_ALPARD', 'MINIBUS_TOYOTA_AVANZA',
    'MINIBUS_TOYOTA_CALYA', 'MINIBUS_TOYOTA_HIACE',
    'MINIBUS_TOYOTA_INNOVA', 'MINIBUS_TOYOTA_KIJANG',
    'MINIBUS_TOYOTA_SIENTA', 'SEDAN_HONDA_CITY',
    'SEDAN_TOYOTA_CAMRY', 'SEDAN_TOYOTA_VIOS',
    'TRUCK_HINO_DUTRO', 'TRUCK_MITSUBISHI_FUSO'
]

# ==============================
# MODEL → CAR TYPE MAPPING
# ==============================
def map_to_car_type(label):
    if label.startswith("JEEP"):
        return "SUV"
    if label.startswith("MINIBUS"):
        return "MPV"
    if label.startswith("SEDAN"):
        return "Sedan"
    if label.startswith("TRUCK") or label.startswith("BAK TERBUKA"):
        return "Truck"
    if "AGYA" in label or "BRIO" in label or "AYLA" in label:
        return "Hatchback"
    return None

# ==============================
# MAIN PROCESS
# ==============================
os.makedirs(OUTPUT_DIR, exist_ok=True)

total_saved = 0
total_skipped = 0

for img_name in os.listdir(YOLO_IMAGE_DIR):
    if not img_name.lower().endswith(".jpg"):
        continue

    img_path = os.path.join(YOLO_IMAGE_DIR, img_name)
    label_path = os.path.join(YOLO_LABEL_DIR, img_name.replace(".jpg", ".txt"))

    if not os.path.exists(label_path):
        continue

    image = cv2.imread(img_path)
    if image is None:
        continue

    h, w, _ = image.shape
    img_area = w * h

    with open(label_path, "r") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        cls_id, xc, yc, bw, bh = map(float, line.strip().split())
        cls_id = int(cls_id)

        label_name = YOLO_CLASSES[cls_id]
        car_type = map_to_car_type(label_name)

        if car_type is None:
            total_skipped += 1
            continue

        # Convert YOLO normalized bbox → pixel
        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)

        # Boundary check
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        box_w = x2 - x1
        box_h = y2 - y1
        area_ratio = (box_w * box_h) / img_area

        # ==============================
        # FILTER NOISE
        # ==============================
        if area_ratio < 0.05:   # terlalu kecil → buang
            total_skipped += 1
            continue

        aspect_ratio = box_h / (box_w + 1e-6)
        if aspect_ratio > 2.5:  # terlalu tinggi & kurus → motor
            total_skipped += 1
            continue

        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            total_skipped += 1
            continue

        save_dir = os.path.join(OUTPUT_DIR, car_type)
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, f"{img_name[:-4]}_{i}.jpg")
        cv2.imwrite(save_path, crop)
        total_saved += 1

print("===================================")
print("Dataset classifier FINAL selesai")
print(f"Total saved  : {total_saved}")
print(f"Total skipped: {total_skipped}")
print("===================================")
