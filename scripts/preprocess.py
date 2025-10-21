import os
os.environ["OPENCV_DISABLE_OPENMP"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Config
DATASET_PATH = "data/raw"
SAVE_PATH = "data/preprocessed"
IMG_SIZE = 224
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]


os.makedirs(SAVE_PATH, exist_ok=True)

# Load images and labes
data = []
labels = []

for idx, label in enumerate(CLASSES):
    folder = os.path.join(DATASET_PATH, label)
    if not os.path.isdir(folder):
        print(f"⚠️ Missing folder: {folder}")
        continue
    for file in os.listdir(folder):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(folder, file)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            data.append(img)
            labels.append(idx)

data = np.array(data, dtype=np.float32)
labels = np.array(labels)

print(f"✅ Loaded {len(data)} images total.")

# Normalize and encode
data = data / 255.0
labels = to_categorical(labels, num_classes=len(CLASSES))

# Split data
X_train, X_temp, y_train, y_temp = train_test_split(
    data, labels, test_size=0.3, random_state=42, stratify=labels
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# Save arrays
np.save(os.path.join(SAVE_PATH, "X_train.npy"), X_train)
np.save(os.path.join(SAVE_PATH, "y_train.npy"), y_train)
np.save(os.path.join(SAVE_PATH, "X_val.npy"), X_val)
np.save(os.path.join(SAVE_PATH, "y_val.npy"), y_val)
np.save(os.path.join(SAVE_PATH, "X_test.npy"), X_test)
np.save(os.path.join(SAVE_PATH, "y_test.npy"), y_test)

print(f"💾 Preprocessed data saved to {SAVE_PATH}")