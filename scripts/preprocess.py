import os
# Ensure single-threading for stability in certain environments
os.environ["OPENCV_DISABLE_OPENMP"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
IMG_SIZE = 224
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

# --- PATH RESOLUTION (CRITICAL FIX) ---
# Check if the script is run from a subfolder like 'notebooks/' or the root
if os.path.isdir("./data/raw"):
    DATASET_PATH = "./data/raw"
    SAVE_PATH = "./data/preprocessed"
elif os.path.isdir("../data/raw"):
    DATASET_PATH = "../data/raw"
    SAVE_PATH = "../data/preprocessed"
else:
    # If the paths are incorrect, try to guess the Colab/Drive absolute path (if applicable)
    # NOTE: You must adjust this if you are not using Google Drive/Colab
    print("🚨 WARNING: Cannot find 'data/raw' relative to current location. Using assumed absolute path.")
    DATASET_PATH = "/content/drive/MyDrive/RecycleBuddy/data/raw"
    SAVE_PATH = "/content/drive/MyDrive/RecycleBuddy/data/preprocessed"


os.makedirs(SAVE_PATH, exist_ok=True)
print(f"Using DATASET_PATH: {DATASET_PATH}")
print(f"Saving to SAVE_PATH: {SAVE_PATH}")


# --- IMAGE LOADING ---
data, labels = [], []

for idx, label in enumerate(CLASSES):
    folder = os.path.join(DATASET_PATH, label)
    if not os.path.isdir(folder):
        print(f"⚠️ Folder missing or path incorrect: {folder}")
        continue
    
    for file in tqdm(os.listdir(folder), desc=f"Loading {label}"):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(folder, file)
            img = cv2.imread(img_path)
            if img is None:
                continue
            # Ensure images are correctly loaded and resized
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert BGR to RGB for consistency
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            data.append(img)
            labels.append(idx)

data = np.array(data, dtype=np.float32)
labels = np.array(labels)

print(f"✅ Loaded {len(data)} images total.")


# --- PREPROCESSING AND SPLIT ---
if len(data) == 0:
    print("❌ ERROR: No images loaded. Cannot perform train_test_split.")
    raise ValueError("Cannot split data because n_samples=0. Check DATASET_PATH.")

# normalize and one-hot encode
data = data / 255.0
labels = to_categorical(labels, num_classes=len(CLASSES))

# split
# Note: stratify=labels requires labels to be 1D, but to_categorical makes them 2D (one-hot).
# We use the original indices for stratify.
X_train, X_temp, y_train, y_temp = train_test_split(
    data, labels, test_size=0.3, random_state=42, stratify=np.argmax(labels, axis=1)
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=np.argmax(y_temp, axis=1)
)

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")


# --- DATA SAVING ---
np.save(os.path.join(SAVE_PATH, "X_train.npy"), X_train)
np.save(os.path.join(SAVE_PATH, "y_train.npy"), y_train)
np.save(os.path.join(SAVE_PATH, "X_val.npy"), X_val)
np.save(os.path.join(SAVE_PATH, "y_val.npy"), y_val)
np.save(os.path.join(SAVE_PATH, "X_test.npy"), X_test)
np.save(os.path.join(SAVE_PATH, "y_test.npy"), y_test)

print(f"💾 Preprocessed data saved to {SAVE_PATH}")


# --- VISUALIZATION (Optional) ---
i = np.random.randint(len(X_train))
plt.imshow(X_train[i])
plt.title(f"Label: {CLASSES[np.argmax(y_train[i])]}")
plt.axis("off")
plt.show()