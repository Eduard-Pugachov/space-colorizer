import os
import random

random.seed(42)

RAW_DIR    = "data/raw"
SPLITS_DIR = "data/splits"

# collect all image filenames
images = sorted([
    f for f in os.listdir(RAW_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

random.shuffle(images)

# 80% train, 20% val
split_idx = int(len(images) * 0.8)
train = images[:split_idx]
val   = images[split_idx:]

os.makedirs(SPLITS_DIR, exist_ok=True)

with open(os.path.join(SPLITS_DIR, "train.txt"), "w") as f:
    f.write("\n".join(train))

with open(os.path.join(SPLITS_DIR, "val.txt"), "w") as f:
    f.write("\n".join(val))

print(f"Total images : {len(images)}")
print(f"Train        : {len(train)}")
print(f"Val          : {len(val)}")
print(f"Splits saved to {SPLITS_DIR}/")