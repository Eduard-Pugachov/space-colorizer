import os
from datasets import load_dataset
from PIL import Image

SAVE_DIR = "data/raw"
MAX_IMAGES = 2000
IMG_SIZE = 256

os.makedirs(SAVE_DIR, exist_ok=True)

ds = load_dataset("Supermaxman/esa-hubble", split="train", streaming=True)

saved = 0
for sample in ds:
    if saved >= MAX_IMAGES:
        break

    try:
        img = sample["image"]

        # making sure it's RGB 
        if img.mode != "RGB":
            img = img.convert("RGB")

        # resized to 256x256 to save storage
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)

        # save as JPEG
        save_path = os.path.join(SAVE_DIR, f"hubble_{saved:05d}.jpg")
        img.save(save_path, "JPEG", quality=90)

        saved += 1

        if saved % 100 == 0:
            print(f"Downloaded {saved}/{MAX_IMAGES} images...")

    except Exception as e:
        print(f"Skipping image due to error: {e}")
        continue

print(f"\nDone. {saved} images saved to {SAVE_DIR}/")
