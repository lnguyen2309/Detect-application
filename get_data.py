# from datasets import load_dataset
# import os
# # Load the dataset
# dataset = load_dataset("biglam/european_art")

# # Check the keys (usually includes 'train')
# print(dataset)

# # Access the train set
# train_data = dataset['train']

# # See what a sample looks like
# print(train_data[586])





# from datasets import load_dataset
# import os

# # Load the dataset
# dataset = load_dataset("biglam/european_art")
# train_data = dataset['train']

# # Create a folder to save images
# os.makedirs("my_dataset/images", exist_ok=True)

# # Loop and save using ID
# for i, item in enumerate(train_data):
#     try:
#         image = item['image']
#         image_id = item['file_id']

#         # Convert to RGB to avoid EXIF issues
#         if image.mode != "RGB":
#             image = image.convert("RGB")

#         image.save(f"my_dataset/images/{image_id}.jpg")
#         if i % 100 == 0:
#             print(f"Saved {i} images")
#     except Exception as e:
#         print(f"Error at index {i}: {e}")

import json
import os
import yaml
from datasets import load_dataset

# === Load class names from dataset.yaml ===
with open(r"C:\Users\nglan\OneDrive\Desktop\Project\my_dataset\dataset.yaml", "r") as f:
    yaml_data = yaml.safe_load(f)
    global_classes = yaml_data["names"]

# Build name → YOLO class ID mapping
name_to_yolo = {name.lower(): i for i, name in enumerate(global_classes)}

# Load HuggingFace dataset
dataset = load_dataset("biglam/european_art")
train_data = dataset["train"]

# Create output folders
os.makedirs("my_dataset/images", exist_ok=True)
os.makedirs("my_dataset/labels", exist_ok=True)

# Optional: debug one image
debug_id = "00017801"

# Loop through dataset
for i in range(len(train_data)):
    if i == 583:
        print(f"🚫 Skipping problematic image at index {i}")
        continue

    try:
        item = train_data[i]
        image = item['image']
        if image.mode != "RGB":
            image = image.convert("RGB")

        file_id = item['file_id']
        annotations_data = json.loads(item['annotations'])

        img_width, img_height = image.size
        image.save(f"my_dataset/images/{file_id}.jpg")

        local_categories = {cat["id"]: cat["name"] for cat in annotations_data["categories"]}

        yolo_lines = []
        for ann in annotations_data["annotations"]:
            if "bbox" not in ann:
                continue

            x, y, w, h = ann["bbox"]
            x_center = (x + w / 2) / img_width
            y_center = (y + h / 2) / img_height
            width = w / img_width
            height = h / img_height

            cat_name = local_categories[ann["category_id"]].strip().lower()
            class_id = name_to_yolo.get(cat_name)

            if class_id is None:
                print(f"⚠️ Warning: '{cat_name}' not found in YAML classes.")
                continue

            yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        with open(f"my_dataset/labels/{file_id}.txt", "w") as f:
            f.write("\n".join(yolo_lines))

        if i % 100 == 0:
            print(f"✅ Saved {i} images/labels")

    except Exception as e:
        print(f"❌ Error at item {i}: {e}")






