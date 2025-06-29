import json
import os
import random
import shutil
import yaml
from datasets import load_dataset
from ultralytics import YOLO
import torch


def load_classes(yaml_path):
    with open(yaml_path, "r") as f:
        yaml_data = yaml.safe_load(f)
        return yaml_data["names"]


def create_dirs(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def convert_and_save_images(train_data, name_to_yolo, images_dir, labels_dir):
    print("\n🔁 Getting image and converting annotations to YOLO format...")
    
    for i in range(len(train_data)):
        if i == 583:
            print(f"🚫 Skipping problematic image at index {i}")
            continue

        try:
            item = train_data[i]
            image = item['image']
            file_id = item['file_id']
            annotations_data = json.loads(item['annotations'])

            img_width, img_height = image.size
            image.save(os.path.join(images_dir, f"{file_id}.jpg"))

            local_categories = {cat["id"]: cat["name"] for cat in annotations_data.get("categories", [])}
            yolo_lines = []

            for ann in annotations_data.get("annotations", []):
                if "bbox" not in ann:
                    continue

                x, y, w, h = ann["bbox"]
                x_center = (x + w / 2) / img_width
                y_center = (y + h / 2) / img_height
                width = w / img_width
                height = h / img_height

                cat_name = local_categories.get(ann["category_id"], "").strip().lower()
                class_id = name_to_yolo.get(cat_name)

                if class_id is None:
                    print(f"⚠️ Warning: '{cat_name}' not in YAML.")
                    continue

                yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

            with open(os.path.join(labels_dir, f"{file_id}.txt"), "w") as f:
                f.write("\n".join(yolo_lines))

            
            if i % 100 == 0:
                print(f"✅ Saved {i} samples")

        except Exception as e:
            print(f"❌ Error at {i} ({item.get('file_id', 'unknown')}): {e}")

def split_and_move(files, set_name, images_dir, labels_dir):
    img_dest = os.path.join(images_dir, set_name)
    lbl_dest = os.path.join(labels_dir, set_name)
    create_dirs(img_dest, lbl_dest)

    for fname in files:
        img_src = os.path.join(images_dir, fname)
        lbl_src = os.path.join(labels_dir, fname.replace(".jpg", ".txt"))
        img_dst = os.path.join(img_dest, fname)
        lbl_dst = os.path.join(lbl_dest, fname.replace(".jpg", ".txt"))

        if os.path.exists(lbl_src):
            shutil.move(img_src, img_dst)
            shutil.move(lbl_src, lbl_dst)


def main():
    # === CONFIG ===
    base_dir = r"C:\Users\nglan\OneDrive\Desktop\Project\my_dataset"
    yaml_path = os.path.join(base_dir, "dataset.yaml")
    images_dir = os.path.join(base_dir, "images")
    labels_dir = os.path.join(base_dir, "labels")
    split_ratio = 0.8
    

    # === Load and prepare ===
    class_names = load_classes(yaml_path)
    name_to_yolo = {name.lower(): i for i, name in enumerate(class_names)}
    dataset = load_dataset("biglam/european_art")
    train_data = dataset["train"]

    create_dirs(images_dir, labels_dir)
    convert_and_save_images(train_data, name_to_yolo, images_dir, labels_dir)

    # === Split ===
    print("\n📦 Splitting into train/val...")
    image_files = [f for f in os.listdir(images_dir) if f.endswith(".jpg")]
    random.shuffle(image_files)
    split_index = int(len(image_files) * split_ratio)
    train_files = image_files[:split_index]
    val_files = image_files[split_index:]

    split_and_move(train_files, "train", images_dir, labels_dir)
    split_and_move(val_files, "val", images_dir, labels_dir)
    print(f"\n✅ Done: {len(train_files)} train, {len(val_files)} val")

    # === Train YOLO ===
    print("\n🚀 Starting YOLO training...")
    print("Using CUDA:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Current device:", torch.cuda.get_device_name(0))

    model = YOLO("yolo11n.pt")
    
    # model.train(data=yaml_path, epochs=30, batch=4, lr0=0.001, patience=10, imgsz=640, device=0)  # use GPU if available
    model.train(
    data=yaml_path,
    epochs=100,                  # Longer training
    imgsz=640,                   # Larger input size increases precision
    device=0,
    batch=8,                    # Depends on your GPU RAM
    lr0=0.001,                   # Lower learning rate for stable convergence
    optimizer="AdamW",           # Better for precision in some cases
    warmup_epochs=3,             # Smooth startup
    patience=50,                 # Early stopping if needed
)


    metrics = model.val(data=yaml_path, imgsz=640, device=0)
    print(metrics)  # Includes precision, recall, mAP


if __name__ == "__main__":
    main()
