from ultralytics import YOLO
import os
from datasets import load_dataset



def download_sample_images(n):
    # === Load N images from WikiArt ===
    dataset = load_dataset("huggan/wikiart", split="train")
    
    output_dir = "test"
    os.makedirs(output_dir, exist_ok=True)

    image_paths = []
    for i in range(n):
        image = dataset[i]["image"]
        image_path = os.path.join(output_dir, f"wikiart_sample_{i}.jpg")
        image.save(image_path)
        image_paths.append(image_path)

    print(f"✅ Downloaded {n} images to: {output_dir}")
    return image_paths

# def predict_on_images(image_paths):
#     # === Load YOLO model ===
#     model_path = r"C:\Users\nglan\OneDrive\Desktop\Project\runs\detect\train10\weights\best.pt"
#     model = YOLO(model_path)

#     output_dir = "output"
#     os.makedirs(output_dir, exist_ok=True)

#     for image_path in image_paths:
#         print(f"🔍 Predicting: {image_path}")
#         results = model(image_path, conf=0.3, iou=0.5)
        
#         # Show (optional)
#         results[0].show()

#         # Save prediction image to output folder
#         results[0].save(output_dir)

#     print(f"📦 All predictions saved to: {output_dir}")


def predict_on_images(image_paths):
    # === Load YOLO model ===
    model_path = r"C:\Users\nglan\OneDrive\Desktop\Project\runs\detect\train16\weights\best.pt"
    model = YOLO(model_path)

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    for image_path in image_paths:
        print(f"🔍 Predicting: {image_path}")
        results = model.predict(
            source=image_path,
            conf=0.3,
            iou=0.5,
            save=True,
            project=output_dir,
            name=".",
            exist_ok=True
        )

        results[0].show()

    print(f"📦 All predictions saved to: {output_dir}")





if __name__ == "__main__":
    # Step 1: Download multiple sample images
    image_paths = download_sample_images(500)

    # # Step 2: Run prediction on all of them
    # predict_on_images(image_paths)
