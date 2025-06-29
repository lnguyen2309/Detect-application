from ultralytics import YOLO
import os

def predict_on_images(image_paths, model_path, conf=0.3, iou=0.7):
    model = YOLO(model_path)
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    saved_images = []
    for image_path in image_paths:
        results = model.predict(
            source=image_path,
            conf=conf,
            iou=iou,
            save=True,
            project=output_dir,
            name="",  # saves directly inside output_dir
            exist_ok=True
        )
        # The saved image has the same basename as the input image
        base_name = os.path.basename(image_path)
        saved_image_path = os.path.join(output_dir, base_name)
        saved_images.append(saved_image_path)
    
    return saved_images

