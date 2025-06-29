from flask import Flask, request, jsonify, render_template, send_from_directory
from predict import predict_on_images
import os
import shutil
import atexit

app = Flask(__name__)

# Absolute paths (adjust to your actual paths)
MODEL_PATH = r"C:\Users\nglan\OneDrive\Desktop\Project\model2.pt"
TEST_IMAGES_FOLDER = r"C:\Users\nglan\OneDrive\Desktop\Project\test"

# Define your base output folder absolute path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FOLDER = os.path.join(BASE_DIR, "output")  # points to ...\Project\output
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploaded")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route('/')
def index():
    image_names = [f for f in os.listdir(TEST_IMAGES_FOLDER)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]
    image_names.sort()
    return render_template("index.html", images=image_names)

@app.route('/test/<filename>')
def serve_test_image(filename):
    return send_from_directory(TEST_IMAGES_FOLDER, filename)

@app.route('/output/predict/<filename>')
def serve_output_predict_image(filename):
    # Serve images inside output/predict folder
    predict_folder = os.path.join(OUTPUT_FOLDER, "predict")
    return send_from_directory(predict_folder, filename)

@app.route('/detect', methods=['POST'])
def detect():
    try:
        data = request.get_json()
        conf = float(data.get('conf', 0.2))
        iou = float(data.get('iou', 0.3))
        image_name = data.get('image')

        if not image_name:
            return jsonify({'success': False, 'error': 'No image specified'}), 400

        image_path = os.path.join(TEST_IMAGES_FOLDER, image_name)
        if not os.path.isfile(image_path):
            return jsonify({'success': False, 'error': 'Image file does not exist'}), 400

        saved_images = predict_on_images([image_path], MODEL_PATH, conf=conf, iou=iou)
        saved_image_name = os.path.basename(saved_images[0])

        # IMPORTANT: Return the filename relative to 'predict' folder since you serve via /output/predict/<filename>
        return jsonify({'success': True, 'output_image': saved_image_name})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No selected file'}), 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    return jsonify({'success': True, 'filename': file.filename})    
    
def cleanup_output_folder():
    if os.path.exists(OUTPUT_FOLDER):
        print(f"Cleaning up entire output folder: {OUTPUT_FOLDER}")
        shutil.rmtree(OUTPUT_FOLDER)
        
def cleanup_upload_folder():
    if os.path.exists(UPLOAD_FOLDER):
        print(f"Cleaning up entire upload folder: {UPLOAD_FOLDER}")
        shutil.rmtree(UPLOAD_FOLDER)
atexit.register(cleanup_output_folder)
atexit.register(cleanup_upload_folder)

if __name__ == '__main__':
    app.run(debug=True)
