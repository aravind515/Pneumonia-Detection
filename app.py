from flask import Flask, request, jsonify
import torch
import pickle
import cv2
import numpy as np
import os
from torchvision import transforms
from werkzeug.utils import secure_filename

# -----------------------------
# Flask App Setup
# -----------------------------
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# -----------------------------
# Load Model
# -----------------------------
model_path = os.path.join("models", "efficient_net.pkl")

try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    model.eval()
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    exit(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# -----------------------------
# Image Preprocessing
# -----------------------------
IMG_SIZE = (224, 224)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225])
])
OPTIMAL_THRESHOLD = 0.78

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# -----------------------------
# Prediction Logic
# -----------------------------
def predict_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, IMG_SIZE)
    image = np.stack([image] * 3, axis=-1)  # Convert grayscale to RGB
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image).squeeze(1)
        prediction = torch.sigmoid(output).item()
        label = "PNEUMONIA" if prediction > OPTIMAL_THRESHOLD else "NORMAL"
        confidence = max(prediction, 1 - prediction)
    return label, confidence

# -----------------------------
# API Endpoint
# -----------------------------
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        label, confidence = predict_image(filepath)
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
    finally:
        # Clean up the uploaded file
        if os.path.exists(filepath):
            os.remove(filepath)

    return jsonify({
        "label": label,
        "confidence": round(confidence, 2)
    })

# -----------------------------
# Run App
# -----------------------------
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
