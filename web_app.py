from flask import Flask, render_template, request, jsonify
import os
import numpy as np
from utils import load_trained_model, preprocess_image

app = Flask(__name__)

# Ensure 'uploads' folder exists
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load trained model
model = load_trained_model()

# Class labels
shape_classes = ["Circle", "Square", "Triangle", "Pentagon", "Ellipse"]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # Preprocess the uploaded image
    processed_img = preprocess_image(filepath)  # Should return shape: (1, 256, 256, 3)

    # Predict shape (no extra expand_dims needed!)
    predictions = model.predict(processed_img)
    shape_idx = np.argmax(predictions[0])
    predicted_shape = shape_classes[shape_idx]

    return jsonify({"shape": predicted_shape})

if __name__ == "__main__":
    app.run(debug=True)
