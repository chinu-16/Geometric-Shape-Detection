# Geometric-Shape-Detection
A deep learning-based application for detecting geometric shapes (circle, square, triangle, etc.) in images using a Convolutional Neural Network (CNN). The project includes synthetic dataset generation, bounding box labeling, model training, real-time detection, and a Flask-based web interface for image uploads.
# 🧠 Geometric Shape Detection using CNN with Bounding Box Labeling

This project leverages a Convolutional Neural Network (CNN) to detect and classify basic geometric shapes in images. It supports bounding box localization and includes a user-friendly web interface for uploading images and visualizing predictions.

---

## 📌 Features

- Detects basic geometric shapes: Circle, Square, Triangle, Pentagon, Ellipse
- Generates a synthetic dataset with labeled shapes
- Trains a CNN for shape classification and bounding box regression
- Real-time shape detection via webcam
- Flask-based web app for easy image uploads and prediction
- Drag-and-drop HTML interface with preview support

---

## 🛠️ Tech Stack

- **Backend:** Python, TensorFlow, OpenCV
- **Frontend:** HTML, CSS, JavaScript (vanilla)
- **Web Framework:** Flask
- **Model:** CNN for shape classification & localization

---

## 📁 Project Structure

GeometricShapeDetection/
│
├── generation.py              # Script to generate synthetic dataset of geometric shapes
├── model_training.py          # Train the CNN model for shape detection and bounding box prediction
├── predict.py                 # Perform prediction on test images using trained model
├── real_time_detection.py     # Detect shapes in real-time using webcam
├── web_app.py                 # Flask web application backend
├── utils.py                   # Utility functions (preprocessing, image loading, etc.)
├── config.py                  # Configuration file for image size and dataset paths
├── requirements.txt           # Project dependencies
│
├── templates/
│   ├── index.html             # Web UI for image upload and shape prediction
│   └── result.html            # (Optional) Page to display detailed prediction results
│
├── uploads/                   # Directory for storing uploaded images (auto-created)
├── trained_model/             # Folder containing the trained model file (.h5)
│
└── .gitignore                 # Ignore virtual env, model files, cache, etc.


---

## 🚀 How to Run

1. Clone the Repository
git clone https://github.com/yourusername/GeometricShapeDetection.git
cd GeometricShapeDetection

2. Create Virtual Environment
python -m venv venv

3. Activate Virtual Environment (Windows)
venv\Scripts\activate

(For macOS/Linux users use this instead)
 source venv/bin/activate

4. Install Required Packages
pip install -r requirements.txt

5. Generate Synthetic Dataset
python generation.py

6. Train the CNN Model
python model_training.py

7. Run Prediction on a Test Image
python predict.py --image test_image_3.png
 (Replace test_image_3.png with your test image)

8. Run Real-Time Shape Detection using Webcam
python real_time_detection.py

9. Launch Flask Web App
python web_app.py
 Visit http://127.0.0.1:5000 in your browser

(Optional) Deactivate virtual environment when done
deactivate

