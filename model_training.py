import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.preprocessing.image import load_img, img_to_array # type: ignore
import numpy as np
import json
import os

# Load dataset annotations
def load_data(json_path="dataset/labels.json"):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"❌ Error: Labels file {json_path} not found!")

    with open(json_path, "r") as f:
        labels = json.load(f)

    print(f"✅ Loaded {len(labels)} labeled images.")
    return labels

# Prepare dataset
def preprocess_data(labels, img_size=(256, 256)):
    images, shape_labels, bbox_labels = [], [], []
    shape_mapping = {"circle": 0, "square": 1, "triangle": 2, "pentagon": 3, "ellipse": 4}

    for item in labels:
        if not os.path.exists(item["filename"]):
            print(f"⚠️ Warning: Image file {item['filename']} not found. Skipping.")
            continue

        img = load_img(item["filename"], target_size=img_size)
        img_array = img_to_array(img) / 255.0  # Normalize image
        images.append(img_array)

        shape_labels.append(shape_mapping[item["shape"]])
        bbox_labels.append([item["x1"], item["y1"], item["x2"], item["y2"]])

    images = np.array(images)
    shape_labels = np.array(shape_labels)
    bbox_labels = np.array(bbox_labels)

    print(f"✅ Dataset Prepared: {images.shape}, Shapes: {shape_labels.shape}, BBoxes: {bbox_labels.shape}")
    return images, shape_labels, bbox_labels

# Load and preprocess data
labels = load_data()
X, y_shapes, y_bboxes = preprocess_data(labels)

# Split into training and validation sets
split_idx = int(0.8 * len(X))
X_train, X_val = X[:split_idx], X[split_idx:]
y_shapes_train, y_shapes_val = y_shapes[:split_idx], y_shapes[split_idx:]
y_bboxes_train, y_bboxes_val = y_bboxes[:split_idx], y_bboxes[split_idx:]

print(f"✅ Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

# Define CNN model for shape classification
shape_model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(256,256,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(5, activation='softmax')  # 5 shape classes
])

shape_model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train shape classification model
print("🚀 Training Shape Classification Model...")
shape_model.fit(X_train, y_shapes_train, epochs=10, validation_data=(X_val, y_shapes_val))

# Save shape model
shape_model.save("shape_classification_model.h5")
print("✅ Shape Classification Model saved as shape_classification_model.h5")

# Define CNN model for bounding box regression
bbox_model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(256,256,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='linear')  # 4 values for bounding box (x1, y1, x2, y2)
])

bbox_model.compile(optimizer=Adam(), loss='mean_squared_error', metrics=['mae'])

# Train bounding box regression model
print("🚀 Training Bounding Box Model...")
bbox_model.fit(X_train, y_bboxes_train, epochs=10, validation_data=(X_val, y_bboxes_val))

# Save bounding box model
bbox_model.save("bounding_box_model.h5")
print("✅ Bounding Box Model saved as bounding_box_model.h5")
