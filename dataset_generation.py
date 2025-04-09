import cv2
import numpy as np
import os
import random
import json
from multiprocessing import Pool

# Define save directory
save_dir = "dataset"
os.makedirs(save_dir, exist_ok=True)

# Debugging: Print the absolute path of the dataset folder
print("Dataset will be saved in:", os.path.abspath(save_dir))

# Shape generation function
def generate_shape(image_size=(256, 256)):
    image = np.ones((*image_size, 3), dtype=np.uint8) * 255  # White background
    shape_types = ["circle", "square", "triangle", "pentagon", "ellipse"]
    shape = random.choice(shape_types)

    # Generate random bounding box coordinates
    x1, y1 = random.randint(20, 180), random.randint(20, 180)
    x2, y2 = x1 + random.randint(30, 60), y1 + random.randint(30, 60)
    color = tuple(random.randint(0, 255) for _ in range(3))

    if shape == "circle":
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        radius = (x2 - x1) // 2
        cv2.circle(image, center, radius, color, -1)

    elif shape == "square":
        cv2.rectangle(image, (x1, y1), (x2, y2), color, -1)

    elif shape == "triangle":
        pts = np.array([[x1, y2], [(x1 + x2) // 2, y1], [x2, y2]], np.int32)
        cv2.fillPoly(image, [pts], color)

    elif shape == "pentagon":
        pts = np.array([
            [x1, y2], [(x1 + x2) // 3, y1], [(x1 + x2) * 2 // 3, y1],
            [x2, (y1 + y2) // 2], [(x1 + x2) // 2, y2]
        ], np.int32)
        cv2.fillPoly(image, [pts], color)

    elif shape == "ellipse":
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        axes = ((x2 - x1) // 2, (y2 - y1) // 2)
        cv2.ellipse(image, center, axes, 0, 0, 360, color, -1)

    return image, shape, x1, y1, x2, y2

# Function to save image and label
def save_image(args):
    index, save_dir = args
    image, shape, x1, y1, x2, y2 = generate_shape()
    filename = os.path.join(save_dir, f"image_{index}.png")
    cv2.imwrite(filename, image)

    # Debugging: Check if image is saved successfully
    if os.path.exists(filename):
        print(f"✅ Image saved: {filename}")
    else:
        print(f"❌ Image failed to save: {filename}")

    return {"filename": filename, "shape": shape, "x1": x1, "y1": y1, "x2": x2, "y2": y2}

# Function to generate dataset
def generate_dataset(image_size=(256, 256), num_images=1000, save_dir="dataset"):
    os.makedirs(save_dir, exist_ok=True)
    print(f"Generating {num_images} images...")

    with Pool() as pool:
        labels = pool.map(save_image, [(i, save_dir) for i in range(num_images)])

    # Save labels to JSON file
    labels_path = os.path.join(save_dir, "labels.json")
    with open(labels_path, "w") as f:
        json.dump(labels, f, indent=4)

    print(f"✅ Dataset generation complete! {num_images} images saved in {save_dir}")
    print(f"📂 Labels saved in: {labels_path}")

# Execute dataset generation
if __name__ == "__main__":
    generate_dataset()
