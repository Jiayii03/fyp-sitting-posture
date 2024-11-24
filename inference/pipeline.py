"""
This script demonstrates the full pipeline for posture classification using a trained MLP model.

The pipeline consists of the following steps:
1. Load an image and extract pose keypoints using Mediapipe.
2. Load a trained MLP model for posture classification.
3. Normalize the extracted keypoints using a StandardScaler.
4. Perform inference using the model to predict the posture.
5. Display the image with the extracted keypoints and predicted posture.

Change the `image_path` and `model_path` variables to the paths of your image and saved model, respectively.

Run the script using the following command:
python pipeline.py --image crossed_legs_1.jpg
"""


import sys
import cv2
import mediapipe as mp
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import argparse

# Add the directory to sys.path
model_dir = "../models/2024-11-24_16-34-03"
sys.path.append(model_dir)

# Import the MLP class
from model import MLP

# Define the landmarks to extract
needed_landmarks = [
    mp.solutions.pose.PoseLandmark.NOSE,
    mp.solutions.pose.PoseLandmark.LEFT_SHOULDER,
    mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER,
    mp.solutions.pose.PoseLandmark.LEFT_HIP,
    mp.solutions.pose.PoseLandmark.RIGHT_HIP,
    mp.solutions.pose.PoseLandmark.LEFT_KNEE,
    mp.solutions.pose.PoseLandmark.RIGHT_KNEE,
    mp.solutions.pose.PoseLandmark.LEFT_ANKLE,
    mp.solutions.pose.PoseLandmark.RIGHT_ANKLE,
]

# Define custom connections for drawing
custom_connections = [
    (0, 9),  # Nose to shoulder midpoint
    (1, 2),  # Left shoulder to right shoulder
    (1, 3),  # Left shoulder to left hip
    (2, 4),  # Right shoulder to right hip
    (3, 4),  # Left hip to right hip
    (3, 5),  # Left hip to left knee
    (4, 6),  # Right hip to right knee
    (5, 7),  # Left knee to left ankle
    (6, 8),  # Right knee to right ankle
]

# Function to extract keypoints from an image
def extract_keypoints(image_path, pose_model, visibility_threshold=0.65):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose_model.process(image_rgb)

    if not results.pose_landmarks:
        print("No pose detected in the image.")
        return None, image

    keypoints = []
    landmarks = results.pose_landmarks.landmark

    # Calculate shoulder midpoint
    left_shoulder = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
    shoulder_midpoint = type("Landmark", (), {
        "x": (left_shoulder.x + right_shoulder.x) / 2,
        "y": (left_shoulder.y + right_shoulder.y) / 2,
        "visibility": min(left_shoulder.visibility, right_shoulder.visibility)
    })

    h, w, _ = image.shape

    # Extract needed landmarks
    for landmark_enum in needed_landmarks:
        landmark = landmarks[landmark_enum]
        if landmark.visibility < visibility_threshold:
            keypoints.append([0, 0])  # Mark low visibility points as (0, 0)
        else:
            keypoints.append([landmark.x, landmark.y])

    # Add shoulder midpoint
    if shoulder_midpoint.visibility < visibility_threshold:
        keypoints.append([0, 0])
    else:
        keypoints.append([shoulder_midpoint.x, shoulder_midpoint.y])

    keypoints = np.array(keypoints)

    # Draw keypoints and connections on the image
    for i, (x, y) in enumerate(keypoints):
        if x != 0 and y != 0:  # Only draw visible keypoints
            cv2.circle(image, (int(x * w), int(y * h)), 5, (0, 255, 0), -1)

    for start_idx, end_idx in custom_connections:
        start_point = keypoints[start_idx]
        end_point = keypoints[end_idx]
        if all(start_point) and all(end_point):  # Both points must be visible
            cv2.line(
                image,
                (int(start_point[0] * w), int(start_point[1] * h)),
                (int(end_point[0] * w), int(end_point[1] * h)),
                (255, 0, 0),
                2,
            )

    return keypoints.flatten(), image


# Function for model inference
def predict_posture(model, keypoints, scaler, class_labels):
    keypoints_normalized = scaler.transform([keypoints])  # StandardScaler expects 2D array

    # Convert to PyTorch tensor
    input_tensor = torch.tensor(keypoints_normalized, dtype=torch.float32)

    # Perform inference
    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)

    # Map predicted index to class label
    return class_labels[predicted.item()]


# Main pipeline
if __name__ == "__main__":
    # Argument parser for image name
    parser = argparse.ArgumentParser(description="Posture classification pipeline.")
    parser.add_argument("--image", type=str, required=True, help="Name of the image file in the images/ directory.")
    args = parser.parse_args()

    # Paths and configurations
    base_image_dir = "images"
    image_path = f"{base_image_dir}/{args.image}"  # Combine base directory and image name
    model_path = "../models/2024-11-24_16-34-03/epochs_150_lr_1e-03_wd_1e-03_acc_8298.pth"  # Replace with your saved model path
    class_labels = ["crossed_legs", "proper", "reclining", "slouching"]  # Your class labels

    # Load Mediapipe pose model
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

    # Extract keypoints from the image
    keypoints, image_with_keypoints = extract_keypoints(image_path, pose)

    if keypoints is not None:
        # Load the saved model
        input_size = len(keypoints)  # Length of the input keypoints vector
        model = MLP(input_size=input_size, num_classes=len(class_labels))
        model.load_state_dict(torch.load(model_path, weights_only=False))
        model.eval()

        # Load the StandardScaler used during training
        scaler = StandardScaler()
        scaler.mean_ = np.load("../models/2024-11-24_16-34-03/scaler_mean.npy")  # Load scaler's mean
        scaler.scale_ = np.load("../models/2024-11-24_16-34-03/scaler_scale.npy")  # Load scaler's scale
    
        # Predict the posture
        predicted_label = predict_posture(model, keypoints, scaler, class_labels)
        print(f"Predicted posture: {predicted_label}")

        # Convert BGR to RGB for Matplotlib
        image_rgb = cv2.cvtColor(image_with_keypoints, cv2.COLOR_BGR2RGB)

        # Display the image using Matplotlib
        plt.imshow(image_rgb)
        plt.axis("off")
        plt.title("Pose Estimation")
        plt.show()
    else:
        print("Pose could not be extracted.")

    # Clean up
    pose.close()