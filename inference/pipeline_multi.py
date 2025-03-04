"""
This script demonstrates the full pipeline for posture classification using a trained MLP model (MULTI-PERSON VERSION).

The pipeline consists of the following steps:
1. Load an image and detect persons using YOLOv5.
2. Crop the detected persons and run Mediapipe pose estimation on the cropped regions.
3. Extract pose landmarks and visualize them on the original image.
4. Load a trained MLP model for posture classification.
5. Normalize the extracted keypoints and perform inference with the model.
6. Display the image with bounding boxes, pose landmarks, and predicted postures.

Change the `image_path` and `model_path` variables to the paths of your image and saved model, respectively.

Run the script using the following command:
python pipeline_multi.py --image crowded_3.jpg

--image: Name of the image file in the images/ directory.
--save: Save the annotated image with the predicted postures.
--reclining_sensitivity: Sensitivity adjustment for the "reclining" class (default: 1.0).
--crossed_legs_sensitivity: Sensitivity adjustment for the "crossed_legs" class (default: 1.0).
--proper_sensitivity: Sensitivity adjustment for the "proper" class (default: 1.0).
--slouching_sensitivity: Sensitivity adjustment for the "slouching" class (default: 1.0).
"""

import sys
import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import argparse
import time

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

def extract_keypoints_multi_person(image_input, pose_model, yolo_model, confidence_threshold=0.30, visibility_threshold=0.65):
    """
    Extract keypoints for multiple persons using YOLO for detection and Mediapipe for pose estimation.

    Args:
    - image_input: Path to the image file or a numpy array (frame).
    - pose_model: Mediapipe pose model instance.
    - yolo_model: YOLO model for person detection.
    - confidence_threshold: Minimum confidence for YOLO detection.
    - visibility_threshold: Minimum visibility to consider a keypoint valid.

    Returns:
    - A list of keypoints arrays, each corresponding to a detected person.
    - Processed image with bounding boxes and connections drawn.
    """
    if isinstance(image_input, str):  # Check if it's a file path
        image = cv2.imread(image_input)
    else:  # Assume it's already an image (numpy array)
        image = image_input

    if image is None:
        raise ValueError("Invalid image input. Ensure the file path or frame is correct.")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]

    # YOLO detection
    results = yolo_model(image_rgb)
    person_detections = results.pandas().xyxy[0]

    keypoints_list = []
    bboxes = []
    person = 1
    for _, detection in person_detections.iterrows():
        x_min, y_min, x_max, y_max, confidence, class_id, name = detection

        # Ensure the detected object is a person
        if name == "person" and confidence > confidence_threshold:
            x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
            bboxes.append((x_min, y_min, x_max, y_max))  # Save bounding box for later use

            # Draw bounding box
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(
                image,
                f"Person {person}: {confidence:.2f}",
                (x_min, y_min - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            # Crop the person from the image
            cropped_person = image[y_min:y_max, x_min:x_max]

            # Run Mediapipe pose estimation
            cropped_rgb = cv2.cvtColor(cropped_person, cv2.COLOR_BGR2RGB)
            results = pose_model.process(cropped_rgb)

            if results.pose_landmarks:
                keypoints_cropped = []  # Keypoints in the cropped coordinate system
                keypoints_scaled = []  # Keypoints scaled to the original image size
                landmarks = results.pose_landmarks.landmark

                # Extract needed landmarks
                for landmark_enum in needed_landmarks:
                    landmark = landmarks[landmark_enum]
                    if landmark.visibility < visibility_threshold:
                        keypoints_cropped.append([0, 0])  # Mark low visibility points as (0, 0)
                        keypoints_scaled.append([0, 0])  # Mark low visibility points as (0, 0)
                    else:
                        # Original cropped keypoints for prediction
                        keypoints_cropped.append([landmark.x, landmark.y])

                        # Scaled keypoints for visualization
                        keypoints_scaled.append([
                            x_min + landmark.x * (x_max - x_min),
                            y_min + landmark.y * (y_max - y_min)
                        ])

                # Add shoulder midpoint (both cropped and scaled versions)
                left_shoulder_cropped = keypoints_cropped[needed_landmarks.index(mp.solutions.pose.PoseLandmark.LEFT_SHOULDER)]
                right_shoulder_cropped = keypoints_cropped[needed_landmarks.index(mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER)]
                left_shoulder_scaled = keypoints_scaled[needed_landmarks.index(mp.solutions.pose.PoseLandmark.LEFT_SHOULDER)]
                right_shoulder_scaled = keypoints_scaled[needed_landmarks.index(mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER)]

                if all(left_shoulder_cropped) and all(right_shoulder_cropped):
                    # Cropped shoulder midpoint
                    shoulder_midpoint_cropped = [
                        (left_shoulder_cropped[0] + right_shoulder_cropped[0]) / 2,
                        (left_shoulder_cropped[1] + right_shoulder_cropped[1]) / 2
                    ]
                else:
                    shoulder_midpoint_cropped = [0, 0]

                if all(left_shoulder_scaled) and all(right_shoulder_scaled):
                    # Scaled shoulder midpoint
                    shoulder_midpoint_scaled = [
                        (left_shoulder_scaled[0] + right_shoulder_scaled[0]) / 2,
                        (left_shoulder_scaled[1] + right_shoulder_scaled[1]) / 2
                    ]
                else:
                    shoulder_midpoint_scaled = [0, 0]

                keypoints_cropped.append(shoulder_midpoint_cropped)
                keypoints_scaled.append(shoulder_midpoint_scaled)

                # Add flattened cropped keypoints for prediction
                keypoints_cropped_flat = np.array(keypoints_cropped).flatten()
                keypoints_list.append(keypoints_cropped_flat)

                # Draw visualization using scaled keypoints
                for i, (x, y) in enumerate(keypoints_scaled):
                    if x != 0 and y != 0:  # Only draw visible keypoints
                        cv2.circle(image, (int(x), int(y)), 5, (0, 255, 0), -1)

                for start_idx, end_idx in custom_connections:
                    start_point = keypoints_scaled[start_idx]
                    end_point = keypoints_scaled[end_idx]
                    if all(start_point) and all(end_point):  # Both points must be visible
                        cv2.line(
                            image,
                            (int(start_point[0]), int(start_point[1])),
                            (int(end_point[0]), int(end_point[1])),
                            (255, 0, 0),
                            2,
                        )
            else:
                keypoints_list.append(None)
                
            person += 1

    return keypoints_list, image, bboxes


def predict_posture(model, keypoints, scaler, class_labels, sensitivity_adjustments=None):
    """
    Perform inference with sensitivity adjustments for specific classes.

    Args:
    - model: Trained PyTorch model.
    - keypoints: Keypoints extracted from the image.
    - scaler: StandardScaler used for normalization.
    - class_labels: List of class labels.
    - sensitivity_adjustments: Dictionary of class sensitivities (e.g., {'reclining': 0.8}).

    Returns:
    - predicted_label: Predicted class label.
    - confidence_scores: Dictionary of confidence scores for each class.
    """
    # Normalize keypoints
    keypoints_normalized = scaler.transform([keypoints])  # StandardScaler expects 2D array

    # Convert to PyTorch tensor
    input_tensor = torch.tensor(keypoints_normalized, dtype=torch.float32)

    # Perform inference
    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)  # Compute softmax to get probabilities

    # Adjust sensitivities if specified
    if sensitivity_adjustments:
        for class_name, adjustment_factor in sensitivity_adjustments.items():
            class_index = class_labels.index(class_name)
            probabilities[0, class_index] *= adjustment_factor

        # Re-normalize probabilities after adjustments
        probabilities /= probabilities.sum(dim=1, keepdim=True)

    # Get the predicted label and confidence scores
    _, predicted = torch.max(probabilities, 1)
    predicted_label = class_labels[predicted.item()]
    confidence_scores = {class_labels[i]: probabilities[0, i].item() for i in range(len(class_labels))}

    return predicted_label, confidence_scores

# Main pipeline
if __name__ == "__main__":
    # Argument parser for image name
    parser = argparse.ArgumentParser(description="Posture classification pipeline.")
    parser.add_argument("--image", type=str, required=True, help="Name of the image file in the images/ directory.")
    parser.add_argument("--save", action="store_true", help="Save the annotated image with predicted postures.")
    parser.add_argument("--reclining_sensitivity", type=float, default=1.0, help="Sensitivity adjustment for reclining class.")
    parser.add_argument("--crossed_legs_sensitivity", type=float, default=1.0, help="Sensitivity adjustment for crossed_legs class.")
    parser.add_argument("--proper_sensitivity", type=float, default=1.0, help="Sensitivity adjustment for proper class.")
    parser.add_argument("--slouching_sensitivity", type=float, default=1.0, help="Sensitivity adjustment for slouching class.")
    args = parser.parse_args()
    
    # Set the path to the YOLOv5 model
    yolo_model_path = "../models/yolo-human-detection/yolov5s.pt"

    # Load YOLOv5 model from the specified local path
    yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path=yolo_model_path, force_reload=False)
    yolo_model.classes = [0]  # Only detect persons (COCO class ID: 0)

    # Paths and configurations
    base_image_dir = "images"
    image_path = f"{base_image_dir}/{args.image}"  # Combine base directory and image name
    sitting_posture_model_path = "../models/2024-11-24_16-34-03/epochs_150_lr_1e-03_wd_1e-03_acc_8298.pth"  # Replace with your saved model path
    class_labels = ["crossed_legs", "proper", "reclining", "slouching"]  # Your class labels

    # Load Mediapipe pose model
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
    
    # Start timing
    start_time = time.time()

    # Extract keypoints list from the image
    keypoints_list, image_with_keypoints, bboxes = extract_keypoints_multi_person(image_path, pose, yolo_model)

    if keypoints_list:
        # Load the saved model
        input_size = len(keypoints_list[0])  # Length of the input keypoints vector
        model = MLP(input_size=input_size, num_classes=len(class_labels))
        model.load_state_dict(torch.load(sitting_posture_model_path, weights_only=False))
        model.eval()

        # Load the StandardScaler used during training
        scaler = StandardScaler()
        scaler.mean_ = np.load("../models/2024-11-24_16-34-03/scaler_mean.npy")
        scaler.scale_ = np.load("../models/2024-11-24_16-34-03/scaler_scale.npy")

        # Sensitivity adjustments
        sensitivity_adjustments = {
            "reclining": args.reclining_sensitivity,
            "crossed_legs": args.crossed_legs_sensitivity,
            "proper": args.proper_sensitivity,
            "slouching": args.slouching_sensitivity,
        }

        # Predict the posture for each person
        for i, (keypoints, bbox) in enumerate(zip(keypoints_list, bboxes)):
            if keypoints is None or np.isnan(keypoints).any():  # Skip if keypoints are invalid
                print(f"Person {i + 1} - No valid keypoints detected. Skipping posture prediction.")
                continue
    
            predicted_label, confidence_scores = predict_posture(
                model, keypoints, scaler, class_labels, sensitivity_adjustments=sensitivity_adjustments
            )

            print(f"Person {i + 1} - Predicted posture: {predicted_label}")
            print(f"Confidence scores: {confidence_scores}")

            # Add predicted label and confidence score to the bounding box
            x_min, y_min, x_max, y_max = bbox
            cv2.putText(
                image_with_keypoints,
                f"{predicted_label}: {max(confidence_scores.values()):.2f}",
                (x_min, y_min - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 165, 255),
                2
            )

        # End timing
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Inference pipeline runtime: {elapsed_time:.4f} seconds")

        # Resize the image to 50% of its original size
        resized_image_with_keypoints = cv2.resize(image_with_keypoints, (0, 0), fx=0.8, fy=0.8)
        # Display the image with YOLO bounding boxes and predictions
        cv2.imshow("YOLO Detected Humans With Keypoints and Postures", resized_image_with_keypoints)
        
        # Save the annotated image if specified
        saved_image_dir = "results"
        if args.save:
            output_image_path = f"{saved_image_dir}/{args.image}"
            cv2.imwrite(output_image_path, resized_image_with_keypoints)
            print(f"Annotated image saved as {output_image_path}")
            
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No persons detected or no pose detected.")

    # Clean up
    pose.close()