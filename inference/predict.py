"""
This script processes all the video frames saved in `frame_dir`, detects the pose using MediaPipe, and predicts the posture using a trained PyTorch model. The results are saved in a JSON file in the specified output directory. Optionally, annotated images with keypoints and connections can be saved as well.

run with CLI arguments:
python predict.py --frames_dir <path_to_frames> --output_dir <path_to_output> --model_dir <path_to_model> --model_file <model_filename> --reclining_sensitivity <value> --crossed_legs_sensitivity <value> --proper_sensitivity <value> --slouching_sensitivity <value>

or use the default values in the script:
python predict.py

"""

import os
import json
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import cv2
import mediapipe as mp
import torch.nn.functional as F
import time
import sys
import argparse

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

class MLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 512),  # Increase neurons
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.model(x)

def extract_keypoints(image_input, pose_model, visibility_threshold=0.65):
    """
    Extract keypoints from an image or frame.

    Args:
    - image_input: Path to the image file or a numpy array (frame).
    - pose_model: Mediapipe pose model instance.
    - visibility_threshold: Minimum visibility to consider a keypoint valid.

    Returns:
    - Flattened keypoints array or None if no keypoints detected.
    - Processed image with keypoints and connections drawn.
    """
    if isinstance(image_input, str):  # Check if it's a file path
        image = cv2.imread(image_input)
    else:  # Assume it's already an image (numpy array)
        image = image_input

    if image is None:
        raise ValueError("Invalid image input. Ensure the file path or frame is correct.")

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

def process_frames(frames_dir, output_dir, model_path, scaler_mean_path, scaler_scale_path, sensitivity_adjustments=None):
    """
    Process all frames in the frames_dir, run posture detection, and save results to output_dir.
    """
    # Make sure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Initialize MediaPipe pose model
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
    
    # Load the model
    class_labels = ["crossed_legs", "proper", "reclining", "slouching"]
    
    # Get list of frame files
    frame_files = [f for f in os.listdir(frames_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if not frame_files:
        print("No frames found in the directory.")
        return
    
    # Process a sample frame to determine input size
    sample_frame_path = os.path.join(frames_dir, frame_files[0])
    sample_keypoints, _ = extract_keypoints(sample_frame_path, pose)
    
    if sample_keypoints is None:
        print(f"Could not detect pose in sample frame {frame_files[0]}. Skipping.")
        return
    
    input_size = len(sample_keypoints)
    
    # Load the model
    model = MLP(input_size=input_size, num_classes=len(class_labels))
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    
    # Load the StandardScaler
    scaler = StandardScaler()
    scaler.mean_ = np.load(scaler_mean_path)
    scaler.scale_ = np.load(scaler_scale_path)
    
    # Initialize results dictionary
    results = {}
    
    # Set default sensitivity adjustments if none provided
    if sensitivity_adjustments is None:
        sensitivity_adjustments = {
            "reclining": 1.0,
            "crossed_legs": 1.0,
            "proper": 1.0,
            "slouching": 1.0
        }
    
    # Process each frame
    print(f"Processing {len(frame_files)} frames...")
    start_time = time.time()
    
    for frame_file in sorted(frame_files):
        print(f"Processing {frame_file}...")
        
        # Read the frame
        frame_path = os.path.join(frames_dir, frame_file)
        
        try:
            # Extract keypoints
            keypoints, image_with_keypoints = extract_keypoints(frame_path, pose)
            
            if keypoints is None:
                print(f"Warning: No pose detected in {frame_file}")
                results[frame_file] = {"prediction": "no_detection", "confidence_scores": {}}
                continue
            
            # Predict posture
            predicted_label, confidence_scores = predict_posture(
                model, keypoints, scaler, class_labels, sensitivity_adjustments
            )
            
              # Save results to JSON file
            video_name = frames_dir.split("/")[-1]
            # Save the annotated image (optional)
            annotated_base_dir = os.path.join(output_dir, "annotated_frames")
            annotated_dir = os.path.join(annotated_base_dir, video_name)
            if not os.path.exists(annotated_dir):
                os.makedirs(annotated_dir)
                
            # Add prediction text to image
            h, w, _ = image_with_keypoints.shape
            text_position = (10, 30)
            cv2.putText(
                image_with_keypoints,
                f"Prediction: {predicted_label}",
                text_position,
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )
            
            cv2.imwrite(os.path.join(annotated_dir, frame_file), image_with_keypoints)
            
            # Store results
            results[frame_file] = {
                "prediction": predicted_label,
                "confidence_scores": confidence_scores
            }
            
            print(f"  Predicted: {predicted_label}")
            print(f"  Confidence scores: {confidence_scores}")
            
        except Exception as e:
            print(f"Error processing {frame_file}: {str(e)}")
            results[frame_file] = {"prediction": "error", "confidence_scores": {}, "error": str(e)}
    
    end_time = time.time()
    print(f"Total processing time: {end_time - start_time:.2f} seconds")
    
    output_file = os.path.join(output_dir, f"posture_predictions_{video_name}.json")
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Results saved to {output_file}")
    
    # Clean up
    pose.close()
    
    return results

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Process frames for posture detection.")
    parser.add_argument("--frames_dir", type=str, default="../../datasets/recordings/extracted_frames/IMG_4115", help="Directory containing extracted frames")
    parser.add_argument("--output_dir", type=str, default="../experiments/recording_output", help="Directory to save prediction results")
    parser.add_argument("--model_dir", type=str, default="../models/2025-04-03_23-55-32", help="Directory containing model files")
    parser.add_argument("--model_file", type=str, default="epochs_300_lr_1e-03_wd_5e-03_acc_8963.pth", help="Model filename")
    parser.add_argument("--reclining_sensitivity", type=float, default=1.0, help="Sensitivity adjustment for reclining class")
    parser.add_argument("--crossed_legs_sensitivity", type=float, default=1.0, help="Sensitivity adjustment for crossed_legs class")
    parser.add_argument("--proper_sensitivity", type=float, default=1.0, help="Sensitivity adjustment for proper class")
    parser.add_argument("--slouching_sensitivity", type=float, default=1.0, help="Sensitivity adjustment for slouching class")
    
    args = parser.parse_args()
    
    # Set up the paths
    frames_dir = args.frames_dir
    output_dir = args.output_dir
    model_path = os.path.join(args.model_dir, args.model_file)
    scaler_mean_path = os.path.join(args.model_dir, "scaler_mean.npy")
    scaler_scale_path = os.path.join(args.model_dir, "scaler_scale.npy")
    
    # Set sensitivity adjustments
    sensitivity_adjustments = {
        "reclining": args.reclining_sensitivity,
        "crossed_legs": args.crossed_legs_sensitivity,
        "proper": args.proper_sensitivity,
        "slouching": args.slouching_sensitivity
    }
    
    print("Starting posture detection pipeline with the following settings:")
    print(f"Frames directory: {frames_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Model path: {model_path}")
    print(f"Sensitivity adjustments: {sensitivity_adjustments}")
    
    # Process frames
    results = process_frames(
        frames_dir, 
        output_dir, 
        model_path, 
        scaler_mean_path, 
        scaler_scale_path,
        sensitivity_adjustments
    )
    
    # Print summary
    if results:
        posture_counts = {}
        for frame, data in results.items():
            prediction = data["prediction"]
            posture_counts[prediction] = posture_counts.get(prediction, 0) + 1
        
        print("\nSummary of predictions:")
        for posture, count in posture_counts.items():
            print(f"{posture}: {count} frames")