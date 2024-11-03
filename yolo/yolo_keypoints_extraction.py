# pip install ultralytics opencv-python numpy
"""
This script tests YOLOv8 pose estimation on a single image.
It extracts only specific keypoints (nose, shoulders, hips, knees, ankles, and shoulder midpoint).
It also creates two images with the keypoints drawn on them: one with a white background and one with a black background.
It only saves high confidence keypoints (visibility > 0.65).
Save the keypoints to a CSV file.

run python keypoints_extraction_test.py class_name image_name
"""

import cv2
import numpy as np
from ultralytics import YOLO
import argparse
import os
import pandas as pd
import time  # Add this import

# Directory paths
raw_dir = "C:/Users/User/Documents/UNM_CSAI/UNM_current_modules/COMP3025_Individual_Dissertation/dev/datasets/raw"
keypoints_dir = "C:/Users/User/Documents/UNM_CSAI/UNM_current_modules/COMP3025_Individual_Dissertation/dev/datasets/keypoints"
keypoints_only_dir = "C:/Users/User/Documents/UNM_CSAI/UNM_current_modules/COMP3025_Individual_Dissertation/dev/datasets/keypoints_only"
vectors_dir = "C:/Users/User/Documents/UNM_CSAI/UNM_current_modules/COMP3025_Individual_Dissertation/dev/datasets/vectors"

def save_keypoints_to_csv(keypoints, class_name, image_id, output_path):
    # Create a row with class, image_id, and all keypoints
    row_data = {
        'class': class_name,
        'image_id': image_id,
    }
    
    # List of point names in order (same as MediaPipe implementation)
    point_names = [
        'nose',
        'left_shoulder',
        'right_shoulder',
        'left_hip',
        'right_hip',
        'left_knee',
        'right_knee',
        'left_ankle',
        'right_ankle',
        'shoulder_midpoint'
    ]
    
    # Add each keypoint component
    for i, point_name in enumerate(point_names):
        point = keypoints[i]
        row_data[f'{point_name}_x'] = point[0]
        row_data[f'{point_name}_y'] = point[1]
        row_data[f'{point_name}_z'] = 0  # YOLOv8 doesn't provide z-coordinates
        row_data[f'{point_name}_visibility'] = point[2]  # confidence score
    
    # Convert to DataFrame and save
    df = pd.DataFrame([row_data])
    if os.path.exists(output_path):
        df.to_csv(output_path, mode='a', header=False, index=False)
    else:
        df.to_csv(output_path, mode='w', header=True, index=False)

# Parse arguments
parser = argparse.ArgumentParser(description='Extract keypoints from a single image using YOLOv8')
parser.add_argument('class_name', type=str, help='Name of the class directory (e.g., proper, improper)')
parser.add_argument('image_name', type=str, help='Name of the image file (e.g., 177.jpg)')
args = parser.parse_args()

# Load YOLOv8 model
'''
yolov8n-pose.pt (fastest, least accurate)
yolov8s-pose.pt
yolov8m-pose.pt
yolov8l-pose.pt
yolov8x-pose.pt (slowest, most accurate)
'''
model = YOLO('yolov8s-pose.pt')

# Test image path
test_image_path = os.path.join(raw_dir, args.class_name, args.image_name)
output_csv = os.path.join(vectors_dir, "filtered_keypoints_vectors_yolo.csv")

# Verify that the image exists
if not os.path.exists(test_image_path):
    print(f"Error: Image not found at {test_image_path}")
    exit(1)

# Read image
image = cv2.imread(test_image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Time the inference
start_time = time.time()
results = model(image_rgb)
end_time = time.time()
inference_time = (end_time - start_time) * 1000  # Convert to milliseconds
print(f"Inference time: {inference_time:.2f} ms")

# Process results
for result in results:
    if result.keypoints is not None:
        # Time the post-processing
        post_start_time = time.time()
        
        # Get keypoints
        all_keypoints = result.keypoints.data[0].cpu().numpy()
        
        # Extract only the keypoints we need
        # YOLO keypoint indices: 0=nose, 5=left_shoulder, 6=right_shoulder, 11=left_hip, 
        # 12=right_hip, 13=left_knee, 14=right_knee, 15=left_ankle, 16=right_ankle
        needed_indices = [0, 5, 6, 11, 12, 13, 14, 15, 16]
        filtered_keypoints = all_keypoints[needed_indices]
        
        # Calculate shoulder midpoint
        left_shoulder = filtered_keypoints[1]  # Index 1 in filtered_keypoints is left_shoulder
        right_shoulder = filtered_keypoints[2]  # Index 2 in filtered_keypoints is right_shoulder
        shoulder_midpoint = np.array([
            (left_shoulder[0] + right_shoulder[0]) / 2,
            (left_shoulder[1] + right_shoulder[1]) / 2,
            min(left_shoulder[2], right_shoulder[2])  # Use minimum confidence
        ])
        
        # Add shoulder midpoint to keypoints
        keypoints = np.vstack([filtered_keypoints, shoulder_midpoint])
        
        # Set low confidence points to zero (similar to MediaPipe visibility threshold)
        confidence_threshold = 0.65
        keypoints[keypoints[:, 2] < confidence_threshold] = 0
        
        # Save to CSV
        save_keypoints_to_csv(keypoints, args.class_name, os.path.splitext(args.image_name)[0], output_csv)
        
        # Create visualization images
        image_with_keypoints = image.copy()
        black_background = np.zeros_like(image)
        
        # Define custom connections (same as MediaPipe implementation)
        custom_connections = [
            (0, 9),   # Nose to shoulder midpoint
            (1, 2),   # Left shoulder to right shoulder
            (1, 3),   # Left shoulder to left hip
            (2, 4),   # Right shoulder to right hip
            (3, 4),   # Left hip to right hip
            (3, 5),   # Left hip to left knee
            (4, 6),   # Right hip to right knee
            (5, 7),   # Left knee to left ankle
            (6, 8),   # Right knee to right ankle
        ]
        
        # Draw connections
        for connection in custom_connections:
            start_idx, end_idx = connection
            if all(keypoints[start_idx]) and all(keypoints[end_idx]):
                start_point = tuple(map(int, keypoints[start_idx][:2]))
                end_point = tuple(map(int, keypoints[end_idx][:2]))
                cv2.line(image_with_keypoints, start_point, end_point, (245, 66, 230), 2)
                cv2.line(black_background, start_point, end_point, (255, 255, 255), 2)
        
        # Draw keypoints
        for point in keypoints:
            if any(point):
                cv2.circle(image_with_keypoints, (int(point[0]), int(point[1])), 4, (245, 117, 66), -1)
                cv2.circle(black_background, (int(point[0]), int(point[1])), 4, (255, 255, 255), -1)
        
        # Show results
        cv2.imshow("YOLOv8 Keypoints", image_with_keypoints)
        cv2.imshow("YOLOv8 Keypoints (Black Background)", black_background)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    else:
        print("No pose detected in the image")

print(f"Processing complete. Vectors saved to {output_csv}") 