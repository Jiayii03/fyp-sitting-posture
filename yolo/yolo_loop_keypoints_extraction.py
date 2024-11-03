# pip install ultralytics opencv-python numpy pandas
"""
This script loops through all images in all class directories and extracts the keypoints using YOLOv8.
It extracts only specific keypoints (nose, shoulders, hips, knees, ankles, and shoulder midpoint).
It also removes keypoints with low confidence (confidence < 0.65).
Save the keypoints to a CSV file.

run python yolo_loop_keypoints_extraction.py
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os
import pandas as pd
import time

# Directory paths - modified for YOLO-specific outputs
raw_dir = "C:/Users/User/Documents/UNM_CSAI/UNM_current_modules/COMP3025_Individual_Dissertation/dev/datasets/raw"
keypoints_dir = "C:/Users/User/Documents/UNM_CSAI/UNM_current_modules/COMP3025_Individual_Dissertation/dev/datasets/keypoints/yolo"
keypoints_only_dir = "C:/Users/User/Documents/UNM_CSAI/UNM_current_modules/COMP3025_Individual_Dissertation/dev/datasets/keypoints_only/yolo"
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

# Load YOLOv8 model
'''
yolov8n-pose.pt (fastest, least accurate)
yolov8s-pose.pt
yolov8m-pose.pt
yolov8l-pose.pt
yolov8x-pose.pt (slowest, most accurate)
'''
model = YOLO('yolov8s-pose.pt')

# Get all class directories
class_dirs = [d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))]

# Output CSV path
output_csv = os.path.join(vectors_dir, "filtered_keypoints_vectors_yolo.csv")

# Process each class directory
for class_name in class_dirs:
    # Skip junk and crowded classes
    if class_name in ["junk", "crowded"]:
        continue

    print(f"Processing class: {class_name}")
    
    # Create class directories if they don't exist
    for directory in [keypoints_dir, keypoints_only_dir]:
        class_dir = os.path.join(directory, class_name)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)
    
    # Get all images in the class directory
    class_path = os.path.join(raw_dir, class_name)
    images = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.jpeg', '.png', '.webp'))]
    
    for image_file in images:
        try:
            print(f"Processing image: {image_file}")
            image_id = os.path.splitext(image_file)[0]
            
            # Read and process image
            image_path = os.path.join(class_path, image_file)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to read image: {image_file}")
                continue
                
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = model(image_rgb)
            
            # Check if any poses were detected
            if not results or len(results) == 0:
                print(f"No results for image: {image_file}")
                continue
                
            result = results[0]  # Get first result
            if not hasattr(result, 'keypoints') or result.keypoints is None or len(result.keypoints.data) == 0:
                print(f"No pose detected in image: {image_file}")
                continue
            
            # Get keypoints
            all_keypoints = result.keypoints.data[0].cpu().numpy()
            
            # Extract only the keypoints we need
            needed_indices = [0, 5, 6, 11, 12, 13, 14, 15, 16]  # YOLO keypoint indices
            filtered_keypoints = all_keypoints[needed_indices]
            
            # Calculate shoulder midpoint
            left_shoulder = filtered_keypoints[1]
            right_shoulder = filtered_keypoints[2]
            shoulder_midpoint = np.array([
                (left_shoulder[0] + right_shoulder[0]) / 2,
                (left_shoulder[1] + right_shoulder[1]) / 2,
                min(left_shoulder[2], right_shoulder[2])
            ])
            
            # Add shoulder midpoint to keypoints
            keypoints = np.vstack([filtered_keypoints, shoulder_midpoint])
            
            # Set low confidence points to zero
            confidence_threshold = 0.65
            keypoints[keypoints[:, 2] < confidence_threshold] = 0
            
            # Save to CSV
            save_keypoints_to_csv(keypoints, class_name, image_id, output_csv)
            
            # Create visualization images
            image_with_keypoints = image.copy()
            black_background = np.zeros_like(image)
            
            # Define custom connections
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
                
            # Save the images
            save_path = os.path.join(keypoints_dir, class_name, f"{image_id}.jpg")
            save_path_black = os.path.join(keypoints_only_dir, class_name, f"{image_id}.jpg")
            
            cv2.imwrite(save_path, image_with_keypoints)
            cv2.imwrite(save_path_black, black_background)
            
            print(f"Saved keypoint images for {image_file}")
            
        except Exception as e:
            print(f"Error processing {image_file}: {str(e)}")
            continue

print(f"Processing complete. Vectors saved to {output_csv}") 