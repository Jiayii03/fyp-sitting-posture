"""
MediaPipe Full Keypoints Extraction

This script loops through all images in all class directories and extracts all default MediaPipe pose keypoints.
It saves the x and y coordinates of all keypoints to a CSV file in the same directory as the script.
No visualization images are saved.

Usage:
python extract_all_mediapipe_keypoints.py --input_dir "path/to/image/directory"
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import pandas as pd
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description="Extract all MediaPipe pose keypoints from images.")
parser.add_argument('--input_dir', type=str, required=True, help='Directory containing class folders with images')
args = parser.parse_args()

# Output CSV file path (in the same directory as the script)
output_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "full_mediapipe_keypoints.csv")

# Initialize the pose model with default settings
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# Get all 33 MediaPipe pose landmark names
landmark_names = [name for name in dir(mp_pose.PoseLandmark) 
                 if not name.startswith('_') and name.isupper()]
landmark_names.sort(key=lambda name: getattr(mp_pose.PoseLandmark, name).value)

# Function to save keypoints to CSV
def save_keypoints_to_csv(keypoints, class_name, image_id, output_path):
    # Create a row with all keypoints
    row_data = {'image_id': image_id}
    
    for i, name in enumerate(landmark_names):
        row_data[f'{name.lower()}_x'] = keypoints[i][0]  # x coordinate
        row_data[f'{name.lower()}_y'] = keypoints[i][1]  # y coordinate
    
    # Add the class label
    row_data['class'] = class_name
    
    # Convert to DataFrame
    df = pd.DataFrame([row_data])
    
    # Write to CSV
    if os.path.exists(output_path):
        df.to_csv(output_path, mode='a', header=False, index=False)
    else:
        # Create the first row with headers
        df.to_csv(output_path, mode='w', header=True, index=False)

# Get all class directories from the input directory
input_dir = args.input_dir
class_dirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]

# Track progress
total_processed = 0
successful_extractions = 0

print(f"Starting MediaPipe keypoint extraction...")
print(f"Output will be saved to: {output_csv}")

# Process each class directory
for class_name in class_dirs:
    # Skip junk and crowded classes if needed
    if class_name in ["junk", "crowded"]:
        continue
    
    print(f"Processing class: {class_name}")
    
    # Get all images in the class directory
    class_path = os.path.join(input_dir, class_name)
    images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
    
    for image_file in images:
        total_processed += 1
        image_id = os.path.splitext(image_file)[0]
        
        # Read and process image
        image_path = os.path.join(class_path, image_file)
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Could not read image: {image_path}")
                continue
                
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            
            if results.pose_landmarks:
                # Extract all 33 keypoints
                keypoints = []
                landmarks = results.pose_landmarks.landmark
                
                # Get all 33 landmarks in order
                for landmark_enum in [getattr(mp_pose.PoseLandmark, name) for name in landmark_names]:
                    landmark = landmarks[landmark_enum]
                    # Store normalized coordinates (x, y)
                    keypoints.append([landmark.x, landmark.y])
                
                # Save to CSV
                save_keypoints_to_csv(keypoints, class_name, image_id, output_csv)
                successful_extractions += 1
                
                if total_processed % 10 == 0:
                    print(f"Processed {total_processed} images, {successful_extractions} successful extractions")
            else:
                print(f"No pose detected in image: {image_file}")
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")

# Clean up
pose.close()
print(f"Processing complete!")
print(f"Total images processed: {total_processed}")
print(f"Successful pose extractions: {successful_extractions}")
print(f"Keypoints saved to: {output_csv}")