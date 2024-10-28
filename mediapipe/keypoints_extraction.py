# pip install mediapipe pandas numpy opencv-python
"""
This script extracts keypoints from images using MediaPipe Pose.
It loops through all classes in the raw dataset directory (datasets/raw) and processes each image in each class.
It saves the keypoints into a CSV file (datasets/vectors/pose_vectors.csv) and visualizes them on the images (datasets/keypoints and datasets/keypoints_only).

In this script, all keypoints are extracted and saved.
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import pandas as pd

# Directory paths
raw_dir = "C:/Users/User/Documents/UNM_CSAI/UNM_current_modules/COMP3025_Individual_Dissertation/dev/datasets/raw"
keypoints_dir = "C:/Users/User/Documents/UNM_CSAI/UNM_current_modules/COMP3025_Individual_Dissertation/dev/datasets/keypoints"
keypoints_only_dir = "C:/Users/User/Documents/UNM_CSAI/UNM_current_modules/COMP3025_Individual_Dissertation/dev/datasets/keypoints_only"
vectors_dir = "C:/Users/User/Documents/UNM_CSAI/UNM_current_modules/COMP3025_Individual_Dissertation/dev/datasets/vectors"

# Create directories if they don't exist
for directory in [keypoints_dir, keypoints_only_dir, vectors_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_keypoints_to_csv(keypoints, class_name, image_id, vectors_dir):
    csv_path = os.path.join(vectors_dir, "pose_vectors.csv")
    
    # Flatten the keypoints array
    flattened = keypoints.flatten()
    
    # Create column names (x0, y0, z0, v0, x1, y1, z1, v1, etc.)
    columns = []
    for i in range(len(keypoints)):
        columns.extend([f'x{i}', f'y{i}', f'z{i}', f'v{i}'])
    
    # Create DataFrame
    df = pd.DataFrame([flattened], columns=columns)
    df['class'] = class_name
    df['image_id'] = image_id
    
    # Append to CSV if it exists, create new if it doesn't
    df.to_csv(csv_path, mode='a', header=not os.path.exists(csv_path), index=False)
    print(f"Keypoints vector appended to {csv_path}")

# Get all class directories
class_dirs = [d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))]

# Initialize the pose model (moved outside the loop for efficiency)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# Loop through each class directory
for class_name in class_dirs:

    # if class name is "junk", or "crowded", skip
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
    images = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    for image_file in images:
        image_id = os.path.splitext(image_file)[0]  # Remove file extension
        print(f"Processing image: {image_id}")
        
        raw_image_path = os.path.join(raw_dir, class_name, image_file)
        save_keypoints_path = os.path.join(keypoints_dir, class_name, image_file)
        save_keypoints_only_path = os.path.join(keypoints_only_dir, class_name, image_file)
        
        # Read and process image
        image = cv2.imread(raw_image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        
        # Check if pose landmarks are detected
        if results.pose_landmarks:
            # Extract keypoints
            keypoints = []
            for landmark in results.pose_landmarks.landmark:
                x = landmark.x * image.shape[1]
                y = landmark.y * image.shape[0]
                z = landmark.z
                visibility = landmark.visibility
                keypoints.append([x, y, z, visibility])
            
            keypoints = np.array(keypoints)
                        
            # Save keypoints vector to CSV
            save_keypoints_to_csv(keypoints, class_name, image_id, vectors_dir)
            print(f"Saving keypoints vector to {vectors_dir}")
            
            # Visualize the keypoints on the image
            mp_drawing = mp.solutions.drawing_utils
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )

            # Save the image with keypoints
            image_with_keypoints = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_keypoints_path, image_with_keypoints)
            print(f"Image of keypoints saved to {save_keypoints_path}")

            # Save keypoints with black background
            image_with_keypoints_black_background = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
            mp_drawing.draw_landmarks(
                image_with_keypoints_black_background, 
                results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),  # White color for better visibility
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2)
            )
            cv2.imwrite(save_keypoints_only_path, image_with_keypoints_black_background)
            print(f"Image of keypoints only saved to {save_keypoints_only_path}")
            
        else:
            print(f"No pose detected in image {image_file}")

# Clean up
pose.close()
