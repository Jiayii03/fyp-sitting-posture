# pip install mediapipe pandas numpy opencv-python
"""
This script extracts keypoints from images using MediaPipe Pose.
It filters out the "junk" and "crowded" classes and saves the keypoints into a CSV file (datasets/vectors/filtered_keypoints_vectors.csv).

It ignores non-essential keypoints and only extracts the ones that are needed (nose, shoulders, elbows, hips, knees, ankles, and shoulder midpoint).
It does not filter out low confidence keypoints.

run python keypoints_extraction_filter.py
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

# Function to save keypoints to CSV
def save_keypoints_to_csv(keypoints, class_name, image_id, output_path):
    # Flatten the keypoints array and create a single row
    keypoints_flat = keypoints.flatten()
    
    # Create a row with class, image_id, and all keypoints
    row_data = {
        'class': class_name,
        'image_id': image_id,
    }
    
    # Add each keypoint component (x, y, z, visibility) to the row
    for i, value in enumerate(keypoints_flat):
        point_idx = i // 4  # Which point (0-11)
        component_idx = i % 4  # Which component (0=x, 1=y, 2=z, 3=visibility)
        component_name = ['x', 'y', 'z', 'visibility'][component_idx]
        
        # Name points 0-10 based on their landmark names, and 11 as midpoint
        if point_idx < 11:
            point_name = needed_landmarks[point_idx].name.lower()
        else:
            point_name = 'shoulder_midpoint'
            
        column_name = f'{point_name}_{component_name}'
        row_data[column_name] = value
    
    # Convert to DataFrame
    df = pd.DataFrame([row_data])
    
    # Append to CSV if it exists, create new if it doesn't
    if os.path.exists(output_path):
        df.to_csv(output_path, mode='a', header=False, index=False)
    else:
        df.to_csv(output_path, mode='w', header=True, index=False)

# Initialize the pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# Define the landmarks we want to keep (moved outside the loop)
needed_landmarks = [
    mp_pose.PoseLandmark.NOSE,
    mp_pose.PoseLandmark.LEFT_SHOULDER,
    mp_pose.PoseLandmark.RIGHT_SHOULDER,
    mp_pose.PoseLandmark.LEFT_ELBOW,
    mp_pose.PoseLandmark.RIGHT_ELBOW,
    mp_pose.PoseLandmark.LEFT_HIP,
    mp_pose.PoseLandmark.RIGHT_HIP,
    mp_pose.PoseLandmark.LEFT_KNEE,
    mp_pose.PoseLandmark.RIGHT_KNEE,
    mp_pose.PoseLandmark.LEFT_ANKLE,
    mp_pose.PoseLandmark.RIGHT_ANKLE,
]

# Get all class directories
class_dirs = [d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))]

# Output CSV path
output_csv = os.path.join(vectors_dir, "filtered_keypoints_vectors.csv")

# Process each class directory
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
        print(f"Processing image: {image_file}")
        image_id = os.path.splitext(image_file)[0]
        
        # Read and process image
        image_path = os.path.join(class_path, image_file)
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        
        if results.pose_landmarks:
            # Extract keypoints
            keypoints = []
            landmarks = results.pose_landmarks.landmark
            
            # Calculate shoulder midpoint
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            shoulder_midpoint = type('Landmark', (), {
                'x': (left_shoulder.x + right_shoulder.x) / 2,
                'y': (left_shoulder.y + right_shoulder.y) / 2,
                'z': (left_shoulder.z + right_shoulder.z) / 2,
                'visibility': min(left_shoulder.visibility, right_shoulder.visibility)
            })
            
            # Add needed landmarks
            for landmark_enum in needed_landmarks:
                landmark = landmarks[landmark_enum]
                x = landmark.x * image.shape[1]
                y = landmark.y * image.shape[0]
                z = landmark.z
                visibility = landmark.visibility
                keypoints.append([x, y, z, visibility])
            
            # Add shoulder midpoint
            keypoints.append([
                shoulder_midpoint.x * image.shape[1],
                shoulder_midpoint.y * image.shape[0],
                shoulder_midpoint.z,
                shoulder_midpoint.visibility
            ])
            
            keypoints = np.array(keypoints)
            
            # Save to CSV
            save_keypoints_to_csv(keypoints, class_name, image_id, output_csv)
            
            # Create custom connections including the new shoulder midpoint
            custom_connections = [
                (0, 11),  # Nose (0) to shoulder midpoint (11, last point)
                (1, 2),   # Left shoulder to right shoulder
                (1, 3),   # Left shoulder to left elbow
                (2, 4),   # Right shoulder to right elbow
                (1, 5),   # Left shoulder to left hip
                (2, 6),   # Right shoulder to right hip
                (5, 6),   # Left hip to right hip
                (5, 7),   # Left hip to left knee
                (6, 8),   # Right hip to right knee
                (7, 9),   # Left knee to left ankle
                (8, 10),  # Right knee to right ankle
            ]
            
            # Visualize the keypoints on the image
            image_with_keypoints = image.copy()
            
            # Draw custom connections
            for connection in custom_connections:
                start_idx, end_idx = connection
                start_point = (int(keypoints[start_idx][0]), int(keypoints[start_idx][1]))
                end_point = (int(keypoints[end_idx][0]), int(keypoints[end_idx][1]))
                cv2.line(image_with_keypoints, start_point, end_point, (245, 66, 230), 2)
            
            # Draw keypoints
            for point in keypoints:
                cv2.circle(image_with_keypoints, (int(point[0]), int(point[1])), 4, (245, 117, 66), -1)
            
            # Save the image with keypoints
            save_path = os.path.join(keypoints_dir, class_name, f"{image_id}.jpg")
            cv2.imwrite(save_path, image_with_keypoints)
            print(f"Image with keypoints saved to {save_path}")
            
            # Save keypoints with black background
            image_with_keypoints_black_background = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
            
            # Draw custom connections on black background
            for connection in custom_connections:
                start_idx, end_idx = connection
                start_point = (int(keypoints[start_idx][0]), int(keypoints[start_idx][1]))
                end_point = (int(keypoints[end_idx][0]), int(keypoints[end_idx][1]))
                cv2.line(image_with_keypoints_black_background, start_point, end_point, (255, 255, 255), 2)
            
            # Draw keypoints on black background
            for point in keypoints:
                cv2.circle(image_with_keypoints_black_background, (int(point[0]), int(point[1])), 4, (255, 255, 255), -1)
            
            save_path_black = os.path.join(keypoints_only_dir, class_name, f"{image_id}.jpg")
            cv2.imwrite(save_path_black, image_with_keypoints_black_background)
            print(f"Image with keypoints only saved to {save_path_black}")
            
        else:
            print(f"No pose detected in image {image_file}")

# Clean up
pose.close()
print(f"Processing complete. Vectors saved to {output_csv}")
