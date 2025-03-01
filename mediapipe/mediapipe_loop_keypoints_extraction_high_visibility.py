# pip install mediapipe pandas numpy opencv-python
"""
This script loops through all images in all class directories and extracts the keypoints.
It removes faces and elbows and only extracts the ones that are needed (nose, shoulders, hips, knees, ankles, and shoulder midpoint).
It also removes keypoints with low visibility (visibility < 0.65).
Save the keypoints to a CSV file.

Before running this script, make sure to delete the existing keypoints CSV file, and all directories in datasets/keypoints and datasets/keypoints_only.

if you want to use augmented data, run 
python mediapipe_loop_keypoints_extraction_high_visibility.py --augmented

else,
python mediapipe_loop_keypoints_extraction_high_visibility.py
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import pandas as pd

# set up a flag to use augmented data using argparse
import argparse
parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument('--augmented', action='store_true', help='use augmented data')
args = parser.parse_args()

# Directory paths
raw_dir = "C:/Users/User/Documents/UNM_CSAI/UNM_current_modules/COMP3025_Individual_Dissertation/dev/datasets/raw"
augmented_dir = "C:/Users/User/Documents/UNM_CSAI/UNM_current_modules/COMP3025_Individual_Dissertation/dev/datasets/augmented"
keypoints_dir = "C:/Users/User/Documents/UNM_CSAI/UNM_current_modules/COMP3025_Individual_Dissertation/dev/datasets/keypoints/mediapipe"
keypoints_only_dir = "C:/Users/User/Documents/UNM_CSAI/UNM_current_modules/COMP3025_Individual_Dissertation/dev/datasets/keypoints_only/mediapipe"
vectors_dir = "C:/Users/User/Documents/UNM_CSAI/UNM_current_modules/COMP3025_Individual_Dissertation/dev/datasets/vectors"
augmented_keypoints_dir = "C:/Users/User/Documents/UNM_CSAI/UNM_current_modules/COMP3025_Individual_Dissertation/dev/datasets/augmented_keypoints"
augmented_keypoints_only_dir  = "C:/Users/User/Documents/UNM_CSAI/UNM_current_modules/COMP3025_Individual_Dissertation/dev/datasets/augmented_keypoints_only"
output_csv = os.path.join(vectors_dir, "augmented_xy_filtered_keypoints_vectors_mediapipe.csv")

if args.augmented:
    raw_dir = augmented_dir
    keypoints_dir = augmented_keypoints_dir
    keypoints_only_dir = augmented_keypoints_only_dir
else:
    raw_dir = raw_dir   

# Function to save keypoints to CSV
def save_keypoints_to_csv(keypoints, class_name, output_path):
    # List of point names in order
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

    # Create a row with all keypoints
    row_data = {}
    for i, point in enumerate(keypoints):
        point_name = point_names[i]
        row_data[f'{point_name}_x'] = point[0]  # x is already normalized
        row_data[f'{point_name}_y'] = point[1]  # y is already normalized

    # Add the class label at the end
    row_data['class'] = class_name

    # Convert to DataFrame
    df = pd.DataFrame([row_data])

    # Write to CSV
    if os.path.exists(output_path):
        df.to_csv(output_path, mode='a', header=False, index=False)
    else:
        df.to_csv(output_path, mode='w', header=True, index=False)

# Initialize the pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# Define the landmarks we want to keep
needed_landmarks = [
    mp_pose.PoseLandmark.NOSE,
    mp_pose.PoseLandmark.LEFT_SHOULDER,
    mp_pose.PoseLandmark.RIGHT_SHOULDER,
    mp_pose.PoseLandmark.LEFT_HIP,
    mp_pose.PoseLandmark.RIGHT_HIP,
    mp_pose.PoseLandmark.LEFT_KNEE,
    mp_pose.PoseLandmark.RIGHT_KNEE,
    mp_pose.PoseLandmark.LEFT_ANKLE,
    mp_pose.PoseLandmark.RIGHT_ANKLE,
]

# Get all class directories
class_dirs = [d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))]

# Process each class directory
for class_name in class_dirs:
    # Skip junk and crowded classes
    if class_name in ["junk", "crowded"]:
        continue

    print(f"Processing class: {class_name}")
    
    # Create class directories if they don't exist
    for directory in [keypoints_dir, keypoints_only_dir]:
        class_dir = os.path.join(directory, class_name)
        
        # if class directory already exists, delete it
        if os.path.exists(class_dir):
            for file in os.listdir(class_dir):
                os.remove(os.path.join(class_dir, file))
            os.rmdir(class_dir)
        
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)
    
    # Get all images in the class directory
    class_path = os.path.join(raw_dir, class_name)
    images = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.jpeg', '.png', '.webp'))]
    
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
                'visibility': min(left_shoulder.visibility, right_shoulder.visibility)
            })
            
            # Add needed landmarks with visibility check
            for landmark_enum in needed_landmarks:
                landmark = landmarks[landmark_enum]
                if landmark.visibility < 0.65:  # Higher threshold for visibility
                    keypoints.append([0, 0])  # Only x, y coordinates
                else:
                    # Use normalized coordinates directly
                    keypoints.append([landmark.x, landmark.y])
            
            # Add shoulder midpoint with visibility check
            if shoulder_midpoint.visibility < 0.65:
                keypoints.append([0, 0])
            else:
                keypoints.append([shoulder_midpoint.x, shoulder_midpoint.y])
            
            keypoints = np.array(keypoints)
            
            # Save to CSV
            save_keypoints_to_csv(keypoints, class_name, output_csv)

            # Create custom connections including the new shoulder midpoint
            custom_connections = [
                (0, 9),   # Nose (0) to shoulder midpoint (9, last point)
                (1, 2),   # Left shoulder to right shoulder
                (1, 3),   # Left shoulder to left hip
                (2, 4),   # Right shoulder to right hip
                (3, 4),   # Left hip to right hip
                (3, 5),   # Left hip to left knee
                (4, 6),   # Right hip to right knee
                (5, 7),   # Left knee to left ankle
                (6, 8),   # Right knee to right ankle
            ]
            
            # Create visualization images
            h, w, _ = image.shape
            image_with_keypoints = image.copy()
            image_with_keypoints_black_background = np.zeros((h, w, 3), dtype=np.uint8)
            
            # Draw connections
            for connection in custom_connections:
                start_idx, end_idx = connection
                if all(keypoints[start_idx]) and all(keypoints[end_idx]):  # Check if both points are visible
                    # Convert normalized coordinates to pixel coordinates
                    start_point = (
                        int(keypoints[start_idx][0] * w), 
                        int(keypoints[start_idx][1] * h)
                    )
                    end_point = (
                        int(keypoints[end_idx][0] * w), 
                        int(keypoints[end_idx][1] * h)
                    )
                    
                    # Draw lines on both images
                    cv2.line(image_with_keypoints, start_point, end_point, (245, 66, 230), 2)
                    cv2.line(image_with_keypoints_black_background, start_point, end_point, (255, 255, 255), 2)
            
            # Draw keypoints
            for point in keypoints:
                if any(point):  # Check if point is not zero
                    # Convert normalized coordinates to pixel coordinates
                    point_px = (
                        int(point[0] * w),
                        int(point[1] * h)
                    )
                    cv2.circle(image_with_keypoints, point_px, 4, (245, 117, 66), -1)
                    cv2.circle(image_with_keypoints_black_background, point_px, 4, (255, 255, 255), -1)
            
            # Save the images
            save_path = os.path.join(keypoints_dir, class_name, f"{image_id}.jpg")
            save_path_black = os.path.join(keypoints_only_dir, class_name, f"{image_id}.jpg")
            
            cv2.imwrite(save_path, image_with_keypoints)
            cv2.imwrite(save_path_black, image_with_keypoints_black_background)
            
            print(f"Saved keypoint images for {image_file}")
            
        else:
            print(f"No pose detected in image {image_file}")

# Clean up
pose.close()
print(f"Processing complete. Vectors saved to {output_csv}")
