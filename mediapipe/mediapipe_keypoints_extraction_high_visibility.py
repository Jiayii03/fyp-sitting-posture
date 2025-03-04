# pip install mediapipe pandas numpy opencv-python
"""
This script is used to test only on a single image.
It removes faces and elbows and only extracts the ones that are needed (nose, shoulders, hips, knees, ankles, and shoulder midpoint).
It also removes keypoints with low visibility (visibility < 0.65).
Save the keypoints to a CSV file.

run python keypoints_extraction_low_visibility.py
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import pandas as pd
import argparse
import time

# Directory paths
raw_dir = "C:/Users/User/Documents/UNM_CSAI/UNM_current_modules/COMP3025_Individual_Dissertation/dev/datasets/raw"
keypoints_dir = "C:/Users/User/Documents/UNM_CSAI/UNM_current_modules/COMP3025_Individual_Dissertation/dev/datasets/keypoints"
keypoints_only_dir = "C:/Users/User/Documents/UNM_CSAI/UNM_current_modules/COMP3025_Individual_Dissertation/dev/datasets/keypoints_only"
vectors_dir = "C:/Users/User/Documents/UNM_CSAI/UNM_current_modules/COMP3025_Individual_Dissertation/dev/datasets/vectors"

# Function to save keypoints to CSV
def save_keypoints_to_csv(keypoints, class_name, image_id, output_path):
    # Create a row with class, image_id, and all keypoints
    row_data = {
        'class': class_name,
        'image_id': image_id,
    }
    
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
    
    # Add each keypoint component (x, y) to the row
    for i, point in enumerate(keypoints):
        point_name = point_names[i]
        row_data[f'{point_name}_x'] = point[0]  # x is already normalized
        row_data[f'{point_name}_y'] = point[1]  # y is already normalized
    
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

# Add argument parser
parser = argparse.ArgumentParser(description='Extract keypoints from a single image')
parser.add_argument('class_name', type=str, help='Name of the class directory (e.g., proper, improper)')
parser.add_argument('image_name', type=str, help='Name of the image file (e.g., 177.jpg)')
args = parser.parse_args()

# Test image path - modified to use command line arguments
test_image_path = os.path.join(raw_dir, args.class_name, args.image_name)
output_csv = os.path.join(vectors_dir, "xy_filtered_keypoints_vectors_mediapipe.csv")

# Verify that the image exists
if not os.path.exists(test_image_path):
    print(f"Error: Image not found at {test_image_path}")
    exit(1)

# Read and process image
image = cv2.imread(test_image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Time the inference
start_time = time.time()
results = pose.process(image_rgb)
end_time = time.time()
inference_time = (end_time - start_time) * 1000  # Convert to milliseconds
print(f"Inference time: {inference_time:.2f} ms")

if results.pose_landmarks:
    # Time the post-processing (keypoint extraction and visualization)
    post_start_time = time.time()
    
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
    
    # Add needed landmarks
    for landmark_enum in needed_landmarks:
        landmark = landmarks[landmark_enum]
        if landmark.visibility < 0.65:
            keypoints.append([0, 0])  # Set to zero if visibility is low
        else:
            # MediaPipe already provides normalized coordinates (0-1)
            keypoints.append([landmark.x, landmark.y])
    
    # Add shoulder midpoint
    if shoulder_midpoint.visibility < 0.65:
        keypoints.append([0, 0])
    else:
        keypoints.append([shoulder_midpoint.x, shoulder_midpoint.y])
    
    keypoints = np.array(keypoints)
    
    # Save to CSV
    save_keypoints_to_csv(keypoints, args.class_name, os.path.splitext(args.image_name)[0], output_csv)
    
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
    
    # Visualize the keypoints on the image
    image_with_keypoints = image.copy()
    h, w, _ = image.shape  # Get image dimensions
    
    # Draw custom connections
    for connection in custom_connections:
        start_idx, end_idx = connection
        if all(keypoints[start_idx]) and all(keypoints[end_idx]):  # Check if both points are not zero
            start_point = (int(keypoints[start_idx][0] * w), int(keypoints[start_idx][1] * h))
            end_point = (int(keypoints[end_idx][0] * w), int(keypoints[end_idx][1] * h))
            cv2.line(image_with_keypoints, start_point, end_point, (245, 66, 230), 2)
    
    # Draw keypoints
    for point in keypoints:
        if any(point):  # Check if point is not zero
            cv2.circle(image_with_keypoints, 
                      (int(point[0] * w), int(point[1] * h)), 
                      4, (245, 117, 66), -1)
    
    # Show image with keypoints
    cv2.imshow("Keypoints", image_with_keypoints)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Save keypoints with black background
    image_with_keypoints_black_background = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    
    # Draw custom connections on black background
    for connection in custom_connections:
        start_idx, end_idx = connection
        if all(keypoints[start_idx]) and all(keypoints[end_idx]):  # Check if both points are not zero
            start_point = (int(keypoints[start_idx][0] * w), int(keypoints[start_idx][1] * h))
            end_point = (int(keypoints[end_idx][0] * w), int(keypoints[end_idx][1] * h))
            cv2.line(image_with_keypoints_black_background, start_point, end_point, (255, 255, 255), 2)
    
    # Draw keypoints on black background
    for point in keypoints:
        if any(point):  # Check if point is not zero
            cv2.circle(image_with_keypoints_black_background, 
                      (int(point[0] * w), int(point[1] * h)), 
                      4, (255, 255, 255), -1)
    
    # show image with keypoints on black background
    cv2.imshow("Keypoints on Black Background", image_with_keypoints_black_background)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
else:
    print("No pose detected in image")

# Clean up
pose.close()
print(f"Processing complete. Vectors saved to {output_csv}")
