#!/usr/bin/env python3
"""
MediaPipe Keypoint Extractor - Raw Version

This script processes all images in a specified input directory and 
saves keypoint visualizations to the raw_keypoints directory.
"""

import os
import cv2
import mediapipe as mp
import glob
from tqdm import tqdm

def extract_keypoints():
    # Define directories - adjust these as needed
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = r"C:\Users\User\Documents\UNM_CSAI\UNM_current_modules\COMP3025_Individual_Dissertation\dev\datasets\raw\proper"  # Change this if your images are in a different directory
    output_dir = os.path.join(script_dir, "raw_keypoints")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    
    # Get all image files
    image_paths = glob.glob(os.path.join(input_dir, "*.jpg")) + \
                  glob.glob(os.path.join(input_dir, "*.png")) + \
                  glob.glob(os.path.join(input_dir, "*.jpeg"))
    
    print(f"Found {len(image_paths)} images to process")
    
    # Initialize MediaPipe Pose
    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        min_detection_confidence=0.5) as pose:
        
        # Process each image
        for img_path in tqdm(image_paths, desc="Processing images"):
            # Skip processing files in the output directory
            if "raw_keypoints" in img_path:
                continue
                
            # Read image
            image = cv2.imread(img_path)
            if image is None:
                print(f"Error: Could not read image {img_path}")
                continue
            
            # Get image filename
            filename = os.path.basename(img_path)
            
            # Convert to RGB for MediaPipe
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process image with MediaPipe
            results = pose.process(image_rgb)
            
            # Extract keypoints if detected
            if results.pose_landmarks:
                # Create a copy of the image for drawing
                annotated_image = image.copy()
                
                # Draw pose landmarks
                mp_drawing.draw_landmarks(
                    annotated_image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS)
                
                # Save annotated image to raw_keypoints directory
                output_path = os.path.join(output_dir, filename)
                cv2.imwrite(output_path, annotated_image)
                
                print(f"✓ Processed {filename}")
            else:
                print(f"✗ No pose detected in {filename}")

if __name__ == "__main__":
    extract_keypoints()