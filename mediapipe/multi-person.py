"""
This script performs multi-person pose estimation on an image using YOLOv5 and Mediapipe.

The script loads an image, runs YOLOv5 to detect persons, crops the detected persons, and runs Mediapipe pose estimation on the cropped regions.
The pose landmarks are then overlaid on the original image to visualize the pose estimation results.

Run the script using the following command:
python multi-person.py

"""

import cv2
import torch
import numpy as np
import mediapipe as mp
from torchvision import transforms

# Set the path to the YOLOv5 model
model_path = "../models/yolo-human-detection/yolov5s.pt"

# Load YOLOv5 model from the specified local path
yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=False)
yolo_model.classes = [0]  # Only detect persons (COCO class ID: 0)

# Initialize Mediapipe pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# Load an image
image_path = "crowded_3.jpg"
image = cv2.imread(image_path)
height, width = image.shape[:2]

# Run YOLOv5 person detection
results = yolo_model(image)

# Parse results
person_detections = results.pandas().xyxy[0]  # Extract bounding box data
for _, detection in person_detections.iterrows():
    x_min, y_min, x_max, y_max, confidence, class_id, name = detection

    # Ensure the detected object is a person
    if name == "person" and confidence > 0.5:
        # Draw the bounding box on the original image
        x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(
            image,
            f"Person: {confidence:.2f}",
            (x_min, y_min - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

        # Crop the person from the image
        cropped_person = image[y_min:y_max, x_min:x_max]

        # Run Mediapipe pose estimation on the cropped region
        cropped_rgb = cv2.cvtColor(cropped_person, cv2.COLOR_BGR2RGB)
        results = pose.process(cropped_rgb)

        if results.pose_landmarks:
            # Draw keypoints and connections on the cropped person image
            annotated_image = cropped_person.copy()
            mp.solutions.drawing_utils.draw_landmarks(
                annotated_image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

            # Overlay the pose landmarks on the original image
            image[y_min:y_max, x_min:x_max] = annotated_image

# Show the result
cv2.imshow("Multi-Person Pose Estimation with Bounding Boxes", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
