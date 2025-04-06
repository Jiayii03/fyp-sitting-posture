import mediapipe as mp
import torch
import torch.nn.functional as F
import numpy as np
import cv2

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

class PostureDetector:
    def __init__(self, model_manager):
        self.model_manager = model_manager
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.yolo_model = self._load_yolo_model()
        self.needed_landmarks = needed_landmarks
        self.custom_connections = custom_connections
        
    def _load_yolo_model(self):
        yolo_model_path = "../models/yolo-human-detection/yolov5n.pt"
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=yolo_model_path, force_reload=False)
        model.classes = [0]  # Only detect persons
        return model
        
    def extract_keypoints(self, image_input, pose_model, visibility_threshold=0.65):
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
        for landmark_enum in self.needed_landmarks:
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

        for start_idx, end_idx in self.custom_connections:
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
        
    def extract_keypoints_multi_person(self, image_input, pose_model, yolo_model, confidence_threshold=0.30, visibility_threshold=0.65):
        """
        Extract keypoints for multiple persons using YOLO for detection and Mediapipe for pose estimation.

        Args:
        - image_input: Path to the image file or a numpy array (frame).
        - pose_model: Mediapipe pose model instance.
        - yolo_model: YOLO model for person detection.
        - confidence_threshold: Minimum confidence for YOLO detection.
        - visibility_threshold: Minimum visibility to consider a keypoint valid.

        Returns:
        - A list of keypoints arrays, each corresponding to a detected person.
        - Processed image with bounding boxes and connections drawn.
        """
        if isinstance(image_input, str):  # Check if it's a file path
            image = cv2.imread(image_input)
        else:  # Assume it's already an image (numpy array)
            image = image_input

        if image is None:
            raise ValueError("Invalid image input. Ensure the file path or frame is correct.")

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]

        # YOLO detection
        results = yolo_model(image_rgb)
        person_detections = results.pandas().xyxy[0]

        keypoints_list = []
        bboxes = []
        person = 1
        for _, detection in person_detections.iterrows():
            x_min, y_min, x_max, y_max, confidence, class_id, name = detection

            # Ensure the detected object is a person
            if name == "person" and confidence > confidence_threshold:
                x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
                bboxes.append((x_min, y_min, x_max, y_max))  # Save bounding box for later use

                # Draw bounding box
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(
                    image,
                    f"Person {person}: {confidence:.2f}",
                    (x_min, y_min - 10),    
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

                # Crop the person from the image
                cropped_person = image[y_min:y_max, x_min:x_max]

                # Run Mediapipe pose estimation
                cropped_rgb = cv2.cvtColor(cropped_person, cv2.COLOR_BGR2RGB)
                results = pose_model.process(cropped_rgb)

                if results.pose_landmarks:
                    keypoints_cropped = []  # Keypoints in the cropped coordinate system
                    keypoints_scaled = []  # Keypoints scaled to the original image size
                    landmarks = results.pose_landmarks.landmark

                    # Extract needed landmarks
                    for landmark_enum in self.needed_landmarks:
                        landmark = landmarks[landmark_enum]
                        if landmark.visibility < visibility_threshold:
                            keypoints_cropped.append([0, 0])  # Mark low visibility points as (0, 0)
                            keypoints_scaled.append([0, 0])  # Mark low visibility points as (0, 0)
                        else:
                            # Original cropped keypoints for prediction
                            keypoints_cropped.append([landmark.x, landmark.y])

                            # Scaled keypoints for visualization
                            keypoints_scaled.append([
                                x_min + landmark.x * (x_max - x_min),
                                y_min + landmark.y * (y_max - y_min)
                            ])

                    # Add shoulder midpoint (both cropped and scaled versions)
                    left_shoulder_cropped = keypoints_cropped[needed_landmarks.index(mp.solutions.pose.PoseLandmark.LEFT_SHOULDER)]
                    right_shoulder_cropped = keypoints_cropped[needed_landmarks.index(mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER)]
                    left_shoulder_scaled = keypoints_scaled[needed_landmarks.index(mp.solutions.pose.PoseLandmark.LEFT_SHOULDER)]
                    right_shoulder_scaled = keypoints_scaled[needed_landmarks.index(mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER)]

                    if all(left_shoulder_cropped) and all(right_shoulder_cropped):
                        # Cropped shoulder midpoint
                        shoulder_midpoint_cropped = [
                            (left_shoulder_cropped[0] + right_shoulder_cropped[0]) / 2,
                            (left_shoulder_cropped[1] + right_shoulder_cropped[1]) / 2
                        ]
                    else:
                        shoulder_midpoint_cropped = [0, 0]

                    if all(left_shoulder_scaled) and all(right_shoulder_scaled):
                        # Scaled shoulder midpoint
                        shoulder_midpoint_scaled = [
                            (left_shoulder_scaled[0] + right_shoulder_scaled[0]) / 2,
                            (left_shoulder_scaled[1] + right_shoulder_scaled[1]) / 2
                        ]
                    else:
                        shoulder_midpoint_scaled = [0, 0]

                    keypoints_cropped.append(shoulder_midpoint_cropped)
                    keypoints_scaled.append(shoulder_midpoint_scaled)

                    # Add flattened cropped keypoints for prediction
                    keypoints_cropped_flat = np.array(keypoints_cropped).flatten()
                    keypoints_list.append(keypoints_cropped_flat)

                    # Draw visualization using scaled keypoints
                    for i, (x, y) in enumerate(keypoints_scaled):
                        if x != 0 and y != 0:  # Only draw visible keypoints
                            cv2.circle(image, (int(x), int(y)), 5, (0, 255, 0), -1)

                    for start_idx, end_idx in self.custom_connections:
                        start_point = keypoints_scaled[start_idx]
                        end_point = keypoints_scaled[end_idx]
                        if all(start_point) and all(end_point):  # Both points must be visible
                            cv2.line(
                                image,
                                (int(start_point[0]), int(start_point[1])),
                                (int(end_point[0]), int(end_point[1])),
                                (255, 0, 0),
                                2,
                            )
                else:
                    keypoints_list.append(None)
                    
                person += 1

        return keypoints_list, image, bboxes
        
    def predict_posture(self, model, keypoints, scaler, class_labels, sensitivity_adjustments=None):
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