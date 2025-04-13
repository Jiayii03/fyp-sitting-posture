import mediapipe as mp
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
from config.settings import ON_RASPBERRY

# Conditionally import TFLite or PyTorch
if ON_RASPBERRY:
    try:
        import tflite_runtime.interpreter as tflite
    except ImportError:
        print("Warning: tflite_runtime not available on Raspberry Pi")
        tflite = None
else:
    tflite = None

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
        self.pose = self.mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=0)
        self.yolo_model = self._load_yolo_model()
        self.needed_landmarks = needed_landmarks
        self.custom_connections = custom_connections
        self.input_size = (416, 416)
        
    def _load_yolo_model(self):
        """Load the appropriate YOLO model based on the platform"""
        if ON_RASPBERRY and tflite is not None:
            # Use TFLite for Raspberry Pi
            print("Loading TFLite YOLO model for Raspberry Pi...")
            tflite_model_path = "../models/yolo-human-detection/yolov5n-fp16.tflite"
            
            if not os.path.exists(tflite_model_path):
                print(f"TFLite model not found at {tflite_model_path}. Please convert the PyTorch model first.")
                print("You can use the export script from YOLOv5 repository:")
                print("python export.py --weights ../models/yolo-human-detection/yolov5n.pt --include tflite")
                return None
            
            interpreter = tflite.Interpreter(model_path=tflite_model_path)
            interpreter.allocate_tensors()
            
            # Get input and output details
            self.input_details = interpreter.get_input_details()
            self.output_details = interpreter.get_output_details()
            
            return interpreter
        else:
            # Use PyTorch YOLOv5 for other platforms
            print("Loading PyTorch YOLO model...")
            try:
                # Check if torch is available
                import torch
                
                # Try to import the YOLOv5 model
                try:
                    model_path = "../models/yolo-human-detection/yolov5n.pt"
                    if os.path.exists(model_path):
                        # Try to load with torch.hub first
                        try:
                            model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
                        except Exception:
                            # If torch.hub fails, try direct loading
                            from models.yolo import attempt_load
                            model = attempt_load(model_path, device='cpu')
                        
                        model.eval()
                        return model
                    else:
                        print(f"PyTorch model not found at {model_path}")
                        return None
                except ImportError:
                    print("YOLOv5 model implementation not found. Trying to load from torch hub...")
                    try:
                        # Attempt to load a pre-trained model from torch hub
                        model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
                        model.eval()
                        return model
                    except Exception as e:
                        print(f"Failed to load YOLOv5 from torch hub: {e}")
                        return None
            except ImportError:
                print("PyTorch not available. Cannot load YOLO model.")
                return None
    
    def _process_tflite_output(self, outputs, img_shape, conf_threshold=0.25, nms_threshold=0.45):
        """Process TFLite YOLO output to get bounding boxes for persons"""
        height, width = img_shape[:2]
        boxes = []
        confidences = []
        
        # Inspect the output to understand its format
        # YOLOv5 TFLite outputs can vary in format
        output = outputs[0]  # Usually the first output tensor contains detection results
        
        if len(output.shape) == 3:  # Format [1, n, 85]
            for detection in output[0]:
                # Extract box data - format is usually [x, y, w, h, obj_conf, class1_conf, class2_conf, ...]
                if len(detection) < 6:  # Need at least x,y,w,h,obj_conf,class1_conf
                    continue
                    
                # Get confidence and class
                confidence = float(detection[4])  # Object confidence
                
                # If there are multiple classes
                if len(detection) > 5:
                    class_scores = detection[5:]
                    class_id = np.argmax(class_scores)
                    class_conf = float(class_scores[class_id])
                    # Combined confidence
                    score = confidence * class_conf
                else:
                    # Only one class (person)
                    class_id = 0
                    score = confidence
                
                # Filter by confidence and class (person = 0)
                if score > conf_threshold and class_id == 0:
                    # YOLOv5 outputs normalized coordinates [cx, cy, w, h]
                    cx, cy, w, h = detection[0:4]
                    
                    # Convert to corner format and scale to image size
                    x1 = int((cx - w/2) * width)
                    y1 = int((cy - h/2) * height)
                    x2 = int((cx + w/2) * width)
                    y2 = int((cy + h/2) * height)
                    
                    # Ensure coordinates are within image bounds
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(width, x2)
                    y2 = min(height, y2)
                    
                    # Add to lists for NMS
                    boxes.append([x1, y1, x2-x1, y2-y1])  # [x, y, w, h]
                    confidences.append(score)
        
        # Apply non-maximum suppression to remove overlapping boxes
        if boxes:
            indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
            
            result = []
            for i in indices:
                # Handle both OpenCV 3.x and 4.x return types
                if isinstance(i, (list, tuple, np.ndarray)):
                    i = i[0]
                    
                box = boxes[i]
                x, y, w, h = box
                result.append({
                    'x1': x,
                    'y1': y,
                    'x2': x + w,
                    'y2': y + h,
                    'confidence': confidences[i]
                })
            
            return result
        
        return []  # Return empty list if no detections
        
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
        
    def extract_keypoints_multi_person(self, image_input, pose_model, confidence_threshold=0.30, visibility_threshold=0.65):
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
        if isinstance(image_input, str):
            image = cv2.imread(image_input)
        else:
            image = image_input

        if image is None:
            raise ValueError("Invalid image input. Ensure the file path or frame is correct.")

        if self.yolo_model is None:
            raise ValueError("YOLO model not loaded properly.")

        # Detect persons using the appropriate model
        if ON_RASPBERRY and tflite is not None and isinstance(self.yolo_model, tflite.Interpreter):
            # TFLite model for Raspberry Pi
            input_shape = self.input_details[0]['shape'][1:3]
            input_tensor = cv2.resize(image, (input_shape[1], input_shape[0]))
            input_tensor = input_tensor.astype(np.float32) / 255.0
            input_tensor = np.expand_dims(input_tensor, axis=0)
            
            self.yolo_model.set_tensor(self.input_details[0]['index'], input_tensor)
            self.yolo_model.invoke()
            
            outputs = []
            for output_detail in self.output_details:
                output_data = self.yolo_model.get_tensor(output_detail['index'])
                outputs.append(output_data)
            
            detections = self._process_tflite_output(outputs, image.shape, conf_threshold=confidence_threshold)
        else:
            # PyTorch model for other platforms
            try:
                # Resize image to match YOLO's expected size
                resized_img = cv2.resize(image, self.input_size)
                results = self.yolo_model(resized_img)
                
                # Convert YOLOv5 results to the same format as our TFLite output processor
                # Results format may vary based on YOLOv5 version
                detections = []
                for detection in results.xyxy[0]:  # Format: x1, y1, x2, y2, confidence, class
                    if detection[5] == 0:  # Class 0 is person
                        if detection[4] >= confidence_threshold:  # Check confidence
                            h, w = image.shape[:2]
                            # Scale coordinates to original image size
                            x1 = int(detection[0].item() * (w / self.input_size[0]))
                            y1 = int(detection[1].item() * (h / self.input_size[1]))
                            x2 = int(detection[2].item() * (w / self.input_size[0]))
                            y2 = int(detection[3].item() * (h / self.input_size[1]))
                            
                            detections.append({
                                'x1': x1,
                                'y1': y1,
                                'x2': x2,
                                'y2': y2,
                                'confidence': detection[4].item()
                            })
            except Exception as e:
                print(f"Error using PyTorch YOLO model: {e}")
                detections = []
        
        keypoints_list = []
        bboxes = []
        person = 1
        
        for detection in detections:
            x_min, y_min = detection['x1'], detection['y1']
            x_max, y_max = detection['x2'], detection['y2']
            confidence = detection['confidence']
            
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
            
            if cropped_person.size == 0:  # Skip if cropped image is empty
                continue

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