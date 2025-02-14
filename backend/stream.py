"""
This script creates a video stream from the camera and serves it as a web API.

To run this script, execute the following command:
python stream.py --camera_index 1

API Endpoints:
- /video_feed: Streams the video feed from the camera. GET request.
- /video_feed_keypoints: Streams the video feed from the camera with keypoints overlay and class labels. GET request.
- /video_feed_keypoints_multi: Streams the video feed from the camera with keypoints overlay and multi-person class labels. GET request.
- /capture: Captures a photo from the camera and saves it to disk. GET request.
- /capture_predict: Captures a photo from the camera, runs sitting posture inference, and returns the predicted posture as a json. POST request.
"""
from flask import Flask, Response, jsonify, request
import cv2
import sys
import os
import argparse
import threading
import time
import torch
import mediapipe as mp
from sklearn.preprocessing import StandardScaler
import numpy as np
import importlib.util
import warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(PROJECT_ROOT)

# SOURCE_DIR = os.path.join(PROJECT_ROOT, "source")
# sys.path.append(SOURCE_DIR)

from inference.pipeline import extract_keypoints, predict_posture
from inference.pipeline_multi import extract_keypoints_multi_person

app = Flask(__name__)

INPUT_SIZE = 20
CLASS_LABELS = ["crossed_legs", "proper", "reclining", "slouching"]
DEFAULT_MODEL_TYPE = "ANN_150e_lr_1e-03_acc_8298"
MODEL_DICT = {
    "ANN_150e_lr_1e-03_acc_8298": {
        "model_dir": "../models/2024-11-24_16-34-03",
        "model_path": "../models/2024-11-24_16-34-03/epochs_150_lr_1e-03_wd_1e-03_acc_8298.pth",
    },
    "ANN_50e_lr_1e-03_acc_76": {
        "model_dir": "../models/2024-11-24_00-04-05",
        "model_path": "../models/2024-11-24_00-04-05/epochs_50_lr_1e-03_acc_76.pth",
    }
}
model_cache = {} # Cache for loaded models

# Function to initialise the camera with the given index and resolution
def initialize_camera(camera_index, width=1920, height=1080):
    """Initialize the camera with the given index and resolution."""
    camera = cv2.VideoCapture(camera_index, cv2.CAP_MSMF)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    actual_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera initialized with resolution: {actual_width}x{actual_height}")
    return camera

# Dynamically import the MLP class from the specified model directory.
def import_mlp_from_dir(model_dir):
    sys.path.insert(0, model_dir)
    spec = importlib.util.spec_from_file_location("MLP", os.path.join(model_dir, "model.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.path.pop(0)
    return module.MLP

# Function to load model dynamically based on query parameter
def load_model(model_type):
    if model_type in model_cache:
        print(f"Model loaded from cache: {model_type}")
        return model_cache[model_type]
    
    model_dir = MODEL_DICT[model_type]["model_dir"]
    MLP = import_mlp_from_dir(model_dir)
    model_path = MODEL_DICT[model_type]["model_path"]
    model = MLP(input_size=INPUT_SIZE, num_classes=len(CLASS_LABELS)) 
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model_cache[model_type] = model
    print(f"Model loaded: {model_type}")
    return model

# Function to load scaler
def load_scaler(model_dir):
    scaler = StandardScaler()
    scaler.mean_ = np.load(os.path.join(model_dir, 'scaler_mean.npy'))
    scaler.scale_ = np.load(os.path.join(model_dir, 'scaler_scale.npy'))
    return scaler

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Video stream server.")
parser.add_argument('--camera_index', type=int, default=0, help='Index of the camera to use (default: 0)')
args = parser.parse_args()

camera_lock = threading.Lock()
camera = initialize_camera(args.camera_index) 
scaler = load_scaler(MODEL_DICT[DEFAULT_MODEL_TYPE]["model_dir"])

@app.route('/video_feed')
def video_feed():
    """Stream video feed."""
    def generate():
        while True:
            with camera_lock:
                success, frame = camera.read()
            if not success:
                break
            # Encode the frame in JPEG format
            _, buffer = cv2.imencode('.jpg', frame)
            # Convert to byte array
            frame_bytes = buffer.tobytes()
            # Yield the frame as part of a multipart response
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_keypoints')
def video_feed_keypoints():
    """Stream video feed with keypoints overlay and posture classification."""
    
    model_type = request.args.get('model_type', DEFAULT_MODEL_TYPE)
    model = load_model(model_type)
    
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def generate():
        while True:
            with camera_lock:
                success, frame = camera.read()
            if not success:
                break

            # Process the frame for keypoints
            keypoints, frame_with_keypoints = extract_keypoints(frame, pose)

            if keypoints is not None:
                predicted_label, confidence_scores = predict_posture(model, keypoints, scaler, CLASS_LABELS)

                # Set color and feedback text
                color = (0, 255, 0) if predicted_label == "proper" else (0, 0, 255)
                feedback_text = f"Posture: {predicted_label}"

                # Draw feedback on the frame
                cv2.putText(frame_with_keypoints, feedback_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            # Encode the frame as JPEG
            _, buffer = cv2.imencode('.jpg', frame_with_keypoints)
            frame_bytes = buffer.tobytes()

            # Yield the frame for the video feed
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_keypoints_multi', methods=['GET'])
def video_feed_keypoints_multi():
    """Stream video feed with keypoints overlay and multi-person posture classification."""
    
    model_type = request.args.get('model_type', DEFAULT_MODEL_TYPE)
    model = load_model(model_type)
    
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Load YOLO model for person detection
    yolo_model_path = "../models/yolo-human-detection/yolov5n.pt"
    yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path=yolo_model_path, force_reload=False)
    yolo_model.classes = [0]  # Only detect persons (COCO class ID: 0)

    def generate():
        while True:
            with camera_lock:
                success, frame = camera.read()
            if not success:
                break

            # Process the frame with YOLO and Mediapipe
            keypoints_list, frame_with_keypoints, bboxes = extract_keypoints_multi_person(frame, pose, yolo_model)

            if keypoints_list:
                for i, keypoints in enumerate(keypoints_list):
                    if keypoints is None or np.isnan(keypoints).any():
                        continue

                    # Predict posture for each detected person
                    predicted_label, confidence_scores = predict_posture(model, keypoints, scaler, CLASS_LABELS)

                    # Get bounding box for the person
                    x_min, y_min, x_max, y_max = bboxes[i]

                    # Set color and feedback text
                    color = (0, 255, 0) if predicted_label == "proper" else (0, 0, 255)
                    feedback_text = f"Person {i + 1}: {predicted_label} ({max(confidence_scores.values()):.2f})"

                    # Draw feedback text on the frame
                    cv2.putText(frame_with_keypoints, feedback_text, (x_min, y_min - 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Encode the frame as JPEG
            _, buffer = cv2.imencode('.jpg', frame_with_keypoints)
            frame_bytes = buffer.tobytes()

            # Yield the frame for the video feed
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture', methods=['GET'])
def capture_photo():
    """Capture a photo and save it to disk."""
    with camera_lock:
        success, frame = camera.read()
    if success:
        # Save the photo with a timestamp
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        photo_path = f"captured_photo_{timestamp}.jpg"
        cv2.imwrite(photo_path, frame)
        return jsonify({"message": "Photo captured successfully.", "file_path": photo_path})
    else:
        return jsonify({"error": "Failed to capture photo."}), 500
    
@app.route('/capture_predict', methods=['POST'])
def capture_predict():
    """
    Capture a photo and run sitting posture inference with sensitivity adjustments.
    Accepts sensitivity adjustments exclusively via JSON request body.
    
    curl -X POST -H "Content-Type: application/json" -d '{
    "sensitivity_adjustments": {
        "reclining": 0.8,
    }
    }' "http://localhost:5000/capture_predict"
    """
    model_type = request.args.get('model_type', DEFAULT_MODEL_TYPE)
    model = load_model(model_type)
    
    # Capture a frame from the camera
    with camera_lock:
        success, frame = camera.read()
    if not success:
        return jsonify({"error": "Failed to capture frame."}), 500
    
    # Get sensitivity adjustments from JSON body
    if not request.is_json:
        return jsonify({"error": "Request body must be JSON."}), 400

    sensitivity_adjustments = request.json.get("sensitivity_adjustments", {})
    if not isinstance(sensitivity_adjustments, dict):
        return jsonify({"error": "Invalid sensitivity adjustments format."}), 400

    # Ensure sensitivity adjustments values are floats
    try:
        sensitivity_adjustments = {k: float(v) for k, v in sensitivity_adjustments.items()}
    except ValueError:
        return jsonify({"error": "Sensitivity adjustments values must be numbers."}), 400

    # Save the captured frame temporarily
    image_path = "temp_image.jpg"
    cv2.imwrite(image_path, frame)
    
    # Extract keypoints
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
    keypoints, _ = extract_keypoints(image_path, pose)

    if keypoints is None:
        return jsonify({"error": "No pose detected."}), 400

    # Predict posture
    predicted_label, confidence_scores = predict_posture(
        model, keypoints, scaler, CLASS_LABELS, sensitivity_adjustments=sensitivity_adjustments
    )

    return jsonify({
        "message": "Prediction successful.",
        "sensitivity_adjustments": sensitivity_adjustments,
        "predicted_posture": predicted_label,
        "confidence_scores": confidence_scores
    })
    

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, debug=True)
    finally:
        # Release the camera on shutdown
        camera.release()
