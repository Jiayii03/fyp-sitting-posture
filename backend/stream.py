"""
This script creates a video stream from the camera and serves it as a web API.

To run this script, execute the following command:
python stream.py --camera_index 1

API Endpoints:
- /video_feed: Streams the video feed from the camera. GET request.
- /video_feed_keypoints: Streams the video feed from the camera with keypoints overlay. GET request.
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

# Add the directory containing the model to sys.path
model_dir = "../models/2024-11-24_16-34-03"
sys.path.append(model_dir)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(project_root)

# Import the MLP class
from model import MLP
from source.inference.pipeline_multi import extract_keypoints, predict_posture

app = Flask(__name__)

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Video stream server.")
parser.add_argument('--camera_index', type=int, default=0, help='Index of the camera to use (default: 0)')
args = parser.parse_args()

# Global camera instance
camera_lock = threading.Lock()
camera = cv2.VideoCapture(args.camera_index, cv2.CAP_MSMF)

# Try setting a higher resolution
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Check if the camera supports it
width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Attempted resolution: {width}x{height}")

# Load the saved model and scaler
model_path = "../models/2024-11-24_16-34-03/epochs_150_lr_1e-03_wd_1e-03_acc_8298.pth"
scaler_mean_path = "../models/2024-11-24_16-34-03/scaler_mean.npy"
scaler_scale_path = "../models/2024-11-24_16-34-03/scaler_scale.npy"
class_labels = ["crossed_legs", "proper", "reclining", "slouching"]

# Load model
input_size = 20  # Update this based on your keypoints input size
model = MLP(input_size=input_size, num_classes=len(class_labels))
model.load_state_dict(torch.load(model_path))
model.eval()

# Load scaler
scaler = StandardScaler()
scaler.mean_ = np.load(scaler_mean_path)
scaler.scale_ = np.load(scaler_scale_path)

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
                predicted_label, confidence_scores = predict_posture(model, keypoints, scaler, class_labels)

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
        model, keypoints, scaler, class_labels, sensitivity_adjustments=sensitivity_adjustments
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
