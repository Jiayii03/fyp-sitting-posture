"""
This script creates a video stream from the camera and serves it as a web API.

To run this script, execute the following command:
python stream.py --camera_index 1
"""
from flask import Flask, Response, jsonify
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
from inference.pipeline import extract_keypoints

app = Flask(__name__)

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Video stream server.")
parser.add_argument('--camera_index', type=int, default=0, help='Index of the camera to use (default: 0)')
args = parser.parse_args()

# Global camera instance
camera_lock = threading.Lock()
camera = cv2.VideoCapture(args.camera_index, cv2.CAP_MSMF)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

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
    
@app.route('/capture_predict', methods=['GET'])
def capture_predict():
    """Capture a photo and run sitting posture inference."""
    with camera_lock:
        success, frame = camera.read()
    if not success:
        return jsonify({"error": "Failed to capture frame."}), 500

    # Save the captured frame temporarily
    image_path = "temp_image.jpg"
    cv2.imwrite(image_path, frame)

    # Load the saved model
    model_path = "../models/2024-11-24_16-34-03/epochs_150_lr_1e-03_wd_1e-03_acc_8298.pth"
    scaler_mean_path = "../models/2024-11-24_16-34-03/scaler_mean.npy"
    scaler_scale_path = "../models/2024-11-24_16-34-03/scaler_scale.npy"

    # Define the class labels
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

    # Extract keypoints
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
    keypoints, _ = extract_keypoints(image_path, pose)

    if keypoints is None:
        return jsonify({"error": "No pose detected."}), 400

    # Predict posture
    keypoints_normalized = scaler.transform([keypoints])
    input_tensor = torch.tensor(keypoints_normalized, dtype=torch.float32)
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
    predicted_label = class_labels[predicted.item()]
    
    print(f"Predicted posture: {predicted_label}")

    return jsonify({"message": "Prediction successful.", "predicted_posture": predicted_label})
    

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, debug=True)
    finally:
        # Release the camera on shutdown
        camera.release()
