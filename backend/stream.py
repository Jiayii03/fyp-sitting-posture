"""
This script creates a video stream from the camera and serves it as a web API.

To run this script, execute the following command:
python stream.py --camera_index 0

API Endpoints:
- /video_feed: Streams the video feed from the camera. GET request.
- /video_feed_keypoints: Streams the video feed from the camera with keypoints overlay and class labels. GET request.
- /video_feed_keypoints_multi: Streams the video feed from the camera with keypoints overlay and multi-person class labels. GET request.
- /capture: Captures a photo from the camera and saves it to disk. GET request.
- /capture_predict: Captures a photo from the camera, runs sitting posture inference, and returns the predicted posture as a json. POST request.
"""
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from kafka import KafkaProducer
from dotenv import load_dotenv
import cv2
import sys
import requests
import os
import argparse
import threading
import time
import torch
import json
import subprocess
import mediapipe as mp
from sklearn.preprocessing import StandardScaler
import numpy as np
import importlib.util
import warnings
warnings.filterwarnings("ignore")

load_dotenv()
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(PROJECT_ROOT) 

from inference.pipeline import extract_keypoints, predict_posture
from inference.pipeline_multi import extract_keypoints_multi_person
from helpers.recommendation_generator import get_recommendation

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

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
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
BAD_POSTURE_FRAME_THRESHOLD = 600 # 10 seconds
ALERT_COOLDOWN = 30 # seconds

# Global variables
model_cache = {}
alert_state = {
    "last_alert_time": 0,
    "bad_posture_count": 0,
    "previous_posture": None
}
inference_running = True
messaging_alert_enabled = False
camera_lock = threading.Lock()
camera = None

def start_docker_containers():
    """Start Kafka and Zookeeper containers if they are not already running."""
    try:
        # Check if Kafka container is running
        kafka_running = subprocess.run(["docker", "ps", "-q", "-f", "name=kafka"], capture_output=True, text=True).stdout.strip()
        zookeeper_running = subprocess.run(["docker", "ps", "-q", "-f", "name=zookeeper"], capture_output=True, text=True).stdout.strip()

        if not kafka_running or not zookeeper_running:
            print("Kafka or Zookeeper container not running. Starting Kafka and Zookeeper containers...")
            # Run Docker Compose to start the containers
            subprocess.run(["docker-compose", "-f", "docker-compose.yml", "up", "-d"], check=True)

            # Wait for Kafka and Zookeeper to become available
            time.sleep(10) # Wait a few seconds to allow Kafka and Zookeeper to initialize
            print("Kafka and Zookeeper containers started successfully.")
        else:
            print("Kafka and Zookeeper containers are already running.")
    except Exception as e:
        print(f"Error while starting Docker containers: {e}")

def initialize_camera(camera_index=0, width=1920, height=1080):
    """Initialize the camera with the given index and resolution."""
    global camera
    with camera_lock:
        if camera is None or not camera.isOpened():
            camera = cv2.VideoCapture(camera_index, cv2.CAP_MSMF)
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            actual_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = camera.get(cv2.CAP_PROP_FPS)
            print(f"Camera initialized with resolution: {actual_width}x{actual_height}, FPS: {fps}")
            
def send_telegram_alert(message, predicted_label=None, detection_count=0):
    """Spawn a background thread to send a Telegram alert with a deepseek recommendation."""
    if messaging_alert_enabled:
        # Start a background thread to process the alert
        thread = threading.Thread(
            target=_send_telegram_alert_worker, 
            args=(message, predicted_label, detection_count)
        )
        thread.daemon = True  # Optional: allows program exit even if thread is running
        thread.start()

def _send_telegram_alert_worker(message, predicted_label, detection_count):
    """Worker function that performs the blocking deepseek call and sends the alert."""
    if predicted_label is not None:
        # Build a formatted header for the deepseek recommendation section
        deepseek_header = (
            "*DeepSeek Recommendation:*\n"
        )
        # Generate recommendation in the background thread
        recommendation = get_recommendation(predicted_label, detection_count)
        # Combine the header and recommendation
        formatted_recommendation = deepseek_header + recommendation
        # Append the formatted recommendation block to the main message
        message += f"\n\n{formatted_recommendation}"
    
    # Example professional alert header (if you want to include it in the message)
    professional_message = (
        "**🚨 Posture Alert 🚨**\n"
        f"**Detected Issue:** *{predicted_label}*\n\n"
        f"{message}"
    )
    
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": professional_message, "parse_mode": "Markdown"}
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        print("✅ Telegram alert sent successfully!")
    else:
        print(f"❌ Failed to send Telegram alert: {response.text}")
        
def reset_alert_state():
    """Reset all global alert state variables."""
    alert_state["last_alert_time"] = 0
    alert_state["bad_posture_count"] = 0
    alert_state["previous_posture"] = None
    
def should_send_alert(predicted_label):
    """Combined approach: Cooldown + Stability."""
    current_time = time.time()

    # Check if the posture is improper
    if predicted_label != "proper":
        # Increment bad posture count if posture remains the same
        if predicted_label == alert_state["previous_posture"]:
            alert_state["bad_posture_count"] += 1
        else:
            alert_state["bad_posture_count"] = 1

        # Check stability and cooldown
        if (alert_state["bad_posture_count"] >= BAD_POSTURE_FRAME_THRESHOLD and
            current_time - alert_state["last_alert_time"] > ALERT_COOLDOWN):
            alert_state["last_alert_time"] = current_time
            alert_state["bad_posture_count"] = 0
            return True

    else:
        # Reset count if posture returns to normal
        alert_state["bad_posture_count"] = 0

    alert_state["previous_posture"] = predicted_label
    return False

def stop_inference():
    """Stop the camera feed by releasing the camera."""
    global inference_running, camera
    with camera_lock:
        if camera and camera.isOpened():
            camera.release()
            print("Camera feed and inference stopped.")
    inference_running = False

def start_inference():
    """Start the camera feed again with the specified resolution."""
    send_telegram_alert("✅ Inference started: Camera feed is now active.")
    global inference_running
    initialize_camera()
    inference_running = True
    print("Camera feed and inference started.")

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
scaler = load_scaler(MODEL_DICT[DEFAULT_MODEL_TYPE]["model_dir"])

start_docker_containers()

# Initialize Kafka producer
producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

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
    
    global inference_running
    reset_alert_state()
    
    reclining_sensitivity = request.args.get('reclining', 1)
    slouching_sensitivity = request.args.get('slouching', 1)
    crossed_legs_sensitivity = request.args.get('crossed_legs', 1)
    sensitivity_adjustments = {
        "reclining": float(reclining_sensitivity),
        "slouching": float(slouching_sensitivity),
        "crossed_legs": float(crossed_legs_sensitivity)
    }
    model_type = request.args.get('model_type', DEFAULT_MODEL_TYPE)
    model = load_model(model_type)
    
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    last_emit_time = 0
    previous_posture = None
    detection_counts = {} 

    def generate():
        nonlocal last_emit_time, previous_posture, detection_counts
        while inference_running:
            with camera_lock:
                success, frame = camera.read()
            if not success:
                break

            # Process the frame for keypoints
            keypoints, frame_with_keypoints = extract_keypoints(frame, pose)

            if keypoints is not None:
                predicted_label, confidence_scores = predict_posture(
                    model, keypoints, scaler, CLASS_LABELS,
                    sensitivity_adjustments=sensitivity_adjustments
                )
                
                current_time = time.time()
                # When posture changes and enough time has passed, log the event
                if predicted_label != previous_posture and (current_time - last_emit_time) > 1:
                    previous_posture = predicted_label
                    last_emit_time = current_time
                    producer.send('posture_events', {
                        'posture': predicted_label,
                        'message': f"Posture changed to: {predicted_label}"
                    })
                    print(f"📤 Sent posture event: Posture changed to: {predicted_label}")

                # Check if an alert should be sent for the current posture
                if should_send_alert(predicted_label):
                    # Get the current detection count for this posture (default to 0)
                    count = detection_counts.get(predicted_label, 0)
                    # Send alert with the current detection count
                    send_telegram_alert(
                        f"⚠️ Alert: Detected bad posture - {predicted_label}",
                        predicted_label,
                        count
                    )
                    # Increment the detection count for this posture
                    detection_counts[predicted_label] = count + 1
                    
                    producer.send('alert_events', {
                        'message': f"Detected bad posture - {predicted_label}"
                    })
                    print(f"Sent alert: Detected bad posture - {predicted_label}")

                # Set display color and feedback text
                color = (0, 255, 0) if predicted_label == "proper" else (0, 0, 255)
                feedback_text = f"Posture: {predicted_label}"
                cv2.putText(frame_with_keypoints, feedback_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            # Encode the frame as JPEG and yield for streaming
            _, buffer = cv2.imencode('.jpg', frame_with_keypoints)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_keypoints_multi', methods=['GET'])
def video_feed_keypoints_multi():
    """Stream video feed with keypoints overlay and multi-person posture classification."""
    
    global inference_running
    reset_alert_state()

    # Sensitivity adjustments from query parameters
    reclining_sensitivity = request.args.get('reclining', 1)
    slouching_sensitivity = request.args.get('slouching', 1)
    crossed_legs_sensitivity = request.args.get('crossed_legs', 1)
    sensitivity_adjustments = {
        "reclining": float(reclining_sensitivity),
        "slouching": float(slouching_sensitivity),
        "crossed_legs": float(crossed_legs_sensitivity)
    }
    model_type = request.args.get('model_type', DEFAULT_MODEL_TYPE)
    model = load_model(model_type)
    
    # Initialize MediaPipe pose detector
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Load YOLO model for person detection
    yolo_model_path = "../models/yolo-human-detection/yolov5n.pt"
    yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path=yolo_model_path, force_reload=False)
    yolo_model.classes = [0]  # Only detect persons (COCO class ID: 0)
    
    # Track postures and last emit times for each person
    person_postures = {}
    last_emit_times = {}

    def generate():
        while inference_running:
            with camera_lock:
                success, frame = camera.read()
            if not success:
                break

            # Detect keypoints for multiple persons
            keypoints_list, frame_with_keypoints, bboxes = extract_keypoints_multi_person(frame, pose, yolo_model)

            if keypoints_list:
                for i, keypoints in enumerate(keypoints_list):
                    if keypoints is None or np.isnan(keypoints).any():
                        continue

                    # Assign a person ID based on index
                    person_id = f"Person {i+1}"

                    # Predict posture
                    predicted_label, confidence_scores = predict_posture(model, keypoints, scaler, CLASS_LABELS, sensitivity_adjustments=sensitivity_adjustments)

                    # Get bounding box for the person
                    x_min, y_min, x_max, y_max = bboxes[i]

                    # Track posture changes and emit logs if changed
                    current_time = time.time()
                    previous_posture = person_postures.get(person_id, None)
                    last_emit_time = last_emit_times.get(person_id, 0)

                    if predicted_label != previous_posture and (current_time - last_emit_time) > 1:
                        # Update stored posture and last emit time
                        person_postures[person_id] = predicted_label
                        last_emit_times[person_id] = current_time

                        # Log the posture change
                        log_message = f"[{person_id}]: Changed to {predicted_label}"
                        producer.send('posture_events', {
                            'posture': f"{predicted_label} [**{person_id}**]",
                            'message': log_message
                        })
                        print(f"📤 Sent posture event: {log_message}")
                        
                    if should_send_alert(predicted_label):
                        send_telegram_alert(f"⚠️ Alert: Detected bad posture - {predicted_label} - {person_id}", predicted_label)
                        producer.send('alert_events', {
                            'message': f"Detected bad posture - {predicted_label} [**{person_id}**]"
                        })
                        print(f"Sent alert: Detected bad posture - {predicted_label}")

                    # Set color and feedback text
                    color = (0, 255, 0) if predicted_label == "proper" else (0, 0, 255)
                    feedback_text = f"{person_id}: {predicted_label} ({max(confidence_scores.values()):.2f})"

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
    
    curl -X POST http://localhost:5000/capture_predict \
    -H "Content-Type: application/json" \
    -d '{
          "sensitivity_adjustments": {
            "reclining": 0.8,
            "slouching": 0.6,
            "crossed_legs": 0.7
          }
        }'
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
    
@app.route('/toggle_inference', methods=['POST'])
def toggle_inference():
    """Toggle the camera feed and inference.
    curl -X POST http://localhost:5000/toggle_inference \
     -H "Content-Type: application/json" \
     -d "{\"action\": \"start\"}"
    """
    global inference_running
    data = request.json
    action = data.get("action")

    if action == "stop":
        stop_inference()
        return jsonify({"status": "success", "message": "Inference stopped"})
    elif action == "start":
        start_inference()
        return jsonify({"status": "success", "message": "Inference started"})
    else:
        return jsonify({"status": "error", "message": "Invalid action"}), 400
    
@app.route('/toggle_messaging_alert', methods=['POST'])
def toggle_messaging_alert():
    """Toggle the messaging alert for posture notifications."""
    global messaging_alert_enabled
    data = request.json
    action = data.get("action")

    if action == "enable":
        messaging_alert_enabled = True
        return jsonify({"status": "success", "message": "Messaging alerts enabled"}), 200
    elif action == "disable":
        messaging_alert_enabled = False
        return jsonify({"status": "success", "message": "Messaging alerts disabled"}), 200
    else:
        return jsonify({"status": "error", "message": "Invalid action. Use 'enable' or 'disable'."}), 400
    
if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, debug=True)
    finally:
        # Release the camera on shutdown
        camera.release()
