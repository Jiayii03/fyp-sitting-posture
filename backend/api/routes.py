from flask import Blueprint, Response, jsonify, request
from config.settings import DEFAULT_MODEL_TYPE, MODEL_DICT, CLASS_LABELS, KAFKA_PUBSUB_EVENT
import cv2
import numpy as np
import mediapipe as mp
import time

# Create Flask Blueprint
api_bp = Blueprint('api', __name__)

# We'll set up these services when the api_bplication starts
video_manager = None
posture_detector = None
model_manager = None
alert_manager = None
kafka_service = None
resource_monitor = None

def init_services(vm, pd, mm, am, ks, rm):
    """Initialize the services that routes depend on"""
    global video_manager, posture_detector, model_manager, alert_manager, kafka_service, resource_monitor
    video_manager = vm
    posture_detector = pd
    model_manager = mm
    alert_manager = am
    kafka_service = ks
    resource_monitor = rm

@api_bp.route('/video_feed')
def video_feed():
    """Stream video feed."""
    def generate():
        print("Starting video feed...")
        while True:
            success, frame = video_manager.read_frame()
            if not success:
                print("Failed to read frame.")
                break
            # Encode the frame in JPEG format
            _, buffer = cv2.imencode('.jpg', frame)
            # Convert to byte array
            frame_bytes = buffer.tobytes()
            # Yield the frame as part of a multipart response
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@api_bp.route('/video_feed_keypoints')
def video_feed_keypoints():
    """Stream video feed with keypoints overlay and posture classification."""
    
    alert_manager.reset_state()
    
    reclining_sensitivity = request.args.get('reclining', 1)
    slouching_sensitivity = request.args.get('slouching', 1)
    crossed_legs_sensitivity = request.args.get('crossed_legs', 1)
    sensitivity_adjustments = {
        "reclining": float(reclining_sensitivity),
        "slouching": float(slouching_sensitivity),
        "crossed_legs": float(crossed_legs_sensitivity)
    }
    model_type = request.args.get('model_type', DEFAULT_MODEL_TYPE)
    model = model_manager.load_model(model_type)
    scaler = model_manager.load_scaler(MODEL_DICT[model_type]["model_dir"])
        
    last_emit_time = 0
    
    def generate():
        nonlocal last_emit_time
        while video_manager.inference_running:
            # Track the start time for this frame
            frame_start_time = time.time()
            
            success, frame = video_manager.read_frame()
            if not success:
                break

            # Get frame dimensions for dynamic font sizing
            height, width = frame.shape[:2]
            
            # Calculate dynamic font scale based on resolution
            base_width = 640
            base_font_scale = 0.7
            font_scale = (width / base_width) * base_font_scale
            font_scale = max(0.4, font_scale)  # Minimum size for readability
            
            # Calculate text thickness based on resolution
            text_thickness = max(1, int(font_scale * 2.5))

            # Process the frame for keypoints
            keypoints, frame_with_keypoints = posture_detector.extract_keypoints(frame, posture_detector.pose)

            if keypoints is not None:
                predicted_label, confidence_scores = posture_detector.predict_posture(
                    model, keypoints, scaler, CLASS_LABELS,
                    sensitivity_adjustments=sensitivity_adjustments
                )
                
                current_time = time.time()
                # When posture changes and enough time has passed, log the event
                if predicted_label != alert_manager.previous_posture and (current_time - last_emit_time) > 1:
                    alert_manager.previous_posture = predicted_label
                    last_emit_time = current_time
                    if KAFKA_PUBSUB_EVENT:
                        try:
                            # Set a timeout for Kafka operations
                            kafka_send_success = kafka_service.send_posture_event(predicted_label, f"Posture changed to: {predicted_label}")
                            if kafka_send_success:
                                print(f"📤 Sent posture event: Posture changed to: {predicted_label}")
                            else:
                                print(f"⚠️ Failed to send posture event to Kafka, but continuing inference")
                        except Exception as e:
                            print(f"⚠️ Error sending to Kafka: {e}, but continuing inference")

                # Check if an alert should be sent for the current posture
                if alert_manager.should_send_alert(predicted_label):
                    # Get the current detection count for this posture (default to 0)
                    count = alert_manager.detection_counts.get(predicted_label, 0)
                    # Send alert with the current detection count
                    alert_manager.send_alert(f"⚠️ Alert: Detected bad posture - {predicted_label}",
                        predicted_label,
                        count)
                    # Increment the detection count for this posture
                    alert_manager.detection_counts[predicted_label] = count + 1
                    
                    # kafka_service.send_alert_event(f"Detected bad posture - {predicted_label}")
                    print(f"Sent alert: Detected bad posture - {predicted_label}")

                # Set display color and feedback text
                color = (0, 255, 0) if predicted_label == "proper" else (0, 0, 255)
                
                # Calculate FPS and display it
                resource_monitor.update_fps()
                current_fps = resource_monitor.get_current_fps()
                
                # Calculate the frame processing time and update latency stats
                frame_processing_time = (time.time() - frame_start_time) * 1000  # Convert to ms
                resource_monitor.record_frame_time(frame_processing_time)
                avg_latency = resource_monitor.get_avg_latency()
                
                # Add FPS and latency to the feedback text
                feedback_text = f"Posture: {predicted_label} | FPS: {current_fps:.1f} | Latency: {avg_latency:.1f}ms"
                cv2.putText(frame_with_keypoints, feedback_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, text_thickness)

            # Encode the frame as JPEG and yield for streaming
            _, buffer = cv2.imencode('.jpg', frame_with_keypoints)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@api_bp.route('/video_feed_keypoints_multi', methods=['GET'])
def video_feed_keypoints_multi():
    """Stream video feed with keypoints overlay and multi-person posture classification."""
    
    alert_manager.reset_state()

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
    model = model_manager.load_model(model_type)
    scaler = model_manager.load_scaler(MODEL_DICT[DEFAULT_MODEL_TYPE]["model_dir"])
    
    # Track postures and last emit times for each person
    person_postures = {}
    last_emit_times = {}

    def generate():
        nonlocal person_postures, last_emit_times
        while video_manager.inference_running:
            # Track the start time for this frame
            frame_start_time = time.time()
            
            success, frame = video_manager.read_frame()
            if not success:
                break
            
            # Get frame dimensions for dynamic font sizing
            height, width = frame.shape[:2]
                        
            # Calculate dynamic font scale based on resolution
            base_width = 640
            base_font_scale = 0.7
            font_scale = (width / base_width) * base_font_scale
            font_scale = max(0.4, font_scale)  # Minimum size for readability
                        
            # Calculate text thickness based on resolution
            text_thickness = max(1, int(font_scale * 2.5))
            
            # Detect keypoints for multiple persons
            keypoints_list, frame_with_keypoints, bboxes = posture_detector.extract_keypoints_multi_person(frame, posture_detector.pose)

            if keypoints_list:
                for i, keypoints in enumerate(keypoints_list):
                    if keypoints is None or np.isnan(keypoints).any():
                        continue

                    # Assign a person ID based on index
                    person_id = f"Person {i+1}"

                    # Predict posture
                    predicted_label, confidence_scores = posture_detector.predict_posture(model, keypoints, scaler, CLASS_LABELS, sensitivity_adjustments=sensitivity_adjustments)

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
                        if KAFKA_PUBSUB_EVENT:
                            try:
                                kafka_send_success = kafka_service.send_posture_event(f"{predicted_label} [**{person_id}**]", log_message)
                                if kafka_send_success:
                                    print(f"📤 Sent posture event: {log_message}")
                                else:
                                    print(f"⚠️ Failed to send posture event to Kafka, but continuing inference")
                            except Exception as e:
                                print(f"⚠️ Error sending to Kafka: {e}, but continuing inference")
                        
                    if alert_manager.should_send_alert(predicted_label):
                        # Ensure there's a dictionary for this person
                        if person_id not in alert_manager.detection_counts_multi:
                            alert_manager.detection_counts_multi[person_id] = {}
                        
                        # Get the current detection count for this posture for this person (default to 0)
                        count = alert_manager.detection_counts_multi[person_id].get(predicted_label, 0)
                        
                        # Send alert with the current count
                        alert_manager.send_alert(
                            f"⚠️ Alert: Detected bad posture - {predicted_label} for {person_id}",
                            predicted_label,
                            count
                        )
                        
                        # Increment the detection count for this posture for this person
                        alert_manager.detection_counts_multi[person_id][predicted_label] = count + 1
                        
                        # kafka_service.send_alert_event(f"Detected bad posture - {predicted_label} for person {person_id}")
                        print(f"Sent alert: Detected bad posture - {predicted_label} for person {person_id}")

                    # Set color and feedback text
                    color = (0, 255, 0) if predicted_label == "proper" else (0, 0, 255)
                    feedback_text = f"{person_id}: {predicted_label} ({max(confidence_scores.values()):.2f})"

                    # Draw feedback text on the frame
                    cv2.putText(frame_with_keypoints, feedback_text, (x_min, y_min - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, text_thickness)
                                
            # Update FPS and latency metrics
            resource_monitor.update_fps()
            current_fps = resource_monitor.get_current_fps()
            
            # Calculate the frame processing time and update latency stats
            frame_processing_time = (time.time() - frame_start_time) * 1000  # Convert to ms
            resource_monitor.record_frame_time(frame_processing_time)
            avg_latency = resource_monitor.get_avg_latency()
            
            # Add FPS and latency to the frame
            cv2.putText(frame_with_keypoints, f"FPS: {current_fps:.1f} | Latency: {avg_latency:.1f}ms", 
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 165, 255), text_thickness)

            # Encode the frame as JPEG
            _, buffer = cv2.imencode('.jpg', frame_with_keypoints)
            frame_bytes = buffer.tobytes()

            # Yield the frame for the video feed
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@api_bp.route('/capture', methods=['GET'])
def capture_photo():
    """Capture a photo and save it to disk."""
    success, frame = video_manager.read_frame()
    if success:
        # Save the photo with a timestamp
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        photo_path = f"captured_photo_{timestamp}.jpg"
        cv2.imwrite(photo_path, frame)
        return jsonify({"message": "Photo captured successfully.", "file_path": photo_path})
    else:
        return jsonify({"error": "Failed to capture photo."}), 500
    
@api_bp.route('/capture_predict', methods=['POST'])
def capture_predict():
    """
    Capture a photo and run sitting posture inference with sensitivity adjustments.
    Accepts sensitivity adjustments exclusively via JSON request body.
    
    curl -X POST http://localhost:5000/capture_predict \
    -H "Content-Type: api_bplication/json" \
    -d '{
          "sensitivity_adjustments": {
            "reclining": 0.8,
            "slouching": 0.6,
            "crossed_legs": 0.7
          }
        }'
    """
    model_type = request.args.get('model_type', DEFAULT_MODEL_TYPE)
    model = model_manager.load_model(model_type)
    scaler = model_manager.load_scaler(MODEL_DICT[DEFAULT_MODEL_TYPE]["model_dir"])
    
    # Capture a frame from the camera
    success, frame = video_manager.read_frame()
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
    
    keypoints, _ = posture_detector.extract_keypoints(image_path, posture_detector.pose)

    if keypoints is None:
        return jsonify({"error": "No pose detected."}), 400

    # Predict posture
    predicted_label, confidence_scores = posture_detector.predict_posture(
        model, keypoints, scaler, CLASS_LABELS, sensitivity_adjustments=sensitivity_adjustments
    )

    return jsonify({
        "message": "Prediction successful.",
        "sensitivity_adjustments": sensitivity_adjustments,
        "predicted_posture": predicted_label,
        "confidence_scores": confidence_scores
    })
    
@api_bp.route('/toggle_inference', methods=['POST'])
def toggle_inference():
    """Toggle the camera feed and inference.
    curl -X POST http://localhost:5000/toggle_inference \
     -H "Content-Type: application/json" \
     -d "{\"action\": \"start\"}"
    """
    data = request.json
    action = data.get("action")

    if action == "stop":
        video_manager.stop_inference()
        return jsonify({"status": "success", "message": "Inference stopped"})
    elif action == "start":
        video_manager.start_inference()
        return jsonify({"status": "success", "message": "Inference started"})
    else:
        return jsonify({"status": "error", "message": "Invalid action"}), 400
    
@api_bp.route('/toggle_messaging_alert', methods=['POST'])
def toggle_messaging_alert():
    """Toggle the messaging alert for posture notifications."""
    data = request.json
    action = data.get("action")

    if action == "enable":
        alert_manager.enable_messaging()
        return jsonify({"status": "success", "message": "Messaging alerts enabled"}), 200
    elif action == "disable":
        alert_manager.disable_messaging()
        return jsonify({"status": "success", "message": "Messaging alerts disabled"}), 200
    else:
        return jsonify({"status": "error", "message": "Invalid action. Use 'enable' or 'disable'."}), 400
    
@api_bp.route('/is_messaging_enabled', methods=['GET'])
def is_messaging_enabled():
    """Check if messaging alerts are enabled."""
    return jsonify({"messaging_enabled": alert_manager.messaging_enabled})

@api_bp.route('/test_kafka_service', methods=['GET'])
def test_kafka_service():
    try:
        # kafka_service.send_test_event("Test event from backend.")
        print(f"Sent test kafka event")
        return jsonify({"status": "success", "message": "Test Kafka event sent."}), 200
    except Exception as e:
        print(e)
        return jsonify({"status": "error", "message": str(e)}), 500

@api_bp.route('/start_recording', methods=['POST'])
def start_recording():
    """Start recording video from the camera.
    
    curl -X POST http://localhost:5000/start_recording
    """
    
    # Make sure inference is running
    if not video_manager.inference_running:
        video_manager.start_inference()
    
    success, message = video_manager.start_recording()
    
    if success:
        return jsonify({
            "status": "success", 
            "message": message
        })
    else:
        return jsonify({
            "status": "error", 
            "message": message
        }), 400

@api_bp.route('/stop_recording', methods=['POST'])
def stop_recording():
    """Stop the current recording if one is in progress.
    
    curl -X POST http://localhost:5000/stop_recording
    """
    success, message = video_manager.stop_recording()
    
    if success:
        return jsonify({
            "status": "success", 
            "message": message,
            "output_file": video_manager.output_filename
        })
    else:
        return jsonify({
            "status": "error", 
            "message": message
        }), 400
        
# Add new routes
@api_bp.route('/resources/start', methods=['POST'])
def start_resource_monitoring():
    """
    Start resource monitoring.
    curl -X POST http://localhost:5000/resources/start
    """
    if resource_monitor:
        resource_monitor.start()
        return jsonify({"status": "success", "message": "Resource monitoring started"})
    return jsonify({"status": "error", "message": "Resource monitor not initialized"}), 500

@api_bp.route('/resources/stop', methods=['POST'])
def stop_resource_monitoring():
    """
    Stop resource monitoring and save results.
    curl -X POST http://localhost:5000/resources/stop
    """
    if resource_monitor:
        resource_monitor.stop()
        return jsonify({"status": "success", "message": "Resource monitoring stopped and data saved"})
    return jsonify({"status": "error", "message": "Resource monitor not initialized"}), 500

@api_bp.route('/resources/report', methods=['GET'])
def get_resource_report():
    """
    Get resource monitoring report as JSON.
    curl -X GET http://localhost:5000/resources/report
    """
    if resource_monitor:
        report = resource_monitor.generate_report()
        return jsonify({"status": "success", "data": report})
    return jsonify({"status": "error", "message": "Resource monitor not initialized"}), 500