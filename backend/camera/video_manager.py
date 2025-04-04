import cv2
import threading
import os 
from datetime import datetime
import time

class VideoManager:
    def __init__(self, am, camera_index=0, width=1920, height=1080):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.camera = None
        self.camera_lock = threading.Lock()
        self.inference_running = True
        self.alert_manager = am
        self.recording = False
        
    def initialize(self):
        print("Initializing camera...")
        with self.camera_lock:
            if self.camera is None or not self.camera.isOpened():
                self.camera = cv2.VideoCapture(self.camera_index, cv2.CAP_MSMF)
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = self.camera.get(cv2.CAP_PROP_FPS)
                print("Camera instance:", self.camera)
                print(f"Camera initialized with resolution: {actual_width}x{actual_height}, FPS: {fps}")
                
    def read_frame(self):
        with self.camera_lock:
            if self.camera and self.camera.isOpened():
                return self.camera.read()
            return False, None
            
    def release(self):
        with self.camera_lock:
            if self.camera and self.camera.isOpened():
                self.camera.release()
                self.camera = None
                
    def stop_inference(self):
        """Stop the camera feed by releasing the camera."""
        self.release()
        # self.alert_manager.send_alert("🚨 Inference stopped: Camera feed is inactive.")
        self.inference_running = False

    def start_inference(self):
        """Start the camera feed again with the specified resolution."""
        self.initialize()
        self.alert_manager.send_alert("🟢 Inference started: Camera feed is active.")
        self.inference_running = True
        
    def start_recording(self, output_dir="../../datasets/recordings"):
        """Start recording video from the camera."""
        if self.recording:
            return False, "Recording already in progress"
        
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate a filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_filename = os.path.join(output_dir, f"recording_{timestamp}.mp4")
        
        # Get camera properties
        if self.camera is None:
            return False, "Camera not initialized"
        
        width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.camera.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            fps = 30  # Default FPS if not detected correctly
        
        # Initialize the video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for MP4 format
        self.video_writer = cv2.VideoWriter(
            self.output_filename, fourcc, fps, (width, height)
        )
        
        # Start recording in a separate thread
        self.recording = True
        self.recording_thread = threading.Thread(target=self._record_video)
        self.recording_thread.daemon = True
        self.recording_thread.start()
        
        return True, f"Started recording to {self.output_filename}"
    
    def stop_recording(self):
        """Stop the current recording if one is in progress."""
        if not self.recording:
            return False, "No recording in progress"
        
        self.recording = False
        
        # Wait for the recording thread to finish
        if self.recording_thread and self.recording_thread.is_alive():
            self.recording_thread.join(timeout=2.0)
        
        # Release the video writer
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        
        return True, f"Recording stopped and saved to {self.output_filename}"
    
    def _record_video(self):
        """Thread function that handles the recording process."""
        while self.recording:
            if self.camera is None or not self.camera.isOpened():
                break
                
            success, frame = self.read_frame()
            if not success:
                break
                
            if self.video_writer:
                self.video_writer.write(frame)
                
            # Small delay to not overload the CPU
            time.sleep(0.01)