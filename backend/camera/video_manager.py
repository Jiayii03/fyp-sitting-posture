import cv2
import threading

class VideoManager:
    def __init__(self, am, camera_index=0, width=1920, height=1080):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.camera = None
        self.camera_lock = threading.Lock()
        self.inference_running = True
        self.alert_manager = am
        
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