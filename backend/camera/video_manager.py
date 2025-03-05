import threading
import time
from picamera2 import Picamera2
import cv2

class VideoManager:
    def __init__(self, am, width=640, height=360, fr=30):
        self.width = width
        self.height = height
        self.frame_rate = fr
        self.picam2 = None
        self.camera_lock = threading.Lock()
        self.inference_running = True
        self.alert_manager = am

    def initialize(self):
        print("Initializing Picamera2...")
        with self.camera_lock:
            # If picam2 already exists, release it first
            if self.picam2 is not None:
                try:
                    self.picam2.close()
                    print("✅ Camera released before reinitializing.")
                except Exception as e:
                    print(f"⚠️ Error releasing camera: {e}")
                self.picam2 = None

            try:
                self.picam2 = Picamera2()
                config = self.picam2.create_preview_configuration(
                    main={"size": (self.width, self.height)}
                )
                self.picam2.configure(config)
                
                # Set controls for a 30 FPS frame rate and natural white balance
                self.picam2.set_controls({
                    "FrameRate": self.frame_rate,
                    "AwbEnable": True,
                    "AwbMode": 5  # Daylight mode for better color
                })
                
                # Disable digital zoom by setting ScalerCrop to the full sensor resolution
                sensor_resolution = self.picam2.camera_properties['PixelArraySize']
                self.picam2.set_controls({"ScalerCrop": (0, 0, sensor_resolution[0], sensor_resolution[1])})
                
                time.sleep(1)  # Allow time for the camera to initialize
                self.picam2.start()
                print("✅ Picamera2 initialized successfully!")
            except RuntimeError as e:
                print(f"⚠️ Failed to initialize camera: {e}. Another process may be using it.")
                self.picam2 = None

    def read_frame(self):
        with self.camera_lock:
            if self.picam2 is not None:
                frame = self.picam2.capture_array()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if frame is not None:
                    return True, frame
            return False, None

    def release(self):
        with self.camera_lock:
            if self.picam2 is not None:
                try:
                    self.picam2.close()
                    print("✅ Picamera2 released.")
                except Exception as e:
                    print(f"⚠️ Error releasing camera: {e}")
                self.picam2 = None

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
