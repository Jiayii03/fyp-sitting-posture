import cv2
import threading
import os
import time
from datetime import datetime

# Try to import Picamera2 but don't fail if not available (laptop environment)
try:
    from picamera2 import Picamera2
    PICAMERA_AVAILABLE = True
except ImportError:
    PICAMERA_AVAILABLE = False

class VideoManager:
    def __init__(self, am, camera_index=0, width=1920, height=1080, frame_rate=30):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.frame_rate = frame_rate
        self.camera = None  # OpenCV camera
        self.picam2 = None  # Picamera2 instance
        self.camera_lock = threading.Lock()
        self.inference_running = True
        self.alert_manager = am
        self.recording = False
        # Check if running on Raspberry Pi
        self.on_raspberry = os.environ.get('ON_RASPBERRY', 'false').lower() == 'true'
        if self.on_raspberry and not PICAMERA_AVAILABLE:
            print("⚠️ Warning: ON_RASPBERRY=true but Picamera2 is not available")
        
    def initialize(self):
        """Initialize the appropriate camera based on the platform"""
        if self.on_raspberry and PICAMERA_AVAILABLE:
            self._initialize_raspberry_camera()
        else:
            self._initialize_laptop_camera()
            
    def _initialize_laptop_camera(self):
        """Initialize the OpenCV camera for laptop"""
        print("Initializing OpenCV camera...")
        with self.camera_lock:
            if self.camera is None or not self.camera.isOpened():
                try:
                    self.camera = cv2.VideoCapture(self.camera_index, cv2.CAP_MSMF)
                    self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                    self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                    actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
                    actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = self.camera.get(cv2.CAP_PROP_FPS)
                    print(f"Camera initialized with resolution: {actual_width}x{actual_height}, FPS: {fps}")
                except Exception as e:
                    print(f"Error initializing OpenCV camera: {e}")
                    self.camera = None
    
    def _initialize_raspberry_camera(self):
        """Initialize the Picamera2 for Raspberry Pi"""
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
                
                # Set controls for frame rate and white balance
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
            except Exception as e:
                print(f"⚠️ Failed to initialize Picamera2: {e}.")
                self.picam2 = None
                
    def read_frame(self):
        """Read a frame from the appropriate camera"""
        with self.camera_lock:
            if self.on_raspberry and PICAMERA_AVAILABLE and self.picam2 is not None:
                # Raspberry Pi camera
                try:
                    frame = self.picam2.capture_array()
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    if frame is not None:
                        return True, frame
                except Exception as e:
                    print(f"Error capturing frame from Picamera2: {e}")
                    return False, None
            elif self.camera and self.camera.isOpened():
                # Laptop camera
                return self.camera.read()
            return False, None
            
    def release(self):
        """Release the appropriate camera"""
        with self.camera_lock:
            if self.on_raspberry and PICAMERA_AVAILABLE and self.picam2 is not None:
                try:
                    self.picam2.close()
                    print("✅ Picamera2 released.")
                except Exception as e:
                    print(f"⚠️ Error releasing Picamera2: {e}")
                self.picam2 = None
            elif self.camera and self.camera.isOpened():
                self.camera.release()
                self.camera = None
                print("✅ OpenCV camera released.")
                
    def stop_inference(self):
        """Stop the camera feed by releasing the camera."""
        self.release()
        self.inference_running = False
        print("🛑 Inference stopped")

    def start_inference(self):
        """Start the camera feed again with the specified resolution."""
        self.initialize()
        self.alert_manager.send_alert("🟢 Inference started: Camera feed is active.")
        self.inference_running = True
        
    def start_recording(self, output_dir="../../datasets/recordings"):
        """Start recording video from the camera (only available on laptop)."""
        if self.on_raspberry:
            return False, "Recording is not supported on Raspberry Pi"
            
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