from flask import Flask
from flask_cors import CORS
from .routes import api_bp, init_services
from camera.video_manager import VideoManager
from inference.posture_detector import PostureDetector
from inference.models import ModelManager
from alerts.alert_manager import AlertManager
from pubSub.kafka_service import KafkaService
from util.resource_monitor import ResourceMonitor
from config.settings import CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FRAME_RATE, DEFAULT_CAMERA_INDEX

def create_app():
    """Create and configure the Flask application"""
    app = Flask(__name__)
    CORS(app, resources={r"/*": {"origins": "*"}})
    
    # Initialize resource monitor
    resource_monitor = ResourceMonitor(sampling_interval=1.0)  # 1-second interval
    resource_monitor.start()

    model_manager = ModelManager()
    
    # PostureDetector needs access to the model_manager
    posture_detector = PostureDetector(model_manager)
    
    # AlertManager for handling posture alerts
    alert_manager = AlertManager()
    
    # KafkaService for event messaging
    kafka_service = KafkaService()
    
    # Initialize all services with platform-appropriate settings
    video_manager = VideoManager(
        am=alert_manager,
        camera_index=DEFAULT_CAMERA_INDEX,
        width=CAMERA_WIDTH,
        height=CAMERA_HEIGHT,
        frame_rate=CAMERA_FRAME_RATE
    )
    
    # Initialize the camera
    video_manager.initialize()
    
    # Inject services into routes
    init_services(
        video_manager,
        posture_detector,
        model_manager,
        alert_manager,
        kafka_service,
        resource_monitor
    )
    
    # Register blueprint
    app.register_blueprint(api_bp)
    
    return app