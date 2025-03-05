from flask import Flask
from flask_cors import CORS
from .routes import api_bp, init_services
from camera.video_manager import VideoManager
from inference.posture_detector import PostureDetector
from inference.models import ModelManager
from alerts.alert_manager import AlertManager
from pubSub.kafka_service import KafkaService
from config.settings import RASPBERRY_HEIGHT, RASPBERRY_WIDTH, RASPBERRY_FRAME_RATE

def create_app():
    """Create and configure the Flask application"""
    app = Flask(__name__)
    CORS(app, resources={r"/*": {"origins": "*"}})

    model_manager = ModelManager()
    
    # PostureDetector needs access to the model_manager
    posture_detector = PostureDetector(model_manager)
    
    # AlertManager for handling posture alerts
    alert_manager = AlertManager()
    
    # KafkaService for event messaging
    kafka_service = KafkaService()
    
    # Initialize all services
    video_manager = VideoManager(
        am=alert_manager,
        width=RASPBERRY_WIDTH,
        height=RASPBERRY_HEIGHT,
        fr=RASPBERRY_FRAME_RATE
    )
    
    # Initialize the camera
    video_manager.initialize()
    
    # Inject services into routes
    init_services(
        video_manager,
        posture_detector,
        model_manager,
        alert_manager,
        kafka_service
    )
    
    # Register blueprint
    app.register_blueprint(api_bp)
    
    return app