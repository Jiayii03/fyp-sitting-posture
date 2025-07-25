import os
from dotenv import load_dotenv

load_dotenv()

# Check if running on Raspberry Pi
ON_RASPBERRY = os.environ.get('ON_RASPBERRY', 'false').lower() == 'true'

# Camera settings
DEFAULT_CAMERA_INDEX = 0

# Laptop camera settings
DEFAULT_WIDTH = 1920
DEFAULT_HEIGHT = 1080
DEFAULT_FRAME_RATE = 30

# Raspberry Pi camera settings
RASPBERRY_WIDTH = 854
RASPBERRY_HEIGHT = 480
RASPBERRY_FRAME_RATE = 20
"""
1280 × 720 (HD)
960 × 540 (qHD)
854 × 480 (480p)
640 × 360 (nHD)
426 × 240 (240p)
"""

# Get the appropriate camera settings based on platform
CAMERA_WIDTH = RASPBERRY_WIDTH if ON_RASPBERRY else DEFAULT_WIDTH
CAMERA_HEIGHT = RASPBERRY_HEIGHT if ON_RASPBERRY else DEFAULT_HEIGHT
CAMERA_FRAME_RATE = RASPBERRY_FRAME_RATE if ON_RASPBERRY else DEFAULT_FRAME_RATE

# Model settings
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
    },
    "ANN_300e_lr_1e-03_acc_8963": {
        "model_dir": "../models/2025-03-02_00-37-23",
        "model_path": "../models/2025-03-02_00-37-23/epochs_300_lr_1e-03_wd_5e-03_acc_8963.pth",
    }
}

# Alert settings
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")
BAD_POSTURE_FRAME_THRESHOLD = 200 # 10 seconds at 20 FPS
ALERT_COOLDOWN = 30  # 30 seconds

# Path settings
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

# Kafka settings - Dynamically set based on environment
KAFKA_PUBSUB_EVENT = True
LAPTOP_IP = os.environ.get("LAPTOP_IP", "172.20.10.4")
KAFKA_BROKER = f"{LAPTOP_IP}:9092"

# Resource monitoring settings
# If True, the resource monitoring will automatically stop after x minutes.
MONITORING_AUTO_STOP = True