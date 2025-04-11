"""
Unified main.py that works on both laptop and Raspberry Pi.

To run the server on a laptop with a webcam:
python main.py --camera_index 0

To run the server on a Raspberry Pi with Picamera2:
python main.py --raspberry

"""

from api.server import create_app
import sys
import os
from config.settings import PROJECT_ROOT, ON_RASPBERRY
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Add project root to path
sys.path.append(PROJECT_ROOT)

def parse_args():
    """Parse command line arguments"""
    import argparse
    parser = argparse.ArgumentParser(description="Video stream server for posture detection.")
    parser.add_argument('--camera_index', type=int, default=0, 
                        help='Index of the camera to use (default: 0, only applicable on laptop)')
    parser.add_argument('--raspberry', action='store_true',
                        help='Force Raspberry Pi mode (alternative to setting ON_RASPBERRY=true)')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    # Set Raspberry Pi mode if specified in command line arguments
    if args.raspberry:
        os.environ['ON_RASPBERRY'] = 'true'
        print("Running in Raspberry Pi mode (set by command line argument)")
    elif ON_RASPBERRY:
        print("Running in Raspberry Pi mode (set by environment variable)")
    else:
        print("Running in laptop mode")
    
    # Override camera index if provided (only for laptop mode)
    if not ON_RASPBERRY and args.camera_index is not None:
        os.environ['CAMERA_INDEX'] = str(args.camera_index)
    
    # Create and run the app
    app = create_app()
    try:
        # Use debug=False on Raspberry Pi to avoid issues with reloader
        debug_mode = False
        port = 5001 if ON_RASPBERRY else 5000
        print(f"Starting posture detection server with debug={debug_mode}...")
        app.run(host='0.0.0.0', port=port, debug=debug_mode)
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        print("Server shut down.")