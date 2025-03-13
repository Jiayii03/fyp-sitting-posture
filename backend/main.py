"""
Refactored main.py to use create_app function from api.server module.

To run the server, execute the following command from the root directory:
python main.py --camera_index 0

"""

from api.server import create_app
import sys
import os
from config.settings import PROJECT_ROOT
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Add project root to path
sys.path.append(PROJECT_ROOT)

def parse_args():
    """Parse command line arguments"""
    import argparse
    parser = argparse.ArgumentParser(description="Video stream server.")
    parser.add_argument('--camera_index', type=int, default=0, 
                        help='Index of the camera to use (default: 0)')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    # Override camera index if provided
    os.environ['CAMERA_INDEX'] = str(args.camera_index)
    
    # Create and run the app
    app = create_app()
    try:
        print("Starting posture detection server...")
        app.run(host='0.0.0.0', port=5000, debug=False)
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        print("Server shut down.")