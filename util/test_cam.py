"""
Test USB camera.

run `python test_cam.py --find_cameras` to find available cameras
"""

import cv2
import argparse

# take parameters from command line
parser = argparse.ArgumentParser(description="Test USB camera.")
parser.add_argument("--find_cameras", action="store_true", help="Find available cameras")
args = parser.parse_args()

def find_available_cameras():
    index = 0
    arr = []
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.read()[0]:
            break
        else:
            arr.append(index)
        cap.release()
        index += 1
    return arr

if args.find_cameras:
    cameras = find_available_cameras()
    print("Available cameras:", cameras)
    exit()

# Open the USB webcam (change the index to your USB camera's index, e.g., 1)
camera_index = 1
cap = cv2.VideoCapture(camera_index)

if not cap.isOpened():
    print(f"Error: Unable to access camera {camera_index}")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    cv2.imshow("USB Webcam", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

