import time
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import sys

# Add OpenPose Python API to the system path
sys.path.append("C:/Users/User/openpose/python")  # Update the path to your OpenPose 'python' folder

# Function to measure Mediapipe inference time
def run_mediapipe(image):
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    start_time = time.time()
    results = pose.process(image)
    end_time = time.time()
    return end_time - start_time

# Function to measure YOLO inference time
def run_yolo(image):
    from yolov5 import YOLO  # Ensure you have YOLO installed and configured
    model = YOLO('path_to_yolo_model')

    start_time = time.time()
    results = model(image)
    end_time = time.time()
    return end_time - start_time

# Function to measure OpenPose inference time
def run_openpose(image):
    from openpose import pyopenpose as op
    params = dict()
    params["model_folder"] = "C:/Users/User/openpose/models"  # Correct model folder path
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    datum = op.Datum()
    datum.cvInputData = image
    start_time = time.time()
    opWrapper.emplaceAndPop([datum])
    end_time = time.time()
    return end_time - start_time

# Main function to compute average inference times
def compute_average_inference_time(image_dir, num_images=100):
    image_paths = list(Path(image_dir).glob("*.jpg"))[:num_images]
    inference_times = {"Mediapipe": [], "YOLO": [], "OpenPose": []}

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Run Mediapipe
        # mediapipe_time = run_mediapipe(image)
        # inference_times["Mediapipe"].append(mediapipe_time)

        # # Run YOLO
        # yolo_time = run_yolo(image)
        # inference_times["YOLO"].append(yolo_time)

        # Run OpenPose
        openpose_time = run_openpose(image)
        inference_times["OpenPose"].append(openpose_time)

    # Calculate average inference times
    avg_inference_times = {algo: sum(times) / len(times) for algo, times in inference_times.items()}
    return avg_inference_times

# Plotting function
def plot_inference_times(avg_inference_times):
    algorithms = list(avg_inference_times.keys())
    times = list(avg_inference_times.values())

    plt.bar(algorithms, times, color=['blue', 'orange', 'green'])
    plt.title('Average Inference Time Comparison')
    plt.xlabel('Algorithm')
    plt.ylabel('Time (seconds)')
    plt.show()

if __name__ == "__main__":
    image_directory = r"C:\Users\User\Documents\UNM_CSAI\UNM_current_modules\COMP3025_Individual_Dissertation\dev\datasets\raw\crossed_legs"
    print("Running inference on 100 images...")

    avg_times = compute_average_inference_time(image_directory)
    print("Average Inference Times:", avg_times)

    plot_inference_times(avg_times)
