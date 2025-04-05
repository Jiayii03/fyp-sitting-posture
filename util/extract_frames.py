"""
This script helps extract frames from a video at specified timestamps.

run with:
python extract_frames.py
"""

import cv2
import os
import numpy as np

def extract_frames(video_path, timestamps, output_folder):
    """
    Extract frames with exact color preservation.
    Places frames in a subfolder named after the video file.
    """
    # Get the video filename without extension to use as subfolder name
    video_filename = os.path.basename(video_path)
    video_name = os.path.splitext(video_filename)[0]
    
    # Create output folder structure: output_folder/video_name/
    video_output_folder = os.path.join(output_folder, video_name)
    if not os.path.exists(video_output_folder):
        os.makedirs(video_output_folder)

    # Open the video
    video = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not video.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Get video properties
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    print(f"Video FPS: {fps}")
    print(f"Video duration: {duration:.2f} seconds")
    print(f"Saving frames to: {video_output_folder}")

    # Process timestamps
    timestamps_in_seconds = []
    for timestamp in timestamps:
        if isinstance(timestamp, str) and ":" in timestamp:
            parts = timestamp.split(":")
            if len(parts) == 2:
                minutes, seconds = map(float, parts)
                timestamps_in_seconds.append(minutes * 60 + seconds)
            elif len(parts) == 3:
                hours, minutes, seconds = map(float, parts)
                timestamps_in_seconds.append(
                    hours * 3600 + minutes * 60 + seconds)
        else:
            timestamps_in_seconds.append(float(timestamp))

    # Extract frames
    for timestamp in timestamps_in_seconds:
        # Convert timestamp to frame number
        frame_number = int(timestamp * fps)

        if frame_number >= total_frames:
            print(
                f"Warning: Timestamp {timestamp}s exceeds video duration. Skipping.")
            continue

        # Set video to the desired frame
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        # Read the frame
        success, frame = video.read()

        if success:
            # Format timestamp for filename
            timestamp_str = f"{int(timestamp//60):02d}_{int(timestamp % 60):02d}"
            output_path = os.path.join(
                video_output_folder, f"frame_{timestamp_str}.png")

            # Save with highest quality PNG settings
            cv2.imwrite(output_path, frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])

            print(
                f"Extracted frame at {timestamp:.2f}s and saved to {output_path}")
        else:
            print(f"Error: Failed to extract frame at {timestamp:.2f}s")

    # Release video
    video.release()


# Example usage
if __name__ == "__main__":
    # Replace with your video path
    video_path = "../../datasets/recordings/IMG_4115.mp4"

    # Replace with your list of timestamps in seconds
    timestamps = [
       "0:12",
       "0.15",
       "0:18",
       "0:22",
       "0:25",
       "0:33",
       "0:38",
       "0:40",
    ]

    output_folder = "../../datasets/recordings/extracted_frames"  

    extract_frames(video_path, timestamps, output_folder)