"""
This script renames the images of all classes in the raw directory to a sequential number.  

To run the script, use the following command:
python rename_images.py <class_name>
"""

import os
import argparse
import shutil

raw_dir = "C:/Users/User/Documents/UNM_CSAI/UNM_current_modules/COMP3025_Individual_Dissertation/dev/datasets/raw"

# add argument parser to get the specific class directory
parser = argparse.ArgumentParser(description='Rename images in a specific class directory')
parser.add_argument('class_name', type=str, help='Name of the class directory (e.g., crossed_legs, proper, reclining, slouching)')
args = parser.parse_args()

# Get all class directories
class_dirs = [d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))]

# Loop through all class directories
for class_dir in class_dirs:
    if class_dir == "junk":
        continue

    if class_dir == args.class_name:
        class_path = os.path.join(raw_dir, class_dir)
        temp_dir = os.path.join(raw_dir, f"{class_dir}_temp")
        
        # Create temporary directory
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        try:
            # Get all images in the class directory and sort them
            images = sorted([f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))])

            # First, move all files to temp directory with new names
            for i, image in enumerate(images, 1):
                old_path = os.path.join(class_path, image)
                extension = os.path.splitext(image)[1].lower()
                new_name = f"{i:03d}{extension}"  # Use 3 digits with leading zeros
                temp_path = os.path.join(temp_dir, new_name)
                shutil.move(old_path, temp_path)
                print(f"Moved {image} to temporary location as {new_name}")

            # Then move them back to original directory
            for image in os.listdir(temp_dir):
                temp_path = os.path.join(temp_dir, image)
                final_path = os.path.join(class_path, image)
                shutil.move(temp_path, final_path)
                print(f"Moved {image} to final location")

        finally:
            # Clean up: remove temporary directory
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)

print("Renaming complete!")
