"""
This script renames the images of all classes in the raw directory to a sequential number.  

To run the script, use the following command:
python rename_images.py <class_name>
"""

import os
import argparse
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
        # Get all images in the class directory
        images = [f for f in os.listdir(os.path.join(raw_dir, class_dir)) if os.path.isfile(os.path.join(raw_dir, class_dir, f))]

        # Loop through all images in the class directory
        for i, image in enumerate(images):
            # Get the old image path
            old_image_path = os.path.join(raw_dir, class_dir, image)

            # Get the extension of the image
            extension = os.path.splitext(image)[1]

            # Create the new image path
            new_image_path = os.path.join(raw_dir, class_dir, f"{i+1}{extension}")

            # Rename the image
            os.rename(old_image_path, new_image_path)

            print(f"Renamed {old_image_path} to {new_image_path}")
    
