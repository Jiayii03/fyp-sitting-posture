import albumentations as A
import cv2
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import random

def create_augmenter():
    """Create an augmentation pipeline"""
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),  # Apply horizontal flip with a 50% chance
        A.Rotate(limit=15, p=0.5),  # Apply rotation with a 50% chance
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0, rotate_limit=0, p=0.5),  # Apply shift only, 50% chance
    ])
    return transform

def get_next_image_number(output_dir):
    """Get the next available image number in the output directory"""
    existing_files = list(output_dir.glob("*.jpg"))  # Adjust if using different extensions
    if not existing_files:
        return 1
    
    # Extract numbers from existing filenames and find the maximum
    numbers = []
    for file in existing_files:
        try:
            num = int(file.stem)
            numbers.append(num)
        except ValueError:
            continue
    
    return max(numbers) + 1 if numbers else 1

def process_image(image_path, output_dir, transform, max_augmentations=3):
    """Process a single image and save its augmented versions"""
    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Failed to read image: {image_path}")
        return
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get the next available number for naming
    next_num = get_next_image_number(output_dir)

    next_num += 1
    
    # Randomize the number of augmentations for this image
    num_augmentations = random.randint(1, max_augmentations)
    
    # Generate augmented versions
    for i in range(num_augmentations):
        augmented = transform(image=image)
        augmented_image = augmented['image']
        
        # Save augmented image with sequential numbering
        aug_output_path = output_dir / f"{next_num:03d}.jpg"
        cv2.imwrite(str(aug_output_path), augmented_image)
        next_num += 1

def process_directory(input_dir, output_base_dir, max_augmentations=3):
    """Process all images in all class directories"""
    input_dir = Path(input_dir)
    output_base_dir = Path(output_base_dir)
    transform = create_augmenter()
    
    # Supported image extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.webp']
    
    # Process each class directory
    for class_dir in input_dir.iterdir():
        if class_dir.is_dir():

            # skip directory 'crowded' and 'junk'
            if class_dir.name in ['crowded', 'junk']:
                continue

            print(f"\nProcessing class: {class_dir.name}")
            
            # Create output directory for this class
            output_dir = output_base_dir / class_dir.name
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Process each image in the class directory
            image_count = 0
            for ext in image_extensions:
                for image_path in class_dir.glob(f"*{ext}"):
                    print(f"Processing image: {image_path.name}")
                    process_image(image_path, output_dir, transform, max_augmentations)
                    image_count += 1
            
            print(f"Processed {image_count} images in {class_dir.name}")

if __name__ == "__main__":
    # Set your input and output directories
    input_directory = "C:/Users/User/Documents/UNM_CSAI/UNM_current_modules/COMP3025_Individual_Dissertation/dev/datasets/raw"
    output_directory = "C:/Users/User/Documents/UNM_CSAI/UNM_current_modules/COMP3025_Individual_Dissertation/dev/datasets/augmented"
    
    # Maximum number of augmented versions to create for each image
    max_augmentations = 2
    
    print(f"Starting batch augmentation process...")
    print(f"Input directory: {input_directory}")
    print(f"Output directory: {output_directory}")
    print(f"Maximum number of augmentations per image: {max_augmentations}")
    
    process_directory(input_directory, output_directory, max_augmentations)
    
    print("\nAugmentation process completed!")
