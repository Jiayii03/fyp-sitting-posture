import albumentations as A
import cv2
import os
from pathlib import Path
import numpy as np

def create_augmenter():
    """Create an augmentation pipeline"""
    transform = A.Compose([
        # Spatial augmentations
        A.HorizontalFlip(p=1.0),
        A.Rotate(limit=15, p=1.0),
        A.Shift(limit=0.1, p=1.0),  # shift by up to 10% of image size
        
        # Optional: you can add more augmentations here
        # A.RandomBrightnessContrast(p=0.2),
        # A.GaussNoise(p=0.2),
        # A.MotionBlur(p=0.2),
    ])
    return transform

def augment_image(image, output_path, filename, class_name, num_augmentations):
    """Apply various augmentations to an image and save them."""
    # Create directory for augmented images if it doesn't exist
    aug_dir = Path(output_path) / class_name
    aug_dir.mkdir(parents=True, exist_ok=True)
    
    # Get filename without extension
    base_name = Path(filename).stem
    
    # Save original image
    cv2.imwrite(str(aug_dir / f"{base_name}_orig.jpg"), image)
    
    # Create augmentation pipeline
    transform = create_augmenter()
    
    # Generate multiple augmented versions
    for i in range(num_augmentations):
        augmented = transform(image=image)
        augmented_image = augmented['image']
        cv2.imwrite(str(aug_dir / f"{base_name}_aug_{i}.jpg"), augmented_image)

def process_directory(input_path, output_path, num_augmentations):
    """Process all images in the input directory and its subdirectories."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Process each class directory
    for class_dir in input_path.iterdir():
        if class_dir.is_dir():
            class_name = class_dir.name
            print(f"Processing class: {class_name}")
            
            # Process each image in the class directory
            for img_path in class_dir.glob("*.jpg, *.png, *.jpeg, *.webp"):  
                print(f"Augmenting image: {img_path.name}")
                image = cv2.imread(str(img_path))
                if image is not None:
                    augment_image(image, output_path, img_path.name, 
                                class_name, num_augmentations)

if __name__ == "__main__":
    # Set your input and output directories
    input_directory = "C:/Users/User/Documents/UNM_CSAI/UNM_current_modules/COMP3025_Individual_Dissertation/dev/datasets/raw"
    output_directory = "C:/Users/User/Documents/UNM_CSAI/UNM_current_modules/COMP3025_Individual_Dissertation/dev/datasets/augmented"
    
    # Set number of augmented versions to create for each image
    num_augmentations = 2
    
    process_directory(input_directory, output_directory, num_augmentations)
    print("Augmentation completed!")
