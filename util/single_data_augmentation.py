import albumentations as A
import cv2
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import random

def create_random_augmenter():
    """Create an augmentation pipeline with random probability"""
    # Define augmentation options with individual probabilities
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),  # Apply horizontal flip with a 50% chance
        A.Rotate(limit=25, p=0.5),  # Rotate with a 50% chance
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0, rotate_limit=0, p=0.5),  # Shift only, 50% chance
    ])
    return transform

def augment_image_randomly(image, max_augmented=3):
    """Randomly generate up to max_augmented augmented images for the given image"""
    transform = create_random_augmenter()
    
    # Randomly decide the number of augmentations to generate (1 to max_augmented)
    num_augmentations = random.randint(1, max_augmented)
    augmented_images = []
    
    for _ in range(num_augmentations):
        augmented = transform(image=image)
        augmented_images.append(augmented['image'])
    
    return augmented_images

def test_random_augmentation():
    # Get the first image from proper directory
    input_directory = "C:/Users/User/Documents/UNM_CSAI/UNM_current_modules/COMP3025_Individual_Dissertation/dev/datasets/raw/proper"
    
    # Get first image
    image_path = None
    for ext in ['*.jpg', '*.png', '*.jpeg', '*.webp']:
        try:
            image_path = next(Path(input_directory).glob(ext))
            break
        except StopIteration:
            continue
    
    if image_path is None:
        raise Exception("No image files found in the directory")
    
    print(f"Testing augmentation on: {image_path.name}")
    
    # Read image
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for matplotlib
    
    # Create augmented images
    augmented_images = augment_image_randomly(image)
    
    # Create figure with subplots
    num_augmentations = len(augmented_images)
    fig, axes = plt.subplots(1, num_augmentations + 1, figsize=(15, 10))
    
    # Show original image
    axes[0].imshow(image)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    # Show augmented versions
    for i, augmented_image in enumerate(augmented_images):
        axes[i + 1].imshow(augmented_image)
        axes[i + 1].set_title(f'Augmentation {i + 1}')
        axes[i + 1].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_random_augmentation()
