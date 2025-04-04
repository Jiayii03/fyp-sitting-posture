"""
Improved Body Part Correlation Chart

This script creates an enhanced visualization of the average absolute correlation by body part,
with slimmer bars, light brown color, and larger text for better readability.

Usage:
python improved_bodypart_chart.py --input_csv "full_mediapipe_keypoints.csv"
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from sklearn.preprocessing import LabelEncoder
import os

# Parse command line arguments
parser = argparse.ArgumentParser(description="Create improved body part correlation chart")
parser.add_argument('--input_csv', type=str, default="full_mediapipe_keypoints.csv", 
                    help='CSV file containing keypoint data')
args = parser.parse_args()

def main():
    # Load the data
    print(f"Loading data from {args.input_csv}...")
    try:
        df = pd.read_csv(args.input_csv)
    except Exception as e:
        print(f"Error loading CSV file: {str(e)}")
        return
    
    # Encode the class labels as numeric values for correlation analysis
    label_encoder = LabelEncoder()
    df['class_encoded'] = label_encoder.fit_transform(df['class'])
    
    # Extract feature columns (all columns containing '_x' or '_y')
    feature_cols = [col for col in df.columns if ('_x' in col or '_y' in col)]
    
    # Calculate correlation between each feature and the encoded class
    correlations = []
    for col in feature_cols:
        correlation = df[col].corr(df['class_encoded'])
        correlations.append((col, correlation))
    
    # Create a DataFrame with the correlations
    corr_df = pd.DataFrame(correlations, columns=['Feature', 'Correlation'])
    corr_df['Abs_Correlation'] = corr_df['Correlation'].abs()
    
    # Define body part groups
    body_parts = {
        'Upper Body': ['left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 
                      'left_wrist', 'right_wrist'],
        'Face': ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'mouth'],
        'Lower Body': ['left_hip', 'right_hip', 'left_knee', 'right_knee', 
                      'left_ankle', 'right_ankle'],
        'Feet': ['left_heel', 'right_heel', 'left_foot_index', 'right_foot_index'],
        'Hands': ['left_pinky', 'right_pinky', 'left_index', 'right_index', 
                 'left_thumb', 'right_thumb']
    }
    
    # Calculate average absolute correlation for each body part
    body_part_corr = {}
    for part, keypoints in body_parts.items():
        part_features = []
        for kp in keypoints:
            part_features.extend([f for f in feature_cols if kp in f])
        
        if part_features:
            avg_corr = corr_df[corr_df['Feature'].isin(part_features)]['Abs_Correlation'].mean()
            body_part_corr[part] = avg_corr
    
    # Create improved bar chart of body part correlations
    plt.figure(figsize=(12, 8))
    parts = list(body_part_corr.keys())
    values = list(body_part_corr.values())
    
    # Sort by correlation value
    sorted_indices = np.argsort(values)[::-1]
    sorted_parts = [parts[i] for i in sorted_indices]
    sorted_values = [values[i] for i in sorted_indices]
    
    # Light brown color and slimmer bars
    light_brown = '#C19A6B'  # Sandy/light brown color
    bar_width = 0.5  # Slimmer bars
    
    # Create bar chart with improved styling
    plt.figure(figsize=(12, 8))
    
    # Create bar chart with custom width
    ax = plt.bar(range(len(sorted_parts)), sorted_values, width=bar_width, color=light_brown)
    
    # Customize the chart
    plt.title('Average Absolute Correlation by Body Part', fontsize=22)
    plt.ylabel('Absolute Correlation', fontsize=18)
    plt.xticks(range(len(sorted_parts)), sorted_parts, rotation=0, fontsize=16)
    plt.yticks(fontsize=14)
    
    # Add value labels on top of bars
    for i, v in enumerate(sorted_values):
        plt.text(i, v + 0.005, f'{v:.3f}', ha='center', fontsize=14)
    
    # Set y-axis limits with a bit of padding
    plt.ylim(0, max(sorted_values) * 1.15)
    
    # Add grid lines for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Improve overall appearance
    plt.tight_layout()
    
    # Save the improved chart
    output_dir = 'correlation_analysis_output'
    os.makedirs(output_dir, exist_ok=True)
    improved_chart_path = os.path.join(output_dir, 'improved_bodypart_correlation.png')
    plt.savefig(improved_chart_path, dpi=300)
    print(f"Saved improved body part correlation chart to {improved_chart_path}")
    
    # Also save as SVG for vector quality
    svg_path = os.path.join(output_dir, 'improved_bodypart_correlation.svg')
    plt.savefig(svg_path)
    print(f"Also saved as vector SVG: {svg_path}")

if __name__ == "__main__":
    main()