"""
Correlation Analysis for MediaPipe Keypoints

This script analyzes the Pearson correlation between MediaPipe keypoint features and class labels,
then visualizes the results using multiple techniques.

Usage:
python correlation_analysis.py --input_csv "full_mediapipe_keypoints.csv"
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from sklearn.preprocessing import LabelEncoder
import os

# Parse command line arguments
parser = argparse.ArgumentParser(description="Analyze correlation between keypoints and class labels")
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
    
    print(f"Dataset shape: {df.shape}")
    
    # Encode the class labels as numeric values for correlation analysis
    print("Encoding class labels...")
    label_encoder = LabelEncoder()
    df['class_encoded'] = label_encoder.fit_transform(df['class'])
    class_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    print(f"Class mapping: {class_mapping}")
    
    # Extract feature columns (all columns containing '_x' or '_y')
    feature_cols = [col for col in df.columns if ('_x' in col or '_y' in col)]
    print(f"Found {len(feature_cols)} feature columns")
    
    # Calculate correlation between each feature and the encoded class
    print("Calculating Pearson correlation coefficients...")
    correlations = []
    for col in feature_cols:
        correlation = df[col].corr(df['class_encoded'])
        correlations.append((col, correlation))
    
    # Create a DataFrame with the correlations
    corr_df = pd.DataFrame(correlations, columns=['Feature', 'Correlation'])
    corr_df['Abs_Correlation'] = corr_df['Correlation'].abs()
    
    # Sort by absolute correlation value (most important first)
    corr_df = corr_df.sort_values('Abs_Correlation', ascending=False)
    
    # Create output directory if it doesn't exist
    output_dir = 'correlation_analysis_output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the correlation results to CSV
    corr_csv_path = os.path.join(output_dir, 'feature_correlations.csv')
    corr_df.to_csv(corr_csv_path, index=False)
    print(f"Saved correlation results to {corr_csv_path}")
    
    # Show top and bottom correlated features
    print("\nTop 10 most correlated features:")
    print(corr_df.head(10))
    
    print("\nBottom 10 least correlated features:")
    print(corr_df.tail(10))

    # Create visualizations
    print("\nCreating visualizations...")
    
    # 1. Top 20 features bar chart
    plt.figure(figsize=(12, 10))
    top_20 = corr_df.head(20)
    colors = ['red' if x < 0 else 'blue' for x in top_20['Correlation']]
    sns.barplot(x='Correlation', y='Feature', data=top_20, palette=colors)
    plt.title('Top 20 Features by Correlation with Class')
    plt.tight_layout()
    bar_chart_path = os.path.join(output_dir, 'top_20_correlation_barchart.png')
    plt.savefig(bar_chart_path)
    print(f"Saved bar chart to {bar_chart_path}")
    
    # 2. Heat map of correlation matrix for the top 20 features
    plt.figure(figsize=(14, 12))
    # Create a DataFrame with only top features and the class
    top_features = top_20['Feature'].tolist()
    corr_matrix = df[top_features + ['class_encoded']].corr()
    
    # Create a mask for the upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Create the heatmap
    sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', annot=True, 
                fmt='.2f', linewidths=0.5, vmin=-1, vmax=1)
    plt.title('Correlation Matrix of Top 20 Features')
    plt.tight_layout()
    heatmap_path = os.path.join(output_dir, 'top_20_correlation_heatmap.png')
    plt.savefig(heatmap_path)
    print(f"Saved heatmap to {heatmap_path}")
    
    # 3. Grouped bar chart (x vs y coordinates)
    plt.figure(figsize=(14, 10))
    
    # Extract base names (without _x or _y)
    corr_df['Base_Feature'] = corr_df['Feature'].apply(lambda x: x[:-2])
    
    # Group by base feature and create a pivot table
    pivot_df = corr_df.pivot_table(index='Base_Feature', 
                                  columns=corr_df['Feature'].str[-1], 
                                  values='Correlation')
    
    # Sort by the maximum absolute correlation for each base feature
    pivot_df['max_abs'] = pivot_df.abs().max(axis=1)
    pivot_df = pivot_df.sort_values('max_abs', ascending=False).drop('max_abs', axis=1)
    
    # Take top 15 base features
    top_pivot = pivot_df.head(15)
    
    # Plot grouped bar chart
    top_pivot.plot(kind='bar', figsize=(14, 10))
    plt.title('Correlation of X vs Y Coordinates for Top 15 Keypoints')
    plt.xlabel('Keypoint')
    plt.ylabel('Correlation with Class')
    plt.legend(['X Coordinate', 'Y Coordinate'])
    plt.tight_layout()
    grouped_bar_path = os.path.join(output_dir, 'xy_comparison_barchart.png')
    plt.savefig(grouped_bar_path)
    print(f"Saved grouped bar chart to {grouped_bar_path}")
    
    # 4. Body part correlation visualization
    plt.figure(figsize=(14, 12))
    
    # Define body part groups
    body_parts = {
        'Face': ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'mouth'],
        'Upper Body': ['left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 
                      'left_wrist', 'right_wrist'],
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
    
    # Create bar chart of body part correlations
    plt.figure(figsize=(10, 6))
    parts = list(body_part_corr.keys())
    values = list(body_part_corr.values())
    
    # Sort by correlation value
    sorted_indices = np.argsort(values)[::-1]
    sorted_parts = [parts[i] for i in sorted_indices]
    sorted_values = [values[i] for i in sorted_indices]
    
    sns.barplot(x=sorted_parts, y=sorted_values)
    plt.title('Average Absolute Correlation by Body Part')
    plt.ylabel('Absolute Correlation')
    plt.xticks(rotation=45)
    plt.tight_layout()
    bodypart_chart_path = os.path.join(output_dir, 'bodypart_correlation.png')
    plt.savefig(bodypart_chart_path)
    print(f"Saved body part correlation chart to {bodypart_chart_path}")
    
    print("\nAnalysis complete! All results saved to the 'correlation_analysis_output' directory.")

if __name__ == "__main__":
    main()