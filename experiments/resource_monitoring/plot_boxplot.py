import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

def plot_comparative_boxplots(unoptimized_csv, optimized_csv):
    """
    Create report-friendly boxplots with larger fonts and narrower width.
    
    Args:
        unoptimized_csv (str): Path to the CSV file with unoptimized statistics
        optimized_csv (str): Path to the CSV file with optimized statistics
    """
    # Read the CSV files
    df_unopt = pd.read_csv(unoptimized_csv)
    df_opt = pd.read_csv(optimized_csv)
    
    # Add a column to identify the source
    df_unopt['optimization'] = 'Unoptimized'
    df_opt['optimization'] = 'Optimized'
    
    # Combine the dataframes
    df_combined = pd.concat([df_unopt, df_opt])
    
    # Define the metrics to plot
    metrics = ['cpu_percent', 'memory_percent', 'temperature', 
              'inference_fps', 'inference_latency_ms']
    
    # Check if temperature has valid data (not all zeros)
    if df_combined['temperature'].max() == 0:
        # Remove temperature if it's not available (all zeros)
        metrics.remove('temperature')
    
    # Create nicer labels for the metrics
    metric_labels = {
        'cpu_percent': 'CPU Usage (%)',
        'memory_percent': 'Memory Usage (%)',
        'temperature': 'Temperature (°C)',
        'inference_fps': 'Inference FPS',
        'inference_latency_ms': 'Latency (ms)'
    }
    
    # Set up the figure with a clean, modern style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Create a figure with narrower width
    # Reduce the width per subplot to make the overall figure more compact
    fig, axes = plt.subplots(1, len(metrics), figsize=(2.5*len(metrics), 6))
    plt.suptitle('Resource Metrics: Unoptimized vs Optimized', fontsize=13, y=0.98)
    
    # Define colors for optimization state - more pleasant pastel colors
    palette = {'Unoptimized': '#f8c1c1', 'Optimized': '#c1e1c1'}  # Lighter pastel red and green
    
    # Plot each metric
    for i, metric in enumerate(metrics):
        # Get the current axis
        ax = axes[i]
        
        # Create the side-by-side boxplot without showing outliers
        # Reduce the width parameter to make boxes narrower
        sns.boxplot(
            x='optimization', 
            y=metric, 
            data=df_combined, 
            ax=ax, 
            palette=palette,
            width=0.5,  # Narrower boxes
            showfliers=False  # This hides the outliers
        )
        
        # Calculate statistics for unoptimized data
        unopt_median = df_unopt[metric].median()
        unopt_mean = df_unopt[metric].mean()
        unopt_min = df_unopt[metric].quantile(0.05)  # 5th percentile
        unopt_max = df_unopt[metric].quantile(0.95)  # 95th percentile
        
        # Calculate statistics for optimized data
        opt_median = df_opt[metric].median()
        opt_mean = df_opt[metric].mean()
        opt_min = df_opt[metric].quantile(0.05)  # 5th percentile
        opt_max = df_opt[metric].quantile(0.95)  # 95th percentile
        
        # Determine a better y-axis range based on the data distribution
        # Get the overall min and max across both datasets (with a small margin)
        overall_min = min(unopt_min, opt_min) * 0.95
        overall_max = max(unopt_max, opt_max) * 1.05
        
        # Adjust the y-axis limits to focus on the relevant range
        ax.set_ylim(overall_min, overall_max)
        
        # Configure the axes with larger fonts
        ax.set_title(metric_labels[metric], fontsize=12)
        ax.set_ylabel('Value', fontsize=11)
        ax.set_xlabel('')
        
        # Add percent improvement text with larger font
        if metric in ['cpu_percent', 'memory_percent', 'inference_latency_ms', 'temperature']:
            # Lower is better
            percent_change = ((opt_mean - unopt_mean) / unopt_mean) * 100
            improvement_text = f"{percent_change:.1f}% {'higher' if percent_change > 0 else 'lower'}"
            color = '#006400' if percent_change < 0 else '#8B0000'  # Dark green or dark red
        else:
            # Higher is better (inference_fps)
            percent_change = ((opt_mean - unopt_mean) / unopt_mean) * 100
            improvement_text = f"{percent_change:.1f}% {'higher' if percent_change > 0 else 'lower'}"
            color = '#006400' if percent_change > 0 else '#8B0000'  # Dark green or dark red
        
        # Add the improvement text with a nicer style and larger font
        ax.text(0.5, 0.02, improvement_text, 
                horizontalalignment='center',
                verticalalignment='bottom',
                transform=ax.transAxes,
                color=color,
                fontweight='bold',
                fontsize=11,  # Larger font
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=3))
        
        # Add mean lines with labels
        ax.axhline(unopt_mean, 0.1, 0.3, color='#FF6666', linestyle='--', linewidth=1.5)
        ax.axhline(opt_mean, 0.7, 0.9, color='#66CC66', linestyle='--', linewidth=1.5)
        
        # Set clean x-tick labels with larger font
        ax.set_xticklabels(['Unoptimized', 'Optimized'], fontsize=11)
        
        # Make y-tick labels easier to read with larger font
        ax.tick_params(axis='y', labelsize=11)
        
        # Add grid lines for better readability
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust layout - make it tighter to reduce white space
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Create output filename 
    unopt_base = os.path.splitext(os.path.basename(unoptimized_csv))[0]
    opt_base = os.path.splitext(os.path.basename(optimized_csv))[0]
    output_file = f"boxplots_{unopt_base}_vs_{opt_base}.png"
    
    # Save with higher DPI for report quality
    plt.savefig(output_file, dpi=400, bbox_inches='tight')
    print(f"Report-ready comparative boxplots saved as {output_file}")
    
    # Display the plot
    plt.show()

# Update these paths with your actual CSV file paths
unoptimized_csv_path = "../../backend/logs/resource_stats_raspberry_20250412_052446.csv"
optimized_csv_path = "../../backend/logs/resource_stats_raspberry_20250412_053534.csv"

# Call the function to generate the comparative boxplots
plot_comparative_boxplots(unoptimized_csv_path, optimized_csv_path)