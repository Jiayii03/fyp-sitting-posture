import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

def plot_resource_stats_boxplot(csv_file):
    """
    Create boxplots for system resource metrics from CSV.
    
    Args:
        csv_file (str): Path to the CSV file with resource statistics
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Define the metrics to plot
    metrics = ['cpu_percent', 'memory_percent', 'temperature', 
              'inference_fps', 'inference_latency_ms']
    
    # Define colors for each metric
    colors = {
        'cpu_percent': 'blue',
        'memory_percent': 'green',
        'temperature': 'red',
        'inference_fps': 'orange',
        'inference_latency_ms': 'purple'
    }
    
    # Check if temperature has valid data (not all zeros)
    if df['temperature'].max() == 0:
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
    
    # Create individual boxplots with appropriate scales
    fig, axes = plt.subplots(1, len(metrics), figsize=(4*len(metrics), 6), sharey=False)
    plt.suptitle('Resource Metrics Distribution', fontsize=16, y=0.98)
    
    # Create individual boxplots
    for i, metric in enumerate(metrics):
        # Get the current axes
        ax = axes[i]
        
        # Create the boxplot
        sns.boxplot(y=df[metric], ax=ax, color=colors[metric], width=0.6)
        
        # Add strip plot for data points (dots)
        sns.stripplot(y=df[metric], ax=ax, color='black', size=3, alpha=0.3)
        
        # Configure the axes
        ax.set_title(metric_labels[metric], fontsize=12)
        ax.set_ylabel('Value')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add some statistical information
        median = df[metric].median()
        mean = df[metric].mean()
        
        # Add horizontal lines for median and mean
        ax.axhline(median, color='black', linestyle='-', linewidth=1.5, alpha=0.7, label=f"Median: {median:.2f}")
        ax.axhline(mean, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=f"Mean: {mean:.2f}")
        
        # Add a legend
        ax.legend(fontsize=10)
        
        # Remove x-tick labels
        ax.set_xticklabels([])
        ax.set_xlabel('')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Create output filename for the boxplots
    base_name = os.path.splitext(os.path.basename(csv_file))[0]
    output_file = f"{base_name}_improved_boxplots.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Improved boxplots saved as {output_file}")
    
    # Display the plot
    plt.show()
    
    # Create a summary table
    print("\nMetric Statistics Summary:")
    print("=" * 80)
    print(f"{'Metric':<20} {'Min':>10} {'Mean':>10} {'Median':>10} {'Max':>10} {'Std Dev':>10}")
    print("-" * 80)
    
    for metric in metrics:
        min_val = df[metric].min()
        mean_val = df[metric].mean()
        median_val = df[metric].median()
        max_val = df[metric].max()
        std_val = df[metric].std()
        
        print(f"{metric_labels[metric]:<20} {min_val:>10.2f} {mean_val:>10.2f} {median_val:>10.2f} {max_val:>10.2f} {std_val:>10.2f}")

# Specify the path to your CSV file directly in the script
csv_file_path = "../../backend/logs/resource_stats_laptop_20250412_020322.csv"

# Call the function to generate the boxplots
plot_resource_stats_boxplot(csv_file_path)