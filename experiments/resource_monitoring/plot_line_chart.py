import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import os
import numpy as np

def plot_resource_stats(csv_file):
    """
    Create a single time-series plot for all system resource statistics from CSV,
    with duration on the x-axis instead of datetime.

    Args:
        csv_file (str): Path to the CSV file with resource statistics
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Calculate duration in seconds from the first timestamp
    start_time = df['timestamp'].min()
    df['duration_seconds'] = (df['timestamp'] - start_time).dt.total_seconds()

    # Set up the figure
    fig, ax1 = plt.subplots(figsize=(14, 8))
    plt.title('System Resource Monitoring - Combined Metrics', fontsize=16)

    # Colors for each metric
    colors = {
        'cpu_percent': 'blue',
        'memory_percent': 'green',
        'temperature': 'red',
        'inference_fps': 'orange',
        'inference_latency_ms': 'purple'
    }

    # Create a secondary y-axis for latency which is often on a different scale
    ax2 = ax1.twinx()

    # Plot CPU, memory, temperature, and FPS on the primary axis
    ax1.plot(df['duration_seconds'], df['cpu_percent'],
             color=colors['cpu_percent'], linewidth=2, label='CPU Usage (%)')
    ax1.plot(df['duration_seconds'], df['memory_percent'],
             color=colors['memory_percent'], linewidth=2, label='Memory Usage (%)')

    # Only plot temperature if it has non-zero values
    if 'temperature' in df.columns and df['temperature'].max() > 0:
        ax1.plot(df['duration_seconds'], df['temperature'],
                 color=colors['temperature'], linewidth=2, label='Temperature (°C)')

    ax1.plot(df['duration_seconds'], df['inference_fps'],
             color=colors['inference_fps'], linewidth=2, label='Inference FPS')

    # Plot latency on the secondary axis
    ax2.plot(df['duration_seconds'], df['inference_latency_ms'], color=colors['inference_latency_ms'], linewidth=2,
             label='Inference Latency (ms)')

    # Set labels and grid
    ax1.set_xlabel('Duration (minutes:seconds)', fontsize=12)
    ax1.set_ylabel('CPU / Memory / Temperature / FPS', fontsize=12)
    ax2.set_ylabel('Latency (ms)', fontsize=12)

    ax1.grid(True, linestyle='--', alpha=0.7)

    # Format the x-axis with appropriate duration labels (minutes:seconds)
    max_duration = df['duration_seconds'].max()

    # Create a formatter function to convert seconds to MM:SS format
    def format_time(x, pos):
        minutes = int(x // 60)
        seconds = int(x % 60)
        return f"{minutes}:{seconds:02d}"

    # Set major ticks every minute
    major_tick_interval = 60  # seconds
    ax1.set_xticks(np.arange(0, max_duration +
                   major_tick_interval, major_tick_interval))
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(format_time))

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc='upper left', fontsize=10)

    # Add a timestamp to the figure
    plt.figtext(0.02, 0.02, f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
                fontsize=8, ha='left')

    # Adjust layout and save the figure
    plt.tight_layout()

    # Create output filename based on input filename
    base_name = os.path.splitext(os.path.basename(csv_file))[0]
    output_file = f"{base_name}_duration_plot.png"

    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Duration plot saved as {output_file}")

    # Display the plot
    plt.show()


# Specify the path to your CSV file directly in the script
csv_file_path = "../../backend/logs/resource_stats_raspberry_20250419_091055.csv"

# Call the function to generate the plot
plot_resource_stats(csv_file_path)
