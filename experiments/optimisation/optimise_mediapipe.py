import mediapipe as mp
import numpy as np
import cv2
import time
import os
import psutil
import matplotlib.pyplot as plt
import gc
import torch


def benchmark_mediapipe_pose(image_path, num_runs=50, model_complexity=1):
    """Benchmark MediaPipe Pose performance and resource utilization"""
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    process = psutil.Process(os.getpid())

    # Dictionary to store all benchmark results
    results = {
        "standard": {"inference_times": [], "cpu_usage": [], "memory_usage": []},
        "lite": {"inference_times": [], "cpu_usage": [], "memory_usage": []}
    }

    # Test standard model
    print(f"\nBenchmarking standard model (complexity={model_complexity})...")
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=True,
        model_complexity=model_complexity,
        min_detection_confidence=0.5
    )

    # Warm up
    pose.process(image_rgb)
    gc.collect()

    # Benchmark standard model
    baseline_memory = process.memory_info().rss / (1024 * 1024)
    for _ in range(num_runs):
        # Measure CPU before
        cpu_before = process.cpu_percent()
        memory_before = process.memory_info().rss / (1024 * 1024)

        # Run inference
        start_time = time.time()
        pose.process(image_rgb)
        inference_time = time.time() - start_time

        # Measure CPU after and store results
        # Small delay for CPU usage to register
        time.sleep(0.01)
        cpu_after = process.cpu_percent()
        memory_after = process.memory_info().rss / (1024 * 1024)

        results["standard"]["inference_times"].append(inference_time)
        results["standard"]["cpu_usage"].append(cpu_after)
        results["standard"]["memory_usage"].append(
            memory_after - baseline_memory)

    # Clean up
    pose.close()

    # Test lite model
    print("Benchmarking lite model (complexity=0)...")
    pose_lite = mp_pose.Pose(
        static_image_mode=True,
        model_complexity=0,
        min_detection_confidence=0.5
    )

    # Warm up
    pose_lite.process(image_rgb)
    gc.collect()

    # Benchmark lite model
    baseline_memory = process.memory_info().rss / (1024 * 1024)
    for _ in range(num_runs):
        # Measure CPU before
        cpu_before = process.cpu_percent()
        memory_before = process.memory_info().rss / (1024 * 1024)

        # Run inference
        start_time = time.time()
        pose_lite.process(image_rgb)
        inference_time = time.time() - start_time

        # Measure CPU after and store results
        time.sleep(0.01)
        cpu_after = process.cpu_percent()
        memory_after = process.memory_info().rss / (1024 * 1024)

        results["lite"]["inference_times"].append(inference_time)
        results["lite"]["cpu_usage"].append(cpu_after)
        results["lite"]["memory_usage"].append(memory_after - baseline_memory)

    # Clean up
    pose_lite.close()

    # Calculate and print summary statistics
    for model_type in ["standard", "lite"]:
        avg_time = np.mean(results[model_type]["inference_times"]) * 1000  # ms
        avg_cpu = np.mean(
            [x for x in results[model_type]["cpu_usage"] if x >= 0])
        avg_memory = np.mean(
            [x for x in results[model_type]["memory_usage"] if x >= 0])

        results[model_type]["avg_inference_ms"] = avg_time
        results[model_type]["avg_cpu_percent"] = avg_cpu
        results[model_type]["avg_memory_mb"] = avg_memory

    # Print results summary
    std_time = results["standard"]["avg_inference_ms"]
    lite_time = results["lite"]["avg_inference_ms"]
    speedup = std_time / lite_time if lite_time > 0 else 0

    print("\n===== MediaPipe Pose Benchmark Summary =====")
    print(f"{'Metric':<25} {'Standard':<15} {'Lite':<15} {'Improvement':<15}")
    print("-" * 70)

    # Inference time comparison
    print(f"{'Inference Time (ms)':<25} {std_time:<15.2f} {lite_time:<15.2f} {(std_time-lite_time)/std_time*100:>+.2f}%")

    # CPU usage comparison
    std_cpu = results["standard"]["avg_cpu_percent"]
    lite_cpu = results["lite"]["avg_cpu_percent"]
    print(f"{'CPU Usage (%)':<25} {std_cpu:<15.2f} {lite_cpu:<15.2f} {(std_cpu-lite_cpu)/std_cpu*100:>+.2f}%")

    # Memory usage comparison
    std_mem = results["standard"]["avg_memory_mb"]
    lite_mem = results["lite"]["avg_memory_mb"]
    print(f"{'Memory Usage (MB)':<25} {std_mem:<15.2f} {lite_mem:<15.2f} {(std_mem-lite_mem)/std_mem*100:>+.2f}%")

    # FPS comparison
    std_fps = 1000 / std_time
    lite_fps = 1000 / lite_time
    print(f"{'Frames Per Second (FPS)':<25} {std_fps:<15.2f} {lite_fps:<15.2f} {(lite_fps-std_fps)/std_fps*100:>+.2f}%")
    print(f"Speedup factor: {speedup:.2f}x")

    # Plot comparison figures
    plot_benchmark_results(results)

    return results


def plot_benchmark_results(results):
    """Create a visualization without red improvement labels"""
    # Calculate metrics
    std_time = results["standard"]["avg_inference_ms"]
    lite_time = results["lite"]["avg_inference_ms"]

    std_cpu = results["standard"]["avg_cpu_percent"]
    lite_cpu = results["lite"]["avg_cpu_percent"]

    std_mem = results["standard"]["avg_memory_mb"]
    lite_mem = results["lite"]["avg_memory_mb"]

    std_fps = 1000 / std_time
    lite_fps = 1000 / lite_time

    # Create figure with increased font size
    plt.rcParams.update({'font.size': 14})  # Increase all font sizes

    # Create grouped bar chart
    metrics = ['Inference Time (ms)', 'CPU Usage (%)',
               'Memory Usage (MB)', 'FPS']
    std_values = [std_time, std_cpu, std_mem, std_fps]
    lite_values = [lite_time, lite_cpu, lite_mem, lite_fps]

    # Set up bar positions
    x = np.arange(len(metrics))
    width = 0.35

    # Create bars
    fig, ax = plt.subplots(figsize=(12, 8))
    bars1 = ax.bar(x - width/2, std_values, width,
                   label='Standard Model', color='#3498db')
    bars2 = ax.bar(x + width/2, lite_values, width,
                   label='Lite Model', color='#2ecc71')

    # Add value labels on top of bars with enough padding
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height * 1.02,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=12)

    # Set y-axis limit with enough headroom for labels
    y_max = max([max(std_values), max(lite_values)]) * 1.3
    ax.set_ylim(0, y_max)

    # Customize chart
    ax.set_xlabel('Performance Metrics', fontsize=16, fontweight='bold')
    ax.set_ylabel('Value', fontsize=16, fontweight='bold')
    ax.set_title('MediaPipe Pose Standard vs. Lite Model Comparison',
                 fontsize=18, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=14)
    ax.legend(fontsize=14)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Add speedup factor as text annotation
    speedup = std_time / lite_time if lite_time > 0 else 0
    fig.text(0.5, 0.02, f'Overall Speedup Factor: {speedup:.2f}x', ha='center',
             fontsize=16, fontweight='bold')

    # Adjust layout to make room for the speedup text
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig("mediapipe_benchmark_combined.png", dpi=300)
    print("Combined benchmark figure saved to mediapipe_benchmark_combined.png")
    plt.close()


def visualize_pose_estimation(image_path, model_complexity=1):
    """Visualize the difference in pose estimation between standard and lite models"""
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Initialize MediaPipe pose solutions
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    # Process with standard model
    pose_standard = mp_pose.Pose(
        static_image_mode=True,
        model_complexity=model_complexity,
        min_detection_confidence=0.5
    )

    # Process with lite model
    pose_lite = mp_pose.Pose(
        static_image_mode=True,
        model_complexity=0,
        min_detection_confidence=0.5
    )

    # Get results
    standard_results = pose_standard.process(image_rgb)
    lite_results = pose_lite.process(image_rgb)

    # Create visualization
    plt.figure(figsize=(16, 8))

    # Standard model results
    standard_image = image_rgb.copy()
    if standard_results.pose_landmarks:
        mp_drawing.draw_landmarks(
            standard_image,
            standard_results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            # Change to (255, 0, 0) for red points
            mp_drawing.DrawingSpec(
                color=(255, 0, 0), thickness=2, circle_radius=2),
            # Change to (255, 255, 255) for white connections
            mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
        )
    plt.subplot(1, 2, 1)
    plt.imshow(standard_image)
    plt.title(f"Standard Model (complexity={model_complexity})")
    plt.axis('off')

    # Lite model results
    lite_image = image_rgb.copy()
    if lite_results.pose_landmarks:
        mp_drawing.draw_landmarks(
            lite_image,
            lite_results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            # Change to (255, 0, 0) for red points
            mp_drawing.DrawingSpec(
                color=(255, 0, 0), thickness=2, circle_radius=2),
            # Change to (255, 255, 255) for white connections
            mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
        )
    plt.subplot(1, 2, 2)
    plt.imshow(lite_image)
    plt.title("Lite Model (complexity=0)")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig("mediapipe_pose_comparison.png")
    print("Pose estimation comparison saved to mediapipe_pose_comparison.png")
    plt.close()

    # Clean up
    pose_standard.close()
    pose_lite.close()


if __name__ == "__main__":
    # Example usage
    # Replace with your image path
    image_path = "../../../datasets/raw/crossed_legs/038.jpg"
    dataset_path = "../../../datasets/vectors/xy_filtered_keypoints_vectors_mediapipe.csv"  # Your dataset

    # Run performance benchmark
    benchmark_results = benchmark_mediapipe_pose(
        image_path, num_runs=50, model_complexity=2)

    # Visualize pose estimation differences
    visualize_pose_estimation(image_path, model_complexity=2)
