#!/usr/bin/env python3
"""
optimise_yolo.py - Convert YOLOv5n model to TFLite and benchmark performance

This script:
1. Loads a pre-trained YOLOv5n model
2. Converts it to TensorFlow Lite format
3. Benchmarks and compares inference speed and resource utilization
4. Visualizes the detection results and performance metrics
"""

import os
import time
import argparse
import psutil
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from PIL import Image
import gc

# For TFLite conversion
try:
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")
except ImportError:
    print("TensorFlow not installed. TFLite conversion won't be available.")
    tf = None

def load_yolo_model(model_path=None):
    """Load YOLOv5n model from local path or download from torch hub"""
    print("Loading YOLOv5n model...")
    
    if model_path and os.path.exists(model_path):
        # Load from local path
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
    else:
        # Download from torch hub
        model = torch.hub.load('ultralytics/yolov5', 'yolov5n')
    
    # Filter to detect only people (class 0)
    model.classes = [0]  # Person class only
    return model

def convert_to_tflite(model, input_shape=(640, 640), output_path="yolov5n_model.tflite"):
    """Convert PyTorch YOLOv5 model to TFLite format"""
    if tf is None:
        print("TensorFlow not available, skipping conversion")
        return None
    
    print(f"Converting YOLOv5 model to TFLite with input shape {input_shape}...")
    
    # First convert to ONNX
    onnx_path = "temp_yolo.onnx"
    batch_size = 1
    
    dummy_input = torch.zeros(batch_size, 3, *input_shape)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        verbose=False,
        opset_version=12,
        input_names=['images'],
        output_names=['output'],
        dynamic_axes={'images': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    
    # Convert ONNX to TensorFlow SavedModel
    os.system(f"python -m tf2onnx.convert --opset 12 --inputs-as-nchw images --input {onnx_path} --output temp_yolo_model")
    
    # Load the SavedModel
    converter = tf.lite.TFLiteConverter.from_saved_model("temp_yolo_model")
    
    # Set optimization parameters
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Convert to TFLite model
    tflite_model = converter.convert()
    
    # Save the model
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"TFLite model saved to: {output_path}")
    
    # Clean up temp files
    if os.path.exists(onnx_path):
        os.remove(onnx_path)
    
    return output_path

def load_tflite_model(tflite_path):
    """Load TFLite model and print expected input shape"""
    if tf is None:
        print("TensorFlow not available, can't load TFLite model")
        return None, None, None
    
    # Load TFLite model and allocate tensors
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Print expected input shape
    if input_details:
        input_shape = input_details[0]['shape']
        print(f"TFLite model expects input shape: {input_shape}")
    
    return interpreter, input_details, output_details

def preprocess_image(image_path, input_shape_pytorch=(640, 640), input_shape_tflite=(416, 416)):
    """Preprocess image for both PyTorch and TFLite models with different input shapes"""
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Create copies for both models
    img_torch = img.copy()  # For PyTorch
    
    # For TFLite - resize to the expected shape
    img_tf = cv2.resize(img.copy(), input_shape_tflite)
    img_tf = cv2.cvtColor(img_tf, cv2.COLOR_BGR2RGB)
    img_tf = img_tf.astype(np.float32) / 255.0
    img_tf = np.expand_dims(img_tf, axis=0)  # Add batch dimension
    
    return img, img_torch, img_tf

def benchmark_pytorch_model(model, img, num_runs=50):
    """Benchmark PyTorch model using the same approach as the MediaPipe benchmark"""
    process = psutil.Process(os.getpid())
    results = {
        "inference_times": [],
        "cpu_usage": [],
        "memory_usage": []
    }
    
    # Force garbage collection
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Warmup
    for _ in range(5):
        model(img)
    
    # Benchmark
    baseline_memory = process.memory_info().rss / (1024 * 1024)  # MB
    
    for _ in range(num_runs):
        # Measure CPU before inference
        cpu_before = process.cpu_percent()
        
        # Run inference and time it
        start_time = time.time()
        _ = model(img)  # Capture output to ensure operation completes
        torch.cuda.synchronize() if torch.cuda.is_available() else None  # Wait for GPU operations
        inference_time = time.time() - start_time
        
        # Measure CPU and memory after inference
        time.sleep(0.01)  # Small delay for CPU usage to register
        cpu_after = process.cpu_percent()
        memory_after = process.memory_info().rss / (1024 * 1024)
        
        # Store results
        results["inference_times"].append(inference_time)
        results["cpu_usage"].append(cpu_after)
        results["memory_usage"].append(memory_after - baseline_memory)
    
    # Calculate averages
    avg_time = np.mean(results["inference_times"]) * 1000  # ms
    avg_cpu = np.mean([x for x in results["cpu_usage"] if x >= 0])
    avg_memory = np.mean([x for x in results["memory_usage"] if x >= 0])
    
    results["avg_inference_ms"] = avg_time
    results["avg_cpu_percent"] = avg_cpu
    results["avg_memory_mb"] = avg_memory
    results["fps"] = 1000 / avg_time
    
    return results

def benchmark_tflite_model(interpreter, input_details, output_details, img_tf, num_runs=50):
    """Benchmark TFLite model using the same approach as the MediaPipe benchmark"""
    process = psutil.Process(os.getpid())
    results = {
        "inference_times": [],
        "cpu_usage": [],
        "memory_usage": []
    }
    
    # Force garbage collection
    gc.collect()
    
    # Warmup
    for _ in range(5):
        interpreter.set_tensor(input_details[0]['index'], img_tf)
        interpreter.invoke()
        interpreter.get_tensor(output_details[0]['index'])
    
    # Benchmark
    baseline_memory = process.memory_info().rss / (1024 * 1024)  # MB
    
    for _ in range(num_runs):
        # Measure CPU before inference
        cpu_before = process.cpu_percent()
        
        # Run inference and time it
        start_time = time.time()
        interpreter.set_tensor(input_details[0]['index'], img_tf)
        interpreter.invoke()
        interpreter.get_tensor(output_details[0]['index'])
        inference_time = time.time() - start_time
        
        # Measure CPU and memory after inference
        time.sleep(0.01)  # Small delay for CPU usage to register
        cpu_after = process.cpu_percent()
        memory_after = process.memory_info().rss / (1024 * 1024)
        
        # Store results
        results["inference_times"].append(inference_time)
        results["cpu_usage"].append(cpu_after)  # Use only the after measurement
        results["memory_usage"].append(memory_after - baseline_memory)
    
    # Calculate averages
    avg_time = np.mean(results["inference_times"]) * 1000  # ms
    avg_cpu = np.mean([x for x in results["cpu_usage"] if x >= 0])
    avg_memory = np.mean([x for x in results["memory_usage"] if x >= 0])
    
    results["avg_inference_ms"] = avg_time
    results["avg_cpu_percent"] = avg_cpu
    results["avg_memory_mb"] = avg_memory
    results["fps"] = 1000 / avg_time
    
    return results

def process_results(pt_path, tflite_path, pytorch_results, tflite_results, output_path="yolo_optimization_results.png"):
    """Process and visualize benchmark results"""
    # Print comparison
    print("\n===== YOLOv5n Optimization Results =====")
    print(f"{'Metric':<25} {'PyTorch':<15} {'TFLite':<15} {'Improvement':<15}")
    print("-" * 70)
    
    # Inference time
    pt_time = pytorch_results["avg_inference_ms"]
    tf_time = tflite_results["avg_inference_ms"]
    time_imp = (pt_time - tf_time) / pt_time * 100
    print(f"{'Inference Time (ms)':<25} {pt_time:<15.2f} {tf_time:<15.2f} {time_imp:>+.2f}%")
    
    # CPU usage
    pt_cpu = pytorch_results["avg_cpu_percent"]
    tf_cpu = tflite_results["avg_cpu_percent"]
    cpu_imp = (pt_cpu - tf_cpu) / pt_cpu * 100
    print(f"{'CPU Usage (%)':<25} {pt_cpu:<15.2f} {tf_cpu:<15.2f} {cpu_imp:>+.2f}%")
    
    # Memory usage - use peak memory if average is zero
    pt_mem = pytorch_results["avg_memory_mb"]
    tf_mem = tflite_results["avg_memory_mb"]
    
    # If average memory is zero, try using peak memory instead
    if pt_mem == 0 or tf_mem == 0:
        pt_mem = pytorch_results.get("peak_memory_mb", 0)
        tf_mem = tflite_results.get("peak_memory_mb", 0)
        print("Using peak memory measurements instead of average")
    
    # Avoid division by zero
    if pt_mem > 0 and tf_mem > 0:
        mem_imp = (pt_mem - tf_mem) / pt_mem * 100
        print(f"{'Memory Usage (MB)':<25} {pt_mem:<15.2f} {tf_mem:<15.2f} {mem_imp:>+.2f}%")
    else:
        print(f"{'Memory Usage (MB)':<25} {pt_mem:<15.2f} {tf_mem:<15.2f} {'N/A':<15}")
    
    # FPS
    pt_fps = pytorch_results["fps"]
    tf_fps = tflite_results["fps"]
    fps_imp = (tf_fps - pt_fps) / pt_fps * 100
    print(f"{'Frames Per Second (FPS)':<25} {pt_fps:<15.2f} {tf_fps:<15.2f} {fps_imp:>+.2f}%")
    
    # Calculate model size
    pt_size = os.path.getsize(pt_path) / (1024 * 1024) if os.path.exists(pt_path) else "N/A"
    tf_size = os.path.getsize(tflite_path) / (1024 * 1024) if os.path.exists(tflite_path) else "N/A"
    if isinstance(pt_size, float) and isinstance(tf_size, float):
        size_imp = (pt_size - tf_size) / pt_size * 100
        print(f"{'Model Size (MB)':<25} {pt_size:<15.2f} {tf_size:<15.2f} {size_imp:>+.2f}%")
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    plt.rcParams.update({'font.size': 14})  # Increase font size
    
    # Create grouped bar chart
    metrics = ['Inference Time (ms)', 'CPU Usage (%)', 'Memory Usage (MB)', 'FPS']
    pt_values = [pt_time, pt_cpu, pt_mem, pt_fps]
    tf_values = [tf_time, tf_cpu, tf_mem, tf_fps]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 8))
    bars1 = ax.bar(x - width/2, pt_values, width, label='PyTorch YOLOv5n', color='#3498db')
    bars2 = ax.bar(x + width/2, tf_values, width, label='TFLite YOLOv5n', color='#2ecc71')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height * 1.02,
                   f'{height:.1f}', ha='center', va='bottom', fontsize=12)
    
    # Set y-axis limit
    y_max = max([max(pt_values), max(tf_values)]) * 1.3
    ax.set_ylim(0, y_max)
    
    # Customize chart
    ax.set_xlabel('Performance Metrics', fontsize=16, fontweight='bold')
    ax.set_ylabel('Value', fontsize=16, fontweight='bold')
    ax.set_title('YOLOv5n PyTorch vs. TFLite Comparison', fontsize=18, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=14)
    ax.legend(fontsize=14)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # # Add speedup text
    # speedup = pt_time / tf_time if tf_time > 0 else 0
    # fig.text(0.5, 0.02, f'Overall Speedup Factor: {speedup:.2f}x', ha='center', 
    #          fontsize=16, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(output_path, dpi=300)
    print(f"Comparison chart saved to: {output_path}")
    
    plt.close()

def visualize_detections(image_path, tflite_path, output_path="yolo_detection_comparison.png"):
    """Compare and visualize detections from both models"""
    # Load and process image
    orig_img, img_torch, img_tf = preprocess_image(image_path)
    
    # Load models
    pytorch_model = load_yolo_model()
    interpreter, input_details, output_details = load_tflite_model(tflite_path)
    
    # Get PyTorch predictions
    pytorch_results = pytorch_model(img_torch)
    pytorch_detections = pytorch_results.xyxy[0].cpu().numpy()  # person detections
    
    # Get TFLite predictions
    interpreter.set_tensor(input_details[0]['index'], img_tf)
    interpreter.invoke()
    tflite_output = interpreter.get_tensor(output_details[0]['index'])
    
    # Process TFLite output (implementation depends on the exact output format)
    # This is a simplified version - you may need to adjust based on your model's output
    tflite_detections = []
    if len(output_details) == 1:
        # Assuming output is [batch, num_detections, 7] where each detection is
        # [x1, y1, x2, y2, confidence, class, ...]
        detections = tflite_output[0]
        for detection in detections:
            if detection[5] == 0:  # person class
                if detection[4] > 0.25:  # confidence threshold
                    x1, y1, x2, y2 = detection[0:4]
                    conf = detection[4]
                    tflite_detections.append([x1, y1, x2, y2, conf, 0])
    
    # Create visualization
    plt.figure(figsize=(12, 6))
    
    # PyTorch detections
    plt.subplot(1, 2, 1)
    plt_img = cv2.cvtColor(img_torch, cv2.COLOR_BGR2RGB)
    plt.imshow(plt_img)
    
    for detection in pytorch_detections:
        x1, y1, x2, y2, conf, class_id = detection
        if class_id == 0:  # person
            plt.gca().add_patch(plt.Rectangle(
                (x1, y1), x2-x1, y2-y1, 
                fill=False, edgecolor='red', linewidth=2
            ))
            plt.text(
                x1, y1-10, f"Person: {conf:.2f}", 
                color='white', fontsize=10,
                bbox=dict(facecolor='red', alpha=0.5)
            )
    
    plt.title("PyTorch YOLOv5n Detections")
    plt.axis('off')
    
    # TFLite detections
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(img_torch, cv2.COLOR_BGR2RGB))
    
    for detection in tflite_detections:
        x1, y1, x2, y2, conf, class_id = detection
        if class_id == 0:  # person
            plt.gca().add_patch(plt.Rectangle(
                (x1, y1), x2-x1, y2-y1, 
                fill=False, edgecolor='green', linewidth=2
            ))
            plt.text(
                x1, y1-10, f"Person: {conf:.2f}", 
                color='white', fontsize=10,
                bbox=dict(facecolor='green', alpha=0.5)
            )
    
    plt.title("TFLite YOLOv5n Detections")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Detection comparison saved to: {output_path}")
    
    plt.close()

def main():
    # Set parameters directly in the code
    image_path = "../../inference/images/crowded_2.jpg"  
    pt_path = "../../models/yolo-human-detection/yolov5n.pt"      
    tflite_path = "../../models/yolo-human-detection/yolov5n-fp16.tflite"
    num_runs = 50                               # Number of benchmark runs
    force_convert = False                       # Whether to force TFLite conversion
    
    # 1. Load PyTorch model
    pytorch_model = load_yolo_model(pt_path)
    
    # 2. Load TFLite model and get input shape
    interpreter, input_details, output_details = load_tflite_model(tflite_path)
    if interpreter is None:
        print("Could not load TFLite model, exiting.")
        return
    
    # Get expected input shape from TFLite model
    tflite_input_shape = tuple(input_details[0]['shape'][1:3])  # Height, Width
    pytorch_input_shape = (640, 640)  # Default PyTorch shape
    
    # 3. Preprocess image with correct shapes
    _, img_torch, img_tf = preprocess_image(
        image_path, 
        input_shape_pytorch=pytorch_input_shape,
        input_shape_tflite=tflite_input_shape
    )
    
    # 4. Benchmark PyTorch model
    print(f"\nBenchmarking PyTorch YOLOv5n ({num_runs} runs)...")
    pytorch_results = benchmark_pytorch_model(pytorch_model, img_torch, num_runs=num_runs)
    
    # 5. Benchmark TFLite model
    print(f"\nBenchmarking TFLite YOLOv5n ({num_runs} runs)...")
    tflite_results = benchmark_tflite_model(interpreter, input_details, output_details, img_tf, num_runs=num_runs)
    
    # 6. Process and visualize results
    process_results(pt_path, tflite_path, pytorch_results, tflite_results)
    
    # 7. Visualize detections
    # visualize_detections(image_path)

if __name__ == "__main__":
    main()