import torch
import torch.quantization
import torch.nn as nn
import os
import time
import psutil
import gc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder

# Load your trained model
def load_and_quantize_model_dynamic(model_path, save_dir):
    # Load the original model
    input_size = 20  # Input size
    num_classes = 4  # Number of posture classes
    
    # Import MLP Class
    import sys
    sys.path.append(os.path.dirname(model_path))
    from model import MLP
    
    # Create model instance
    model = MLP(input_size=input_size, num_classes=num_classes)
    
    # Load trained weights
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Apply dynamic quantization to the model
    # This converts weights from 32-bit floating-point to 8-bit integers
    quantized_model = torch.quantization.quantize_dynamic(
        model,                   # Original model
        {torch.nn.Linear},       # Quantize only Linear layers
        dtype=torch.qint8        # Use 8-bit integers instead of 32-bit floats
    )
    
    # Save the quantized model
    model_name = model_path.split("/")[-1].split(".")[0]
    quantized_model_name = f"{model_name}_d_quantized.pth"
    quantized_model_path = os.path.join(save_dir, quantized_model_name)
    torch.save(quantized_model.state_dict(), quantized_model_path)
    
    print(f"Original model size: {get_model_size(model_path):.2f} MB")
    print(f"Quantized model size: {get_model_size(quantized_model_path):.2f} MB")
    print(f"Quantized model saved to: {quantized_model_path}")
    
    return model, quantized_model

def convert_to_half_precision(model_path, save_dir):
    """Convert model to half precision (float16) for faster inference"""
    # Import your original MLP class and load model
    import sys
    sys.path.append(os.path.dirname(model_path))
    from model import MLP
    import copy
    
    # Load original model
    original_model = MLP(input_size=20, num_classes=4)
    original_model.load_state_dict(torch.load(model_path, weights_only=True))
    original_model.eval()
    
    # Create a DEEP COPY of the model before converting to half precision
    model_for_half = copy.deepcopy(original_model)
    
    # Create a wrapper class that handles type conversion
    class HalfPrecisionWrapper(nn.Module):
        def __init__(self, model):
            super(HalfPrecisionWrapper, self).__init__()
            self.model = model
            # Convert model to half precision
            self.model.half()
            
        def forward(self, x):
            # Convert input to half precision
            x = x.half()
            # Run model
            output = self.model(x)
            # Convert output back to float for compatibility
            return output.float()
    
    # Create wrapped model with the copy, leaving original_model untouched
    half_model = HalfPrecisionWrapper(model_for_half)
    
    # Save the half precision model
    model_name = os.path.basename(model_path).split(".")[0]
    half_model_path = os.path.join(save_dir, f"{model_name}_fp16.pth")
    torch.save(half_model.state_dict(), half_model_path)
    
    print(f"Original model size: {get_model_size(model_path):.2f} MB")
    print(f"Half precision model size: {get_model_size(half_model_path):.2f} MB")
    print(f"Half precision model saved to: {half_model_path}")
    
    return original_model, half_model

def load_and_quantize_model_static(model_path, save_dir, calibration_loader=None):
    """Load model and apply post-training static quantization matching your specific MLP architecture"""
    import sys
    import copy
    sys.path.append(os.path.dirname(model_path))
    from model import MLP
    
    # Load original model
    original_model = MLP(input_size=20, num_classes=4)
    original_model.load_state_dict(torch.load(model_path, weights_only=True))
    original_model.eval()
    
    # Create a copy for quantization
    model_to_quantize = copy.deepcopy(original_model)
    
    # Create a wrapper matching your specific architecture
    class QuantWrapper(nn.Module):
        def __init__(self, model):
            super(QuantWrapper, self).__init__()
            
            # Extract all layers from the sequential model
            seq_model = model.model
            
            # First block: Linear -> BatchNorm -> LeakyReLU -> Dropout
            self.linear1 = seq_model[0]
            self.bn1 = seq_model[1]
            self.leaky1 = seq_model[2]
            self.dropout1 = seq_model[3]
            
            # Second block: Linear -> BatchNorm -> LeakyReLU -> Dropout
            self.linear2 = seq_model[4]
            self.bn2 = seq_model[5]
            self.leaky2 = seq_model[6]
            self.dropout2 = seq_model[7]
            
            # Third block: Linear -> BatchNorm -> LeakyReLU
            self.linear3 = seq_model[8]
            self.bn3 = seq_model[9]
            self.leaky3 = seq_model[10]
            
            # Final layer
            self.linear4 = seq_model[11]
            
            # Add QuantStub and DeQuantStub
            self.quant = torch.quantization.QuantStub()
            self.dequant = torch.quantization.DeQuantStub()
        
        def forward(self, x):
            x = self.quant(x)
            
            # First block
            x = self.linear1(x)
            x = self.bn1(x)
            x = self.leaky1(x)
            x = self.dropout1(x)
            
            # Second block
            x = self.linear2(x)
            x = self.bn2(x)
            x = self.leaky2(x)
            x = self.dropout2(x)
            
            # Third block
            x = self.linear3(x)
            x = self.bn3(x)
            x = self.leaky3(x)
            
            # Final layer
            x = self.linear4(x)
            
            x = self.dequant(x)
            return x
        
        def fuse_modules(self):
            # Fuse batchnorm into preceding linear layers when possible
            # This greatly improves quantization performance
            torch.quantization.fuse_modules(self, ['linear1', 'bn1'], inplace=True)
            torch.quantization.fuse_modules(self, ['linear2', 'bn2'], inplace=True)
            torch.quantization.fuse_modules(self, ['linear3', 'bn3'], inplace=True)
    
    # Wrap model
    quant_model = QuantWrapper(model_to_quantize)
    quant_model.eval()  # Important for BatchNorm fusion
    
    # Fuse modules for better quantization
    quant_model.fuse_modules()
    
    # Set qconfig for the entire model
    quant_model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    
    # Prepare model for quantization
    prepared_model = torch.quantization.prepare(quant_model)
    
    # Calibrate with representative data
    print("Calibrating model with representative data...")
    with torch.no_grad():
        if calibration_loader is not None:
            for inputs, _ in calibration_loader:
                prepared_model(inputs)
        else:
            print("No calibration data provided, using random data...")
            for _ in range(100):
                random_input = torch.randn(1, 20)
                prepared_model(random_input)
    
    # Convert to quantized model
    quantized_model = torch.quantization.convert(prepared_model)
    
    # Save the quantized model
    model_name = os.path.basename(model_path).split(".")[0]
    quantized_model_path = os.path.join(save_dir, f"{model_name}_static_quantized.pth")
    torch.save(quantized_model.state_dict(), quantized_model_path)
    
    print(f"Original model size: {get_model_size(model_path):.2f} MB")
    print(f"Quantized model size: {get_model_size(quantized_model_path):.2f} MB")
    print(f"Quantized model saved to: {quantized_model_path}")
    
    return original_model, quantized_model

def get_model_size(model_path):
    """Get the size of a model file in MB"""
    return os.path.getsize(model_path) / (1024 * 1024)

# Compare inference speed
def compare_inference_speed(original_model, quantized_model, input_size=20, num_runs=100, batch_size=16):
    # Create random input tensor for testing
    test_input = torch.randn(batch_size, input_size)
    
    # Warm up
    for _ in range(10):
        with torch.no_grad():
            original_model(test_input)
            quantized_model(test_input)
    
    # Measure original model inference time
    start_time = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            original_model(test_input)
    original_time = time.time() - start_time
    
    # Measure quantized model inference time
    start_time = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            quantized_model(test_input)
    quantized_time = time.time() - start_time
    
    # Print results
    print(f"\nInference time comparison ({num_runs} runs):")
    print(f"Original model: {original_time*1000:.2f} ms total, {original_time*1000/num_runs:.4f} ms per inference")
    print(f"Quantized model: {quantized_time*1000:.2f} ms total, {quantized_time*1000/num_runs:.4f} ms per inference")
    print(f"Speedup: {original_time/quantized_time:.2f}x")

# Compare model accuracy 
def compare_accuracy(original_model, quantized_model, test_loader, device='cpu'):
    original_model.to(device)
    quantized_model.to(device)
    
    models = {"Original": original_model, "Quantized": quantized_model}
    
    for name, model in models.items():
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f"{name} model accuracy: {accuracy:.2f}%")
        
def get_test_dataloader(csv_path, scaler_mean_path, scaler_scale_path, batch_size=32):
    """Create a test dataloader from a CSV dataset"""
    
    class PostureDataset(Dataset):
        def __init__(self, data, scaler_mean, scaler_scale):
            from sklearn.preprocessing import LabelEncoder
            
            # Extract features and labels
            self.X = data.drop(columns=['class']).values  # Features
            
            # Convert string labels to numeric using LabelEncoder
            label_encoder = LabelEncoder()
            self.y = label_encoder.fit_transform(data['class'].values)  # Transform string labels to integers
            
            # Load scaler parameters
            self.scaler_mean = scaler_mean
            self.scaler_scale = scaler_scale
            
            # Apply normalization
            self.X = (self.X - self.scaler_mean) / self.scaler_scale
            
            self.X = torch.tensor(self.X, dtype=torch.float32)
            self.y = torch.tensor(self.y, dtype=torch.long)
        
        def __len__(self):
            return len(self.X)
        
        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]
    
    # Load data
    data = pd.read_csv(csv_path)
    
    # Load scaler parameters
    scaler_mean = np.load(scaler_mean_path)
    scaler_scale = np.load(scaler_scale_path)
    
    # Create dataset and dataloader
    test_dataset = PostureDataset(data, scaler_mean, scaler_scale)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return test_loader

# Create calibration data from your test dataset
def create_calibration_loader(csv_path, scaler_mean_path, scaler_scale_path, batch_size=8):
    """Create a small calibration dataloader from your CSV dataset"""
    # Get the full test loader first
    test_loader = get_test_dataloader(
        csv_path=csv_path,
        scaler_mean_path=scaler_mean_path,
        scaler_scale_path=scaler_scale_path,
        batch_size=batch_size
    )
    
    # Get the underlying dataset from the loader
    test_dataset = test_loader.dataset
    
    # Create a subset with a limited number of samples (max 100)
    num_calibration_samples = min(100, len(test_dataset))
    calibration_dataset = torch.utils.data.Subset(test_dataset, range(num_calibration_samples))
    
    # Create dataloader from the subset
    calibration_loader = DataLoader(
        calibration_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    print(f"Created calibration loader with {num_calibration_samples} samples")
    
    return calibration_loader

def compare_resource_utilization(original_model, quantized_model, input_size=20, num_warmup=10, num_runs=100, batch_size=16):
    """Compare CPU and memory usage between original and quantized models"""
    # Create random input tensor for testing
    test_input = torch.randn(batch_size, input_size)
    
    # Dictionary to store results
    results = {
        "original": {"inference_times": [], "cpu_usage": [], "memory_usage": []},
        "quantized": {"inference_times": [], "cpu_usage": [], "memory_usage": []}
    }
    
    process = psutil.Process(os.getpid())
    
    # Test original model
    print("\nTesting original model resource usage...")
    # Force garbage collection before testing
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Warm up
    for _ in range(num_warmup):
        with torch.no_grad():
            original_model(test_input)
    
    # Measure resource usage
    for _ in range(num_runs):
        # Record CPU usage before inference
        cpu_before = process.cpu_percent()
        # Record memory usage before inference
        memory_before = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Perform inference and measure time
        start_time = time.time()
        with torch.no_grad():
            original_model(test_input)
        inference_time = time.time() - start_time
        
        # Wait briefly for CPU measurement to register
        time.sleep(0.01)
        
        # Record CPU and memory after inference
        cpu_after = process.cpu_percent()
        memory_after = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Record results
        results["original"]["inference_times"].append(inference_time)
        results["original"]["cpu_usage"].append(cpu_after - cpu_before)
        results["original"]["memory_usage"].append(memory_after - memory_before)
    
    # Test quantized model
    print("Testing quantized model resource usage...")
    # Force garbage collection before testing
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Warm up
    for _ in range(num_warmup):
        with torch.no_grad():
            quantized_model(test_input)
    
    # Measure resource usage
    for _ in range(num_runs):
        # Record CPU usage before inference
        cpu_before = process.cpu_percent()
        # Record memory usage before inference
        memory_before = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Perform inference and measure time
        start_time = time.time()
        with torch.no_grad():
            quantized_model(test_input)
        inference_time = time.time() - start_time
        
        # Wait briefly for CPU measurement to register
        time.sleep(0.01)
        
        # Record CPU and memory after inference
        cpu_after = process.cpu_percent()
        memory_after = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Record results
        results["quantized"]["inference_times"].append(inference_time)
        results["quantized"]["cpu_usage"].append(cpu_after - cpu_before)
        results["quantized"]["memory_usage"].append(memory_after - memory_before)
    
    # Calculate average metrics
    for model_type in ["original", "quantized"]:
        avg_inf_time = np.mean(results[model_type]["inference_times"]) * 1000  # ms
        avg_cpu = np.mean([x for x in results[model_type]["cpu_usage"] if x >= 0])  # Filter negative values
        avg_memory = np.mean([x for x in results[model_type]["memory_usage"] if x >= 0])  # Filter negative values
        
        results[model_type]["avg_inference_ms"] = avg_inf_time
        results[model_type]["avg_cpu_percent"] = avg_cpu 
        results[model_type]["avg_memory_mb"] = avg_memory
    
    # Print results
    print("\n===== Resource Utilization Comparison =====")
    print(f"{'Metric':<25} {'Original':<15} {'Quantized':<15} {'Improvement':<15}")
    print("-" * 70)
    
    # Calculate and print improvement percentages
    avg_metrics = {
        "avg_inference_ms": "Inference Time (ms)",
        "avg_cpu_percent": "CPU Usage (%)",
        "avg_memory_mb": "Memory Usage (MB)"
    }
    
    for key, label in avg_metrics.items():
        orig_val = results["original"][key]
        quant_val = results["quantized"][key]
        
        # Calculate improvement (for all metrics, lower is better)
        improvement = (orig_val - quant_val) / orig_val * 100 if orig_val > 0 else 0
        
        print(f"{label:<25} {orig_val:<15.4f} {quant_val:<15.4f} {improvement:>+.2f}%")
    
    # Calculate FPS (higher is better)
    orig_fps = 1000 / results["original"]["avg_inference_ms"]
    quant_fps = 1000 / results["quantized"]["avg_inference_ms"]
    fps_improvement = (quant_fps - orig_fps) / orig_fps * 100
    
    print(f"{'Frames Per Second (FPS)':<25} {orig_fps:<15.2f} {quant_fps:<15.2f} {fps_improvement:>+.2f}%")
    
    # Plot results
    plot_resource_comparison(results)
    
    return results

def plot_resource_comparison(results):
    """Create visualizations of resource usage comparison"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Inference time comparison
    axes[0, 0].boxplot([
        np.array(results["original"]["inference_times"]) * 1000,
        np.array(results["quantized"]["inference_times"]) * 1000
    ], labels=["Original", "Quantized"])
    axes[0, 0].set_title("Inference Time (ms)")
    axes[0, 0].set_ylabel("Time (ms)")
    axes[0, 0].grid(True, linestyle='--', alpha=0.7)
    
    # CPU usage comparison
    axes[0, 1].boxplot([
        [x for x in results["original"]["cpu_usage"] if x >= 0],
        [x for x in results["quantized"]["cpu_usage"] if x >= 0]
    ], labels=["Original", "Quantized"])
    axes[0, 1].set_title("CPU Usage (%)")
    axes[0, 1].set_ylabel("CPU Percent (%)")
    axes[0, 1].grid(True, linestyle='--', alpha=0.7)
    
    # Memory usage comparison
    axes[1, 0].boxplot([
        [x for x in results["original"]["memory_usage"] if x >= 0],
        [x for x in results["quantized"]["memory_usage"] if x >= 0]
    ], labels=["Original", "Quantized"])
    axes[1, 0].set_title("Memory Usage (MB)")
    axes[1, 0].set_ylabel("Memory (MB)")
    axes[1, 0].grid(True, linestyle='--', alpha=0.7)
    
    # FPS comparison
    orig_fps = 1000 / results["original"]["avg_inference_ms"]
    quant_fps = 1000 / results["quantized"]["avg_inference_ms"]
    axes[1, 1].bar(["Original", "Quantized"], [orig_fps, quant_fps])
    axes[1, 1].set_title("Frames Per Second (FPS)")
    axes[1, 1].set_ylabel("FPS")
    axes[1, 1].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    # plt.savefig("resource_comparison.png")
    plt.close()
    
    print(f"Resource comparison plots saved to resource_comparison.png")

# Main function
if __name__ == "__main__":
    # Paths
    model_path = "../../models/2025-04-03_23-55-32/epochs_300_lr_1e-03_wd_5e-03_acc_8963.pth"
    save_dir = "../../models/2025-04-03_23-55-32"
    dataset_path = "../../../datasets/vectors/xy_filtered_keypoints_vectors_mediapipe.csv"
    batch_size = 16
    
    # Get calibration data if possible
    try:
        calibration_loader = create_calibration_loader(
            csv_path=dataset_path, 
            scaler_mean_path=os.path.join(save_dir, "scaler_mean.npy"),
            scaler_scale_path=os.path.join(save_dir, "scaler_scale.npy"),
            batch_size=batch_size
        )
    except Exception as e:
        print(f"Could not create calibration loader: {e}")
        calibration_loader = None
        
    # Load and quantize model - static
    original_model, quantized_model = load_and_quantize_model_static(
        model_path=model_path,
        save_dir=save_dir,
        calibration_loader=calibration_loader
    )
    
    # Load and quantize model - dynamic
    # original_model, quantized_model = load_and_quantize_model_dynamic(model_path, save_dir)
    
    # Convert to half precision
    # original_model, quantized_model = convert_to_half_precision(model_path, save_dir)
    
    resource_results = compare_resource_utilization(original_model, quantized_model, batch_size=16)
    
    # Compare inference speed
    compare_inference_speed(original_model, quantized_model)
    
    # Compare model accuracy
    test_loader = get_test_dataloader(
        csv_path=dataset_path,
        scaler_mean_path=os.path.join(save_dir, "scaler_mean.npy"),
        scaler_scale_path=os.path.join(save_dir, "scaler_scale.npy"),
        batch_size=16
    )
    compare_accuracy(original_model, quantized_model, test_loader)