import torch
import torch.nn.utils.prune as prune
import os
import time
import numpy as np
import pandas as pd
import psutil
import matplotlib.pyplot as plt
import gc
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder

def load_and_prune_model(model_path, save_dir, pruning_amount=0.3):
    """Load model and apply pruning to reduce parameters"""
    # Load the original model
    input_size = 20  # Input size
    num_classes = 4  # Number of posture classes
    
    # Import MLP Class
    import sys
    sys.path.append(os.path.dirname(model_path))
    from model import MLP
    
    # Create model instance and load weights
    model = MLP(input_size=input_size, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    
    # Create a copy for pruning
    pruned_model = MLP(input_size=input_size, num_classes=num_classes)
    pruned_model.load_state_dict(torch.load(model_path, weights_only=True))
    pruned_model.eval()
    
    # Count parameters before pruning
    orig_params = sum(p.numel() for p in pruned_model.parameters() if p.requires_grad)
    
    # Apply pruning to linear layers
    for name, module in pruned_model.named_modules():
        if isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=pruning_amount)
    
    # Make pruning permanent
    for name, module in pruned_model.named_modules():
        if isinstance(module, torch.nn.Linear):
            prune.remove(module, 'weight')
    
    # Count parameters after pruning
    pruned_params = sum(p.numel() for p in pruned_model.parameters() if p.requires_grad)
    
    # Calculate sparsity
    zero_params = 0
    total_params = 0
    for param in pruned_model.parameters():
        if param.requires_grad:
            zero_params += torch.sum(param == 0).item()
            total_params += param.numel()
    
    sparsity = 100 * zero_params / total_params if total_params > 0 else 0
    
    # Save pruned model
    model_name = os.path.basename(model_path).split(".")[0]
    pruned_model_path = os.path.join(save_dir, f"{model_name}_pruned_{int(pruning_amount*100)}pct.pth")
    torch.save(pruned_model.state_dict(), pruned_model_path)
    
    print(f"Original parameters: {orig_params}")
    print(f"Pruned parameters: {pruned_params}")
    print(f"Parameter reduction: {orig_params - pruned_params} ({(orig_params - pruned_params) / orig_params * 100:.2f}%)")
    print(f"Model sparsity: {sparsity:.2f}%")
    print(f"Original model size: {get_model_size(model_path):.2f} MB")
    print(f"Pruned model size: {get_model_size(pruned_model_path):.2f} MB")
    print(f"Pruned model saved to: {pruned_model_path}")
    
    return model, pruned_model

def get_model_size(model_path):
    """Get the size of a model file in MB"""
    return os.path.getsize(model_path) / (1024 * 1024)

def compare_resource_utilization(original_model, pruned_model, input_size=20, num_warmup=10, num_runs=100, batch_size=16):
    """Compare CPU, memory usage and inference time between original and pruned models"""
    # Create random input tensor for testing
    test_input = torch.randn(batch_size, input_size)
    
    # Dictionary to store results
    results = {
        "original": {"inference_times": [], "cpu_usage": [], "memory_usage": []},
        "pruned": {"inference_times": [], "cpu_usage": [], "memory_usage": []}
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
    
    # Measure baseline memory
    baseline_memory = process.memory_info().rss / (1024 * 1024)
    
    # Measure resource usage
    for _ in range(num_runs):
        # Record CPU usage before inference
        cpu_before = process.cpu_percent()
        
        # Perform inference and measure time
        start_time = time.time()
        with torch.no_grad():
            original_model(test_input)
        inference_time = time.time() - start_time
        
        # Wait briefly for CPU measurement to register
        time.sleep(0.01)
        
        # Record CPU and memory after inference
        cpu_after = process.cpu_percent()
        memory_after = (process.memory_info().rss / (1024 * 1024)) - baseline_memory
        
        # Record results
        results["original"]["inference_times"].append(inference_time)
        results["original"]["cpu_usage"].append(cpu_after)
        results["original"]["memory_usage"].append(memory_after)
    
    # Test pruned model
    print("Testing pruned model resource usage...")
    # Force garbage collection before testing
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Warm up
    for _ in range(num_warmup):
        with torch.no_grad():
            pruned_model(test_input)
    
    # Measure baseline memory
    baseline_memory = process.memory_info().rss / (1024 * 1024)
    
    # Measure resource usage
    for _ in range(num_runs):
        # Record CPU usage before inference
        cpu_before = process.cpu_percent()
        
        # Perform inference and measure time
        start_time = time.time()
        with torch.no_grad():
            pruned_model(test_input)
        inference_time = time.time() - start_time
        
        # Wait briefly for CPU measurement to register
        time.sleep(0.01)
        
        # Record CPU and memory after inference
        cpu_after = process.cpu_percent()
        memory_after = (process.memory_info().rss / (1024 * 1024)) - baseline_memory
        
        # Record results
        results["pruned"]["inference_times"].append(inference_time)
        results["pruned"]["cpu_usage"].append(cpu_after)
        results["pruned"]["memory_usage"].append(memory_after)
    
    # Calculate average metrics
    for model_type in ["original", "pruned"]:
        avg_inf_time = np.mean(results[model_type]["inference_times"]) * 1000  # ms
        avg_cpu = np.mean(results[model_type]["cpu_usage"])
        avg_memory = np.mean([x for x in results[model_type]["memory_usage"] if x >= 0])
        
        results[model_type]["avg_inference_ms"] = avg_inf_time
        results[model_type]["avg_cpu_percent"] = avg_cpu 
        results[model_type]["avg_memory_mb"] = avg_memory
    
    # Print results
    print("\n===== Resource Utilization Comparison =====")
    print(f"{'Metric':<25} {'Original':<15} {'Pruned':<15} {'Improvement':<15}")
    print("-" * 70)
    
    # Calculate and print improvement percentages
    avg_metrics = {
        "avg_inference_ms": "Inference Time (ms)",
        "avg_cpu_percent": "CPU Usage (%)",
        "avg_memory_mb": "Memory Usage (MB)"
    }
    
    for key, label in avg_metrics.items():
        orig_val = results["original"][key]
        pruned_val = results["pruned"][key]
        
        # Calculate improvement (for all metrics, lower is better)
        improvement = (orig_val - pruned_val) / orig_val * 100 if orig_val > 0 else 0
        
        print(f"{label:<25} {orig_val:<15.4f} {pruned_val:<15.4f} {improvement:>+.2f}%")
    
    # Calculate FPS (higher is better)
    orig_fps = 1000 / results["original"]["avg_inference_ms"]
    pruned_fps = 1000 / results["pruned"]["avg_inference_ms"]
    fps_improvement = (pruned_fps - orig_fps) / orig_fps * 100
    
    print(f"{'Frames Per Second (FPS)':<25} {orig_fps:<15.2f} {pruned_fps:<15.2f} {fps_improvement:>+.2f}%")
    
    # Plot results
    plot_resource_comparison(results)
    
    return results

def plot_resource_comparison(results):
    """Create visualizations of resource usage comparison"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Inference time comparison
    axes[0, 0].boxplot([
        np.array(results["original"]["inference_times"]) * 1000,
        np.array(results["pruned"]["inference_times"]) * 1000
    ], labels=["Original", "Pruned"])
    axes[0, 0].set_title("Inference Time (ms)")
    axes[0, 0].set_ylabel("Time (ms)")
    axes[0, 0].grid(True, linestyle='--', alpha=0.7)
    
    # CPU usage comparison
    axes[0, 1].boxplot([
        results["original"]["cpu_usage"],
        results["pruned"]["cpu_usage"]
    ], labels=["Original", "Pruned"])
    axes[0, 1].set_title("CPU Usage (%)")
    axes[0, 1].set_ylabel("CPU Percent (%)")
    axes[0, 1].grid(True, linestyle='--', alpha=0.7)
    
    # Memory usage comparison
    axes[1, 0].boxplot([
        [x for x in results["original"]["memory_usage"] if x >= 0],
        [x for x in results["pruned"]["memory_usage"] if x >= 0]
    ], labels=["Original", "Pruned"])
    axes[1, 0].set_title("Memory Usage (MB)")
    axes[1, 0].set_ylabel("Memory (MB)")
    axes[1, 0].grid(True, linestyle='--', alpha=0.7)
    
    # FPS comparison
    orig_fps = 1000 / results["original"]["avg_inference_ms"]
    pruned_fps = 1000 / results["pruned"]["avg_inference_ms"]
    axes[1, 1].bar(["Original", "Pruned"], [orig_fps, pruned_fps])
    axes[1, 1].set_title("Frames Per Second (FPS)")
    axes[1, 1].set_ylabel("FPS")
    axes[1, 1].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    # plt.savefig("pruning_comparison.png")
    plt.close()
    
    print(f"Resource comparison plots saved to pruning_comparison.png")

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

def compare_accuracy(original_model, pruned_model, test_loader, device='cpu'):
    """Compare accuracy between original and pruned models"""
    original_model.to(device)
    pruned_model.to(device)
    
    models = {"Original": original_model, "Pruned": pruned_model}
    
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

# Main function
if __name__ == "__main__":
    # Paths
    model_path = "../../models/2025-04-03_23-55-32/epochs_300_lr_1e-03_wd_5e-03_acc_8963.pth"
    save_dir = "../../models/2025-04-03_23-55-32"
    dataset_path = "../../../datasets/vectors/xy_filtered_keypoints_vectors_mediapipe.csv"
    
    # Set pruning amount (0.3 = 30% of weights)
    pruning_amount = 0.3
    
    # Load and prune model
    original_model, pruned_model = load_and_prune_model(model_path, save_dir, pruning_amount)
    
    # Compare resource utilization
    resource_results = compare_resource_utilization(original_model, pruned_model, batch_size=16)
    
    # Compare accuracy
    test_loader = get_test_dataloader(
        csv_path=dataset_path,
        scaler_mean_path=os.path.join(save_dir, "scaler_mean.npy"),
        scaler_scale_path=os.path.join(save_dir, "scaler_scale.npy"),
        batch_size=32
    )
    compare_accuracy(original_model, pruned_model, test_loader)