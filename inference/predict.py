import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import torch.nn as nn

# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),  # Add batch normalization
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),  # Add batch normalization
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# Function for model inference
def predict_posture(model, keypoints, scaler, class_labels):
    """
    Perform inference using the model.
    """
    # Normalize keypoints
    keypoints_normalized = scaler.transform([keypoints])  # StandardScaler expects 2D array

    # Convert to PyTorch tensor
    input_tensor = torch.tensor(keypoints_normalized, dtype=torch.float32)

    # Perform inference
    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)

    # Map predicted index to class label
    return class_labels[predicted.item()]


# Main pipeline
if __name__ == "__main__":
    # Define the data point (replace with your specific keypoints)
    data_point = {
        "nose_x": 0.292448, "nose_y": 0.124726,
        "left_shoulder_x": 0.255454, "left_shoulder_y": 0.220369,
        "right_shoulder_x": 0.170209, "right_shoulder_y": 0.230945,
        "left_hip_x": 0.263503, "left_hip_y": 0.47753,
        "right_hip_x": 0.199283, "right_hip_y": 0.497099,
        "left_knee_x": 0.0, "left_knee_y": 0.0,
        "right_knee_x": 0.479481, "right_knee_y": 0.582801,
        "left_ankle_x": 0.494829, "left_ankle_y": 0.76498,
        "right_ankle_x": 0.456039, "right_ankle_y": 0.797581,
        "shoulder_midpoint_x": 0.212831, "shoulder_midpoint_y": 0.225657,
        "class": "proper"
    }

    # Extract features and true label
    features = np.array([v for k, v in data_point.items() if k != "class"], dtype=np.float32)
    true_label = data_point["class"]

    # Path to saved model
    model_path = "../models/2024-11-24_16-34-03/epochs_150_lr_1e-03_wd_1e-03_acc_8298.pth"  # Replace with your saved model path
    class_labels = ["crossed_legs", "proper", "reclining", "slouching"]  # Your class labels

    # Load the saved model
    input_size = features.shape[0]  # Length of the input keypoints vector
    model = MLP(input_size=input_size, num_classes=len(class_labels))
    model.load_state_dict(torch.load(model_path, weights_only=False))
    model.eval()

    # Load the StandardScaler used during training
    scaler = StandardScaler()
    scaler.mean_ = np.load("../models/2024-11-24_16-34-03/scaler_mean.npy")  # Load scaler's mean
    scaler.scale_ = np.load("../models/2024-11-24_16-34-03/scaler_scale.npy")  # Load scaler's scale

    # Predict the posture
    predicted_label = predict_posture(model, features, scaler, class_labels)
    print(f"True Label: {true_label}, Predicted Label: {predicted_label}")
