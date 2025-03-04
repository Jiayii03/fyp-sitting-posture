import torch
import os
import sys
import importlib.util
import numpy as np
from sklearn.preprocessing import StandardScaler
from config.settings import MODEL_DICT, INPUT_SIZE, CLASS_LABELS

class ModelManager:
    def __init__(self):
        self.model_cache = {}
        
    def load_model(self, model_type):
        if model_type in self.model_cache:
            print(f"Model loaded from cache: {model_type}")
            return self.model_cache[model_type]
        
        model_dir = MODEL_DICT[model_type]["model_dir"]
        MLP = self.import_mlp_from_dir(model_dir)
        model_path = MODEL_DICT[model_type]["model_path"]
        model = MLP(input_size=INPUT_SIZE, num_classes=len(CLASS_LABELS)) 
        model.load_state_dict(torch.load(model_path))
        model.eval()
        self.model_cache[model_type] = model
        print(f"Model loaded: {model_type}")
        return model

    def load_scaler(self, model_dir):
        scaler = StandardScaler()
        scaler.mean_ = np.load(os.path.join(model_dir, 'scaler_mean.npy'))
        scaler.scale_ = np.load(os.path.join(model_dir, 'scaler_scale.npy'))
        return scaler
        
    def import_mlp_from_dir(self, model_dir):
        sys.path.insert(0, model_dir)
        spec = importlib.util.spec_from_file_location("MLP", os.path.join(model_dir, "model.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.path.pop(0)
        return module.MLP