"""LoRA (Low-Rank Adaptation) for diffusion models."""
import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class LoRAConfig:
    rank: int = 8
    alpha: float = 16.0
    target_modules: list = None
    dropout: float = 0.0
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["to_q", "to_k", "to_v", "to_out.0"]

class LoRALayer:
    def __init__(self, in_features, out_features, config: LoRAConfig):
        self.rank = config.rank
        self.alpha = config.alpha
        self.scaling = config.alpha / config.rank
        
        # Low-rank matrices
        self.lora_A = np.random.randn(config.rank, in_features).astype(np.float32) * 0.01
        self.lora_B = np.zeros((out_features, config.rank), dtype=np.float32)
        
    def forward(self, x):
        return x + (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
    
    def merge(self, original_weight):
        return original_weight + self.lora_B @ self.lora_A * self.scaling

class LoRAModel:
    def __init__(self, config: LoRAConfig):
        self.config = config
        self.layers = {}
        
    def add_layer(self, name, in_features, out_features):
        self.layers[name] = LoRALayer(in_features, out_features, self.config)
        
    def trainable_params(self):
        count = 0
        for layer in self.layers.values():
            count += layer.lora_A.size + layer.lora_B.size
        return count
    
    def save(self, path):
        state = {name: {"A": l.lora_A, "B": l.lora_B} for name, l in self.layers.items()}
        np.savez(path, **{f"{k}_{k2}": v for k, d in state.items() for k2, v in d.items()})
    
    def load(self, path):
        data = np.load(path)
        for name in self.layers:
            self.layers[name].lora_A = data[f"{name}_A"]
            self.layers[name].lora_B = data[f"{name}_B"]
