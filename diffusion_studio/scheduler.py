"""Noise schedulers for diffusion models."""
import numpy as np
from typing import List
from dataclasses import dataclass

@dataclass
class SchedulerConfig:
    num_train_timesteps: int = 1000
    beta_start: float = 0.00085
    beta_end: float = 0.012
    beta_schedule: str = "linear"

class DDPMScheduler:
    def __init__(self, config: SchedulerConfig = None):
        self.config = config or SchedulerConfig()
        self.betas = self._get_betas()
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas)
        
    def _get_betas(self):
        c = self.config
        if c.beta_schedule == "linear":
            return np.linspace(c.beta_start, c.beta_end, c.num_train_timesteps)
        elif c.beta_schedule == "cosine":
            steps = c.num_train_timesteps
            t = np.linspace(0, steps, steps + 1)
            alphas_cumprod = np.cos(((t / steps) + 0.008) / 1.008 * np.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return np.clip(betas, 0, 0.999)
        return np.linspace(c.beta_start, c.beta_end, c.num_train_timesteps)
    
    def add_noise(self, sample, noise, timestep):
        alpha_prod = self.alphas_cumprod[timestep]
        return np.sqrt(alpha_prod) * sample + np.sqrt(1 - alpha_prod) * noise
    
    def step(self, model_output, timestep, sample):
        alpha_prod = self.alphas_cumprod[timestep]
        alpha_prod_prev = self.alphas_cumprod[timestep - 1] if timestep > 0 else 1.0
        beta = 1 - alpha_prod / alpha_prod_prev
        
        pred_original = (sample - np.sqrt(1 - alpha_prod) * model_output) / np.sqrt(alpha_prod)
        pred_original = np.clip(pred_original, -1, 1)
        
        mean = np.sqrt(alpha_prod_prev) * beta * pred_original + np.sqrt(1 - alpha_prod_prev) * (1 - beta) * sample
        variance = beta
        noise = np.random.randn(*sample.shape)
        
        return mean + np.sqrt(variance) * noise
    
    def set_timesteps(self, num_inference_steps):
        step_ratio = self.config.num_train_timesteps // num_inference_steps
        return list(range(0, self.config.num_train_timesteps, step_ratio))[:num_inference_steps]
