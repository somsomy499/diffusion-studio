"""Tests for schedulers."""
import numpy as np
from diffusion_studio.scheduler import DDPMScheduler, SchedulerConfig

def test_ddpm_init():
    sched = DDPMScheduler()
    assert len(sched.betas) == 1000
    assert sched.alphas_cumprod[0] > sched.alphas_cumprod[-1]

def test_add_noise():
    sched = DDPMScheduler()
    sample = np.ones((1, 3, 64, 64))
    noise = np.random.randn(*sample.shape)
    noisy = sched.add_noise(sample, noise, timestep=500)
    assert noisy.shape == sample.shape
    assert not np.allclose(noisy, sample)
