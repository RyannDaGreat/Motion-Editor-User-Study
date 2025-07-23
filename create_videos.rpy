# 2025-03-07 07:29:36.931516
#pip install rp textual textual[syntax]

import rp
import torch
from diffusers import StableDiffusionPipeline
from contextlib import contextmanager


def add_forward_hooks(module):
    """Add hooks to record stats during forward pass of all submodules"""
    
    hooks = []
    
    def forward_hook(module, input, output):
        if rp.is_torch_tensor(output):
            if not hasattr(module, "forward_stats"):
                module.forward_stats = []
                
            stats = rp.as_easydict(
                shape=output.shape,
                dtype=output.dtype,
                device=output.device,
                min=float(output.min()),
                max=float(output.max()),
                mean=float(output.mean()),
                std=float(output.std()),
            )
            
            module.forward_stats.append(stats)
    
    # Register hook for all submodules
    for name, submodule in module.named_modules():
        if hasattr(submodule, "forward") and callable(submodule.forward):
            # Clear existing stats
            if hasattr(submodule, "forward_stats"):
                del submodule.forward_stats
            
            # Register the hook
            handle = submodule.register_forward_hook(forward_hook)
            hooks.append(handle)
    
    return hooks


@contextmanager
def stats_collection(module):
    """Context manager to collect stats during forward pass"""
    hooks = add_forward_hooks(module)
    try:
        yield
    finally:
        # Remove hooks when done
        for hook in hooks:
            hook.remove()

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", revision="fp16", torch_dtype=torch.float16)
pipe = pipe.to('mps')

# Use the context manager for clean hook management
with stats_collection(pipe.unet):
    pipe('Girl', num_inference_steps=3)

# Explore the collected stats
rp.explore_torch_module(pipe.unet)
