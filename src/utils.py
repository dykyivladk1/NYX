import torch
import torch.nn as nn

import math

from random import randrange

def is_present(value):
    return value is not None

def use_default(primary, fallback):
    if is_present(primary):
        return primary
    return fallback

def evaluation_mode_decorator(method):
    def wrapped_method(obj, *args, **kwargs):
        
        current_training_state = obj.training
        
        obj.eval()
        
        output = method(obj, *args, **kwargs)
        
        obj.train(mode=current_training_state)
        
        return output
    return wrapped_method

def log(t, eps = 1e-20):
    return torch.log(torch.clamp(t, min = eps))

def gumbel_noise(t):
    
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature=1., dim=-1):
    
    gumbel_scaled_logits = (t / max(temperature, 1e-10)) + gumbel_noise(t)

    return gumbel_scaled_logits.argmax(dim = dim)

def identity(t, *args, **kwargs):
    
    output = t  
    return output

def top_k(logits, thres):
    
    num_logits = logits.size(-1)
    k = math.ceil((1 - thres) * num_logits)

    top_vals, top_inds = torch.topk(logits, k, dim=-1)

    masked_logits = torch.full_like(logits, float('-inf'))

    masked_logits.scatter_(dim=1, index=top_inds, src=top_vals)

    return masked_logits

def norm(t):
    return nn.functional.normalize(t, dim = -1)

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer('beta', torch.zeros(dim))

    def forward(self, x):
        return nn.functional.layer_norm(x, x.shape[-1:], self.gamma, self.beta)
    
class SwiGLU(nn.Module):
    def forward(self, x):
        input_tensor, gate = x.chunk(2, dim=-1)
        return nn.functional.silu(gate) * input_tensor

def rotate_(x):
    half1, half2 = x.chunk(2, dim=-1)
    return torch.cat((-half2, half1), dim=-1)

def apply_rotary(position_encodings, tensor, scale=1.0):
    
    return (tensor * position_encodings.cos() * scale) + (rotate_(tensor) * position_encodings.sin() * scale)

