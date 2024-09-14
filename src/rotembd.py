import torch
import torch.nn as nn

from einops import rearrange

class RotaryEmbedding(nn.Module):
    def __init__(self, embedding_dim, scale_base=512, use_xpos=True):
        super().__init__()
        
        inv_frequency = 1.0 / (10000 ** (torch.arange(0, embedding_dim, 2).float() / embedding_dim))
        self.register_buffer("inv_frequency", inv_frequency)
        self.use_xpos = use_xpos
        
        self.scale_base = scale_base
        scaling_factors = (torch.arange(0, embedding_dim, 2) + 0.4 * embedding_dim) / (1.4 * embedding_dim)
        self.register_buffer('scaling_factors', scaling_factors)

    def forward(self, sequence_length, device):
        
        time_steps = torch.arange(sequence_length, device=device).type_as(self.inv_frequency)
        
        frequencies = torch.einsum('i, j -> i j', time_steps, self.inv_frequency)
        frequencies = torch.cat((frequencies, frequencies), dim=-1)

        if not self.use_xpos:
            return frequencies, torch.ones(1, device=device)

        position_offsets = (time_steps - (sequence_length // 2)) / self.scale_base
        scaling_values = self.scaling_factors ** rearrange(position_offsets, 'n -> n 1')
        scaling_values = torch.cat((scaling_values, scaling_values), dim=-1)

        return frequencies, scaling_values