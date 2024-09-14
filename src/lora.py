import torch
import torch.nn as nn

class LowRankAdapter(nn.Module):
    def __init__(self, input_dim, output_dim, rank=8, alpha=None):
        super(LowRankAdapter, self).__init__()

        alpha = rank if alpha is None else alpha
        self.scaling = alpha / rank

        self.matrix_a = nn.Parameter(torch.randn(input_dim, rank))
        self.matrix_b = nn.Parameter(torch.zeros(rank, output_dim))

    @property
    def weight_matrix(self):
        
        return torch.mm(self.matrix_a, self.matrix_b) * self.scaling

    def forward(self, x):
        
        return torch.mm(x, self.weight_matrix)
