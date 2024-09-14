import torch.nn as nn

class Residual(nn.Module):
    def __init__(self, func):
        super(Residual, self).__init__()

        self.func = func

    def forward(self, x, **kwargs):
        res = self.func(x, **kwargs)
        if not any([t.requires_grad for t in (x, res)]):
            return x.add_(res)
        
        return res + x