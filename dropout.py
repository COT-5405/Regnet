import torch.nn as nn

class FixedDropout(nn.Module):
    def __init__(self, p=0.5):
        super(FixedDropout, self).__init__()
        self.p = p
        
    def forward(self, x):
        if not self.training:
            return x
        mask = x.new_empty(x.shape, dtype=torch.bool).bernoulli_(1 - self.p)
        return x.masked_fill(mask, 0) / (1 - self.p)
