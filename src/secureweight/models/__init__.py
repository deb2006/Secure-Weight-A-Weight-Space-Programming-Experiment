"""Small, single-hidden-layer classification networks."""

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_dim: int = 4, hidden_dim: int = 8, output_dim: int = 3):
        super().__init__()
        if any(type(v) is not int or v < 1 for v in (input_dim, hidden_dim, output_dim)):
            raise ValueError("model dimensions must be positive integers")
        self.architecture = {"input_dim": input_dim, "hidden_dim": hidden_dim,
                             "output_dim": output_dim}
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.output(torch.relu(self.hidden(x)))

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())
