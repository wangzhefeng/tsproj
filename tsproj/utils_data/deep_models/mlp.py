import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self,
                 d_in: int,
                 d_hid: int,
                 d_out: int,
                 num_layers: int,
                 activation: str = "relu"):
        super().__init__()
        self.num_layers = num_layers
        hidden_dims = [d_hid] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([d_in] + hidden_dims, hidden_dims + [d_out]))
        self.act = getattr(F, activation) if activation is not None else nn.Sequential()
        self._reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        for idx, layer in enumerate(self.layers):
            x = self.act(layer(x)) if idx < self.num_layers - 1 else layer(x)
        return x

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

