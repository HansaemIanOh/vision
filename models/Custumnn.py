import torch
from torch import Tensor
from torch import nn

class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
    def forward(self, x: Tensor) -> Tensor:
        
        expansion = False
        if len(x.shape) > 2:
            expansion = True
        if expansion:
            order, inverse_order = self.reorder_dimensions(len(x.shape))
            x = x.permute(order).contiguous()
        x = self.linear(x)
        if expansion:
            x = x.permute(inverse_order).contiguous()
        return x
    @staticmethod
    def reorder_dimensions(shape_len):
        order = list(range(shape_len))
        order = [0] + order[2:] + [1]
        inverse_order = [order.index(i) for i in range(shape_len)]
        return order, inverse_order