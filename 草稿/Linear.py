import torch.nn as nn
import torch
from math import sqrt
from einops import einsum
class Linear(nn.Module):
    def __init__(self,in_features: int, out_features: int, device: torch.device|None = None, dtype: torch.dtype|None = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        self.W = nn.Parameter(torch.empty(self.out_features,self.in_features))
        self.std = sqrt(2/(self.in_features + self.out_features))
        nn.init.trunc_normal_(self.W, mean=0, std = self.std,a = -3*self.std, b = 3*self.std)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #assert x.shape[-1] == self.in_features, f"输入维度{x.shape}与线行层{self.W.shape}不匹配"
        if x.shape[-1] != self.in_features:
            raise ValueError(f"输入维度{x.shape}与线行层{self.W.shape}不匹配") 
        return einsum(x, self.W, "... in_features, out_features in_features -> ... out_features")
if __name__ == "__main__":
    linear = Linear(5, 10)
    x = torch.ones(10,6)
    y = linear.forward(x)
    print(y, y.shape)