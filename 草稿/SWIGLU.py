import torch
import torch.nn as nn
from einops import einsum
class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, device = None, dtype = None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.W_1 = nn.Parameter(torch.randn(self.d_ff,self.d_model))
        self.W_2 = nn.Parameter(torch.randn(self.d_model,self.d_ff))
        self.W_3 = nn.Parameter(torch.randn(self.d_ff,self.d_model))
        self.sigmoid = torch.sigmoid
    def forward(self, x):
        # swiglu = W_2@(silu(W_1@x)*W_3@x)
        W_1x = einsum(self.W_1, x, "dff dmodel, ... dmodel -> ... dff")
        # silu = x * sigmoid(x)
        silu = W_1x * self.sigmoid(W_1x)
        W_3x = einsum(self.W_3, x, "dff dmodel, ... dmodel -> ... dff")
        GLU = silu * W_3x
        return einsum(self.W_2, GLU, "d_model dff, ... dff -> ... d_model")