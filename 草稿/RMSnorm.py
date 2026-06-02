import torch
import torch.nn as nn
class RMSNorm(nn.Module):
    def __init__(self, 
                d_model: int,
                eps: float = 1e-5,
                device = None,
                dtype = None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        self.g = nn.Parameter(torch.ones(d_model,device=self.device))
    def forward(self, x: torch.Tensor) -> torch.Tensor :
        in_dtype = x.dtype
        x = x.to(torch.float32)
        inv_rms = torch.rsqrt(torch.mean(torch.square(x),-1,keepdim=True)+self.eps)
        y = x * inv_rms
        y = y * self.g
        return y.to(in_dtype)
if __name__ == "__main__":
    norm = RMSNorm(d_model=5)
    x = torch.ones(2,3,5)
    y = norm.forward(x)
    print(y, y.shape)


