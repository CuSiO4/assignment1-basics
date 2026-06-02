import torch
import torch.nn as nn
class RotrayPositionEmbedding(nn.Module):
    def __init__(self, d_k: int,theta: float, max_seq_len: int, device = None):
        '''
        RoPE模块
        theta: 基准频率（通常为10000）
        d_k: headnum 每个注意力的维度
        max_seq_len: 序列最大长度
        '''
        super().__init__()
        self.d_k = d_k
        # 1.计算omega_k = 1/theta^(-2k / d)
        powers = torch.arange(0,d_k,2,device = device).float() / d_k
        freqs = 1.0 / (theta**powers)
        # 2.创建位置序列
        t = torch.arange(max_seq_len,device=device).float()
        # 3.创建m*\omega
        freqs_matrix = torch.outer(t, freqs)
        # 4.将sin、cos作为buffer注册
        # 关于buffer：
        # register_buffer(name, tensor, persistent（存checkpoint时是否保存）)
        # 
        self.register_buffer("sin", freqs_matrix.sin(), persistent=False)
        self.register_buffer("cos", freqs_matrix.cos(), persistent=False)
    def forward(self, x: torch.Tensor, token_position:torch.Tensor):
        cos = self.cos_cached[token_position]
        sin = self.sin_cached[token_position]

        # 维度对齐
        if x.ndim > sin.ndim and sin.ndim > 3:
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
        cos = cos.to(x.dtype)
        sin = sin.to(x.dtype)

        x_even = x[...,0::2]
        x_odd = x[...,1::2]

        output = torch.empty_like(x)
        output[...,0::2] = x_even * cos - x_odd * sin
        output[...,1::2] = x_even * sin - x_odd * cos
        return output

