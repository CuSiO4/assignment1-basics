import torch
import torch.nn as nn
from einops import einsum, rearrange
from RoPE import RotrayPositionEmbedding
from softmax import softmax

class Attention(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, 
                Q,K,V,mask):
        QKT = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys")
        # if mask is not None:
        #     QKT[mask==False] = -inf
        # 为True的不变，为false的修改为-inf、
        if mask is not None:
            QKT = torch.where(mask, QKT, float('-inf'))
        d = torch.tensor(Q.shape[-1])
        sm = softmax(QKT / torch.sqrt(d), dim=-1)
        return einsum(sm, V, "... queries keys, ... keys d_v -> ... queries d_v")
class Multihead_self_attention(nn.Module):
    def __init__(self, d_models:int, nums_head: int, theta: int, max_seq_len: int):
        super().__init__()
        self.d_models = d_models
        self.nums_head = nums_head
        self.d_k = d_models / nums_head
        self.W_q = nn.Parameter(torch.empty(d_models,d_models))
        self.W_k = nn.Parameter(torch.empty(d_models,d_models))
        self.W_v = nn.Parameter(torch.empty(d_models,d_models))
        self.W_o = nn.Parameter(torch.empty(d_models,d_models))
        self.attention = Attention()
        self.rope = RotrayPositionEmbedding(self.d_k, theta, max_seq_len)
    def forward(self, in_features: torch.Tensor, token_position : torch.Tensor = None):
        b, s, d = in_features.shape
        # 这里可以直接用Linear表示
        Q = einsum(in_features, self.W_q, "... sequence_length d_model, d_model d_model -> ... sequence_length d_model")
        K = einsum(in_features, self.W_k, "... sequence_length d_model, d_model d_model -> ... sequence_length d_model")
        V = einsum(in_features, self.W_v, "... sequence_length d_model, d_model d_model -> ... sequence_length d_model")
        Q_m = rearrange(Q,"... sequence (nums_head d_k) -> ... nums_head sequence d_k",nums_head = self.nums_head)
        K_m = rearrange(K,"... sequence (nums_head d_k) -> ... nums_head sequence d_k",nums_head = self.nums_head)
        V_m = rearrange(V,"... sequence (nums_head d_k) -> ... nums_head sequence d_k",nums_head = self.nums_head)
        
        if self.rope is not None:
            Q_m = self.rope(Q_m, token_position)
            K_m = self.rope(K_m, token_position)
        
        # 创建一个下三角为True矩阵
        mask = []
        for i in range(s):
            t = []
            for j in range(s):
                if (j > i): t.append(False)
                else: t.append(True)
            mask.append(t)
        mask = torch.tensor(mask)
        attention_m = self.attention(Q_m, K_m, V_m, mask)
        attention = rearrange(attention_m,"... nums_head sequence d_k -> ... sequence (nums_head d_k)")
        return einsum(self.W_o, attention, "d_model d_model, ... sequence d_model -> ... sequence d_model")
if __name__ == "__main__":
    x = torch.ones(2,3,4)
    # 用括号 () 告诉 einops 这些维度是拆分关系
    y = rearrange(x,"... sequence (nums_head d_k) -> ... nums_head sequence d_k",nums_head = 2)
    print(y.shape)


