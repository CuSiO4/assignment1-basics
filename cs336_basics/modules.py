#cls ds
import torch.nn as nn
import torch, math
import numpy as np
import numpy.typing as npt
from math import sqrt
from einops import einsum
from einops import einsum, rearrange
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
            raise f"输入维度{x.shape}与线行层{self.W.shape}不匹配"
        return einsum(x, self.W, "... in_features, out_features in_features -> ... out_features")
class Embedding(nn.Module):
    def __init__(self, num_embeddings: int,  #vocab 大小
                embedding_dim: int, 
                device: torch.device|None = None,
                dtype: torch.dtype|None = None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        self.embedding_matrix = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        nn.init.trunc_normal_(self.embedding_matrix, mean = 0, std = 1, a = -3, b = 3)
    def forward(self, token_ids: torch.Tensor) ->torch.Tensor:
        return self.embedding_matrix[token_ids]
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
class RoPE(nn.Module):
    def __init__(self, d_k: int, theta: float, max_seq_len: int,device = None):
        super().__init__()
        self.d_k = d_k
        power = torch.arange(0, d_k, 2,device = device).float() / d_k
        freq = 1.0 / (theta**power)
        position = torch.arange(max_seq_len,device = device).float()
        freq_matrix = torch.outer(position, freq)
        self.register_buffer("sin", freq_matrix.sin(), persistent=False)
        self.register_buffer("cos", freq_matrix.cos(), persistent=False)
    def forward(self, x, token_positions):
        sin = self.sin[token_positions]
        cos = self.cos[token_positions]

        sin = sin.to(x.dtype)
        cos = cos.to(x.dtype)
        # 如果x四维，cos三维，则需要加一项目
        if x.ndim > sin.ndim and cos.ndim >= 3:
            sin = sin.unsqueeze(1)
            cos = cos.unsqueeze(1)
        output = torch.empty_like(x)
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        output[...,0::2] = x_even * cos - x_odd * sin
        output[...,1::2] = x_even * sin + x_odd * cos
        return output
def softmax(x:torch.Tensor, dim:int):
    x_max = torch.max(x,keepdim=True,dim=dim).values
    x_exp = torch.exp(x - x_max)
    x_sum = torch.sum(x_exp, dim=dim, keepdim=True)
    return x_exp / x_sum
class Scaled_dot_product_attention(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, 
                Q,K,V,mask= None,
                ):
        QKT = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys")
        # if mask is not None:
        #     QKT[mask==False] = -inf
        # 为True的不变，为false的修改为-inf、
        if mask is not None:
            QKT = torch.where(mask, QKT, float("-inf")) 
        d = torch.tensor(Q.shape[-1])
        # 每一行是一个 query 对所有 keys 的评分，对其使用softmax
        sm = softmax(QKT / torch.sqrt(d), dim=-1)
        return einsum(sm, V, "... queries keys, ... keys d_v -> ... queries d_v")
class Multihead_self_attention(nn.Module):
    def __init__(self, d_models:int, nums_head: int, theta: int|None = None, max_seq_len: int|None = None, device = None):
        super().__init__()
        self.d_models = d_models
        self.nums_head = nums_head
        self.d_k = d_models // nums_head
        self.theta = theta
        # self.W_q = nn.Parameter(torch.empty(d_models,d_models))
        # self.W_k = nn.Parameter(torch.empty(d_models,d_models))
        # self.W_v = nn.Parameter(torch.empty(d_models,d_models))
        # self.W_o = nn.Parameter(torch.empty(d_models,d_models))
        self.linear_q = Linear(d_models, d_models, device=device)
        self.linear_k = Linear(d_models, d_models, device=device)
        self.linear_v = Linear(d_models, d_models, device=device)
        self.linear_o = Linear(d_models, d_models, device=device)
        self.attention = Scaled_dot_product_attention()
        if theta is not None:
            self.rope = RoPE(self.d_k, theta, max_seq_len)
        else:
            self.rope = None
    def forward(self, in_features: torch.Tensor, token_position : torch.Tensor = None):
        b, s, d = in_features.shape
        # Linear 都是输入X * W_T, W为（out_channel, in_channel）
        # Q = einsum(in_features, self.W_q, "... sequence_length d_model, d_k d_model  -> ... sequence_length d_k")
        # K = einsum(in_features, self.W_k, "... sequence_length d_model, d_k d_model  -> ... sequence_length d_k")
        # V = einsum(in_features, self.W_v, "... sequence_length d_model, d_k d_model  -> ... sequence_length d_k")
        Q = self.linear_q(in_features)
        K = self.linear_k(in_features)
        V = self.linear_v(in_features)
        Q_m = rearrange(Q,"... sequence (nums_head d_k) -> ... nums_head sequence d_k",nums_head = self.nums_head)
        K_m = rearrange(K,"... sequence (nums_head d_k) -> ... nums_head sequence d_k",nums_head = self.nums_head)
        V_m = rearrange(V,"... sequence (nums_head d_k) -> ... nums_head sequence d_k",nums_head = self.nums_head)
        
        if self.rope is not None:
            if token_position is None:
                # .expand = shape 变大、stride 置零，data 不变 → 不同索引找到同一块内存，看起来像复制了，实际没有。
                token_position = torch.arange(s, device=in_features.device).expand(b, s)
            Q_m = self.rope(Q_m, token_position)
            K_m = self.rope(K_m, token_position)
        
        #创建一个下三角为True矩阵
        mask = []
        for i in range(s):
            t = []
            for j in range(s):
                if (j > i): t.append(False)
                else: t.append(True)
            mask.append(t)
        mask = torch.tensor(mask,device=in_features.device)
        #mask = torch.tril(torch.ones(s, s, dtype=torch.bool))
        attention_m = self.attention.forward(Q_m, K_m, V_m, mask)
        attention = rearrange(attention_m,"... nums_head sequence d_k -> ... sequence (nums_head d_k)")
        # 同一个操作数内出现两次 d_model，会导致出错
        #return einsum(attention,self.W_o, "... sequence d_model , d_k d_model -> ... sequence d_k")
        return self.linear_o(attention)
class Transform_block(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, theta: int = None, max_seq_len: int = None, device = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rmsnorm1 = RMSNorm(d_model, device=device)
        self.rmsnorm2 = RMSNorm(d_model, device=device)
        self.multihead_attention = Multihead_self_attention(d_model, num_heads,theta=theta, max_seq_len=max_seq_len,device = device)
        self.swiglu = SwiGLU(d_model, d_ff, device)
    def forward(self, x, token_position=None):
        y = self.rmsnorm1(x)
        y = self.multihead_attention(y, token_position)
        y = y + x
        z = self.rmsnorm2(y)
        z = self.swiglu(z)
        z = y + z
        return z
class Transformer_lm(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, num_layers: int, d_model: int, num_heads: int,d_ff: int, theta: int|None = None, device = None):
        """
        vocab_size: int The size of the vocabulary, necessary for determining the dimensionality of the 
        token embedding matrix.
        context_length: int The maximum context length, necessary for determining the dimensionality 
        of the RoPE sin and cos buffer.
        num_layers: int The number of Transformer blocks to use.
        """
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model)
        self.norm = RMSNorm(d_model)
        self.output_linear = Linear(d_model, vocab_size)
        self.softmax = softmax
        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(Transform_block(d_model, num_heads, d_ff, theta, context_length))
        if device is not None:
            self.to(device)  # 一行搞定所有子模块
    def forward(self, x: torch.Tensor, token_positions = None):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x,token_positions)
        x_norm = self.norm(x)
        return self.output_linear(x_norm)
class Cross_entropy(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, logits, targets):
        # logits (batch, vocab_size)
        # target (batch)
        # 先计算然后选择对应target的概率///先计算可以将exp和log抵消
        logits_max = torch.max(logits, dim=-1,keepdim = True).values
        log_softmax_logits = logits - logits_max - torch.log(torch.sum(torch.exp(logits - logits_max),dim = -1,keepdim=True))
        index = targets.unsqueeze(-1)
        # gather要求：两个矩阵有一个dim不同，其他相同
        return torch.mean(torch.gather(-log_softmax_logits, dim = -1, index = index)) # 不指定 dim，消灭所有维度
        # 困惑度 在原本的基础上取一个exp 交叉熵适合优化（对数空间梯度好），困惑度适合报告（线性空间人类好理解）。两者信息量完全一样，只是尺度不同。
class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr, weight_decay, betas, eps):
        if lr < 0:
            raise ValueError(f"Invalid learning rate lr: {lr}")
        default = {"lr" : lr, "betas" : betas, "weight_decay" : weight_decay, "eps": eps}
        super().__init__(params, default)
    def step(self, closure = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1 = group["betas"][0]
            beta2 = group["betas"][1]
            epsilion = group["eps"]
            llambda = group["weight_decay"]
            for p in group["params"]:
                if p.grad == None:
                    continue
                s = len(p)
                state = self.state[p]
                t = state.get("t", 1)
                # m二阶矩 n一阶矩 尺寸和参数s一致
                m = state.get("m", torch.zeros_like(p))
                v = state.get("v", torch.zeros_like(p))
                g = p.grad.data
                # 修正lr---训练初期m，v偏小
                lr_t = lr * sqrt(1 - beta2**t)/(1 - beta1**t)
                # 权重衰减
                p.data -= p.data * lr * llambda
                # 更新m，v 一阶矩二阶矩
                m = beta1 * m + (1 - beta1)*g
                v = beta2 * v + (1 - beta2)*(g**2)
                # 更新权重
                p.data -= lr_t * (m / (torch.sqrt(v) + epsilion))
                state["m"] = m
                state["v"] = v
                state["t"] = t + 1
        return loss
def get_lr_cosine(t: int, alpha_max: float, alpha_min: float, T_w: int, T_c: int) -> float:
    if t < T_w:
        alpha_t = t / T_w * alpha_max
    elif t<= T_c and t>= T_w:
        alpha_t = alpha_min + (1 + math.cos(math.pi / (T_c - T_w) * (t - T_w) )) / 2 * (alpha_max - alpha_min)
    elif t > T_c:
        alpha_t = alpha_min
    else :
        raise ValueError(f"t in out of range ({T_w}, {T_c})")
    return alpha_t
def gradient_clipping(parameters, max_l2_norm: float, eps: float = 1e-6):
    '''
    实现全局梯度裁剪
    parameters: 模型的所有参数
    max_norm: 允许的最大梯度 L2 范数
    '''
    params_with_grad = [param for param in parameters if param.grad is not None]
    if not params_with_grad:
        return 
    total_norm = 0
    for p in params_with_grad:
        param_norm = torch.norm(p.grad.detach(), p = 2)
        total_norm += param_norm ** 2
    total_norm = total_norm ** 0.5
    if total_norm > max_l2_norm:
        clip_coef = max_l2_norm / (total_norm + eps)
        for p in params_with_grad:
            # 原地 （in-place）修改 x ，把 x 变成 x * y
            p.grad.detach().mul_(clip_coef)
def data_loading(dataset: npt.NDArray, batch_size: int, context_length: int, device: str):
    '''
    随机采样一个训练批次
    返回值：
        x:输入张量(batch_size, context_length)
        y:目标张量 (batch_size, context_length)
    '''
    n = len(dataset)
    max_idx = n - context_length - 1
    # 随机起始点的张量
    ix = torch.randint(0, max_idx + 1,(batch_size,))
    # from_numpy 从numpy转变为torch
    # stack将一系列形状相同的向量堆叠起来
    x = torch.stack([torch.from_numpy(dataset[i: i+context_length].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(dataset[i + 1: i+context_length + 1].astype(np.int64)) for i in ix])
    return (x.to(device), y.to(device))
    # 内存映射
    '''
    不会直接f.read()
    而是 data = np.memmap(path, dtype = np.uint16,mode = 'r)
    '''
def save_checkpoint(model, optimizer, iteration, out):
    '''
    将模型、优化器、迭代次数保存到对象out当中
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    iteration: int
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
    '''
    check_point = {
        'model_state_dict' : model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration
    }
    torch.save(check_point, out)
def load_checkpoint(src, model, optimizer):
    '''
    从src加载检查点，然后从该点恢复model和optimizer的状态
    '''
    check_point = torch.load(src, map_location='cpu')
    model.load_state_dict(check_point['model_state_dict'])
    optimizer.load_state_dict(check_point['optimizer_state_dict'])
    return check_point['iteration']
    