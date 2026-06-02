import torch
def softmax(x:torch.Tensor, dim:int):
    x_max = torch.max(x,keepdim=True,dim=dim).values
    x_exp = torch.exp(x - x_max)
    x_sum = torch.sum(x_exp, dim=dim, keepdim=True)
    return x_exp / x_sum
x = torch.tensor([[1, 2, 3],[4, 5, 6]])
print(x)
y = softmax(x, 0)
print(y)