from collections.abc import Callable, Iterable
from typing import Optional
import torch,math
class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr = 1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate : {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults) # 调用父类的方法，并且把参数打包为param_groups/更加灵活
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad
                # 看类型：不可变类型是"赋值"，可变类型是"引用"
                state["t"] = t+1
                print(state)
                break
            break
        return loss
class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr, beta1, beta2, epsilion, llambda):
        if lr < 0:
            raise ValueError(f"Invalid learning rate lr: {lr}")
        default = {"lr" : lr, "beta1" : beta1, "beta2" : beta2, "epsilion": epsilion, "lambda" : llambda}
        super().__init__(params, default)
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            epsilion = group["epsilion"]
            llambda = group["lambda"]
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
                lr_t = lr * math.sqrt(1 - beta2**t)/(1 - beta1**t)
                p.data -= p.data * lr * llambda
                m = beta1 * m + (1 - beta1)*g
                v = beta2 * v + (1 - beta2)*(g**2)
                p.data -= lr_t * (m / (torch.sqrt(v) + epsilion))
                state["m"] = m
                state["v"] = v
                state["t"] = t + 1
        return loss


weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
opt = AdamW([weights], lr = 1e-1, beta1 = 0.9, beta2 = 0.99, epsilion = 1e-8, llambda = 1e-3)
for t in range(10):
    opt.zero_grad()
    loss = (weights**2).mean()
    print(loss.cpu().item())
    loss.backward()
    opt.step()