"""
Author: Sai Manohar Vemuri
Institute: Illinois Institute of Technology
Date: 04/14/2025
Email: svemuri6@hawk.iit.edu
"""
import torch
import torch.nn as nn

def sigmoid_base2(x, k=1.44):
    return 1 / (1 + torch.pow(2.0, -k * x))

class SwishApproxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        sigmoid_approx = sigmoid_base2(i)
        ctx.save_for_backward(i, sigmoid_approx)
        return i * sigmoid_approx

    @staticmethod
    def backward(ctx, grad_output):
        i, sigmoid_approx = ctx.saved_tensors
        grad_input = grad_output * (sigmoid_approx + i * sigmoid_approx * torch.log(torch.tensor(2.0)) * 1.44 * (1 - sigmoid_approx))
        return grad_input

class SwishApprox(nn.Module):
    def forward(self, x):
        return SwishApproxFunction.apply(x)

#just wrapper cn remove if needed
class AdaptiveSwishApprox(nn.Module):
    def __init__(self):
        super().__init__()
        self.swish = SwishApprox()

    def forward(self, x):
        return self.swish(x)
    


#readme:
#swish=AdaptiveSwishApprox()
#swish(x)


