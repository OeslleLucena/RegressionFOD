import torch
import torch.nn as nn

class L2Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(size_average=False)
    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))

class L2Loss_niftynet(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, yhat, y):
        return torch.sum((yhat-y).pow(2))/2
