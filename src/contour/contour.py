"""
Calculating the differentiable edge detection and contour based loss
"""
import numpy as np
import cv2
import torch
from torch import nn
from torchvision.transforms import GaussianBlur
import torch
from scipy.signal.windows import gaussian


device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


# https://github.com/chaddy1004/sobel-operator-pytorch/blob/master/model.py
class Sobel(nn.Module):
    def __init__(self):
        super().__init__()
        self.filter1 = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=1, bias=False)

        Gx = torch.tensor([[-2.0, 0.0, 2.0], [-4.0, 0.0, 4.0], [-2.0,  0.0,  2.0]], device=device)
        Gy = torch.tensor([[ 2.0, 4.0, 2.0], [ 0.0, 0.0, 0.0], [-2.0, -4.0, -2.0]], device=device)
        G = torch.cat([Gx.unsqueeze(0), Gy.unsqueeze(0)], 0)
        G = G.unsqueeze(1)
        self.filter1.weight = nn.Parameter(G, requires_grad=False)

    def forward(self, img):
        x = self.filter1(img)
        x = torch.abs(x)
        # x = torch.where(x >= 1e-6, x, torch.ones_like(x) * 1e-6)
        # x = torch.mul(x, x)
        x = torch.sum(x, dim=1, keepdim=True)
        # x = torch.sqrt(x)

        return x
