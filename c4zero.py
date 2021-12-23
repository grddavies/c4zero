import torch
from torch import nn
import torch.nn.functional as F

import numpy as np


class ConvBlock(nn.Module):
    """Convolutional Block"""

    def __init__(self, action_size) -> None:
        super(ConvBlock, self).__init__()
        self.action_size = action_size
        self.conv1 = nn.Conv2d(3, 128, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(128)

    def forward(self, s: torch.FloatTensor):
        s = s.view(-1, 3, 6, 7)  # batch_size x channels x board_x x board_y
        s = F.relu(self.bn1(self.conv1(s)))
        return s


class ResBlock(nn.Module):
    """Residual Block"""

    def __init__(self, inplanes=128, planes=128, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


class OutBlock(nn.Module):
    """Output Block"""

    def __init__(self):
        super(OutBlock, self).__init__()
        self.conv = nn.Conv2d(128, 3, kernel_size=1)  # value head
        self.bn = nn.BatchNorm2d(3)
        self.fc1 = nn.Linear(3 * 6 * 7, 32)
        self.fc2 = nn.Linear(32, 1)

        self.conv1 = nn.Conv2d(128, 32, kernel_size=1)  # policy head
        self.bn1 = nn.BatchNorm2d(32)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.fc = nn.Linear(6 * 7 * 32, 7)

    def forward(self, s):
        v = F.relu(self.bn(self.conv(s)))  # value head
        v = v.view(-1, 3 * 6 * 7)  # batch_size X channel X height X width
        v = F.relu(self.fc1(v))
        v = torch.tanh(self.fc2(v))

        p = F.relu(self.bn1(self.conv1(s)))  # policy head
        p = p.view(-1, 6 * 7 * 32)
        p = self.fc(p)
        p = self.logsoftmax(p).exp()
        return p, v


class C4Zero(nn.Module):
    """Connect 4 solving network"""

    def __init__(self):
        super(C4Zero, self).__init__()
        self.conv = ConvBlock(7)  # Action size
        for block in range(19):
            setattr(self, "res_%i" % block, ResBlock())
        self.outblock = OutBlock()

    def forward(self, s):
        # Pass through conv block
        s = self.conv(s)
        # Pass through each residual block in order
        for block in range(19):
            s = getattr(self, "res_%i" % block)(s)
        # Pass through output block
        s = self.outblock(s)
        return s

    def predict(self, X: np.ndarray):
        s = torch.FloatTensor(X)
        self.eval()
        policy, value = self.forward(s)
        return policy.data.cpu().numpy()[0], value.data.cpu().numpy()[0]


class AlphaLoss(nn.Module):
    """Sum of the mean-squared error value and cross-entropy policy losses"""

    def __init__(self):
        super(AlphaLoss, self).__init__()

    def forward(self, y_value, value, y_policy, policy):
        value_error = (value - y_value) ** 2
        policy_error = torch.sum((-policy * (1e-8 + y_policy.float()).float().log()), 1)
        total_error = (value_error.view(-1).float() + policy_error).mean()
        return total_error
