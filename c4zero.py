from typing import Tuple
import torch
from torch import nn
from torch.functional import Tensor
import torch.nn.functional as F

import numpy as np


class ConvBlock(nn.Module):
    """Convolutional Block"""

    def __init__(self, n_channels, in_x, in_y) -> None:
        super(ConvBlock, self).__init__()
        self.n_channels = n_channels
        self.in_x, self.in_y = in_x, in_y
        self.conv1 = nn.Conv2d(n_channels, 128, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(128)

    def forward(self, s: torch.FloatTensor):
        # batch_size * n_channels * width * height
        s = s.view(-1, self.n_channels, self.in_x, self.in_y)
        return F.relu(self.bn1(self.conv1(s)))


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

    def __init__(self, n_channels, in_x, in_y, action_size):
        super(OutBlock, self).__init__()
        self.n_channels, self.in_x, self.in_y = n_channels, in_x, in_y
        self.conv = nn.Conv2d(128, 1, kernel_size=1)  # value head
        self.bn = nn.BatchNorm2d(1)
        self.fc1 = nn.Linear(1 * in_x * in_y, 32)
        self.fc2 = nn.Linear(32, 1)

        self.conv1 = nn.Conv2d(128, 32, kernel_size=1)  # policy head
        self.bn1 = nn.BatchNorm2d(32)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.fc = nn.Linear(in_x * in_y * 32, action_size)

    def forward(self, s: torch.Tensor) -> Tuple[Tensor, Tensor]:
        # batch_size * n_channels * width * height
        v = F.relu(self.bn(self.conv(s)))  # value head
        v = v.view(-1, self.n_channels * self.in_x * self.in_y)
        v = F.relu(self.fc1(v))
        v = torch.tanh(self.fc2(v))

        p = F.relu(self.bn1(self.conv1(s)))  # policy head
        p = p.view(-1, self.in_x * self.in_y * 32)
        p = self.fc(p)
        p = self.logsoftmax(p).exp()
        return p, v


class C4Zero(nn.Module):
    """Connect-N solving network"""

    def __init__(self, n_channels, in_x, in_y, action_size, device="cpu"):
        super(C4Zero, self).__init__()
        self.device = device
        self.conv = ConvBlock(n_channels, in_x, in_y)
        for block in range(19):
            setattr(self, "res_%i" % block, ResBlock())
        self.outblock = OutBlock(n_channels, in_x, in_y, action_size)

    def forward(self, s: torch.Tensor):
        # Pass through conv block
        s = self.conv(s)
        # Pass through each residual block in order
        for block in range(19):
            s = getattr(self, "res_%i" % block)(s)
        # Pass through output block
        return self.outblock(s)

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        s = torch.FloatTensor(X).to(self.device)
        self.eval()
        with torch.no_grad():
            policy, value = self.forward(s)
        return policy.numpy(), value.numpy()


class AlphaLoss(nn.Module):
    """Sum of the mean-squared error value and cross-entropy policy losses"""

    def __init__(self):
        super(AlphaLoss, self).__init__()

    def forward(self, y_value, value, y_policy, policy):
        value_error = (value - y_value) ** 2
        policy_error = torch.sum((-policy * (1e-8 + y_policy.float()).float().log()), 1)
        total_error = (value_error.view(-1).float() + policy_error).mean()
        return total_error


class Connect2Model(nn.Module):
    def __init__(self, in_x, in_y, action_size, device):

        super(Connect2Model, self).__init__()

        self.device = device
        self.in_x = in_x
        self.in_y = in_y
        self.action_size = action_size

        self.fc1 = nn.Linear(in_features=self.in_x * self.in_y, out_features=16)
        self.fc2 = nn.Linear(in_features=16, out_features=16)

        # Two heads on our network
        self.action_head = nn.Linear(in_features=16, out_features=self.action_size)
        self.value_head = nn.Linear(in_features=16, out_features=1)

        self.to(device)

    def forward(self, x):
        x = x.view(-1, 1, self.in_x * self.in_y)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        action_logits = self.action_head(x)
        value_logit = self.value_head(x)

        return F.softmax(action_logits, dim=-1), torch.tanh(value_logit)

    def predict(self, board):
        board = torch.FloatTensor(board.astype(np.float32)).to(self.device)
        board = board.view(1, 1, self.in_x * self.in_y)
        self.eval()
        with torch.no_grad():
            pi, v = self.forward(board)

        return pi.data.cpu().numpy()[0], v.data.cpu().numpy()[0]
