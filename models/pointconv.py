import torch
import torch.nn as nn
import torch.nn.functional as F

class PointConv(nn.Module):
    def __init__(self, in_channels, out_channels, num_points=1024):
        super(PointConv, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, 1)
        self.bn = nn.BatchNorm1d(out_channels)
        self.num_points = num_points

    def forward(self, x, coords):
        # x: (batch_size, in_channels, num_points)
        # coords: (batch_size, num_points, 2 or 3)
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        return x