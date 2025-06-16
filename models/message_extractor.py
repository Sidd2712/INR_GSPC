import torch
import torch.nn as nn
import torch.nn.functional as F
from models.pointconv import PointConv

class PointCloudMessageExtractor(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=512, message_length=128):
        super(PointCloudMessageExtractor, self).__init__()
        self.conv1 = PointConv(input_dim, hidden_dim)
        self.conv2 = PointConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, message_length)

    def forward(self, point_cloud):
        # point_cloud: (batch_size, num_points, input_dim)
        x = point_cloud.permute(0, 2, 1)  # (batch_size, input_dim, num_points)
        x = self.conv1(x, point_cloud)
        x = self.conv2(x, point_cloud)
        x = x.mean(dim=2)  # Global average pooling
        x = self.fc(x)
        return torch.sigmoid(x)  # (batch_size, message_length)