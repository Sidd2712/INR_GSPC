import torch.nn as nn
from config import Config

class FixedMessageExtractor(nn.Module):
    """Fixed neural network for message extraction"""
    def __init__(self, input_dim=3, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Initialize with fixed random weights
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize with fixed random weights"""
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                torch.manual_seed(Config.SEED)
                nn.init.normal_(layer.weight, mean=0.0, std=0.02)
                nn.init.constant_(layer.bias, 0.0)
    
    def forward(self, x):
        return self.net(x)