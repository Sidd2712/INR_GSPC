import torch.nn as nn
from config import Config

class INR_MLP(nn.Module):
    """Implicit Neural Representation using MLP"""
    def __init__(self, input_dim=Config.INPUT_DIM, 
                 hidden_dim=Config.HIDDEN_DIM, 
                 output_dim=Config.OUTPUT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()  # For RGB values in [0,1]
        )
    
    def forward(self, x):
        return self.net(x)