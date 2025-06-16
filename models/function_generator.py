import torch
import torch.nn as nn
import torch.nn.functional as F

class FunctionGenerator(nn.Module):
    def __init__(self, latent_dim=256, hidden_dim=512, output_dim=3):
        super(FunctionGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # MLP for generating function weights
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * output_dim),
        )

    def forward(self, z, coords):
        # z: latent vector (batch_size, latent_dim)
        # coords: point cloud coordinates (batch_size, num_points, 2 or 3)
        weights = self.mlp(z)  # (batch_size, hidden_dim * output_dim)
        weights = weights.view(-1, self.hidden_dim, self.output_dim)  # (batch_size, hidden_dim, output_dim)

        # Apply function to coordinates
        out = torch.matmul(coords, weights)  # (batch_size, num_points, output_dim)
        return out

class Discriminator(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=512):
        super(Discriminator, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, point_cloud):
        # point_cloud: (batch_size, num_points, input_dim)
        return self.mlp(point_cloud).mean(dim=1)  # (batch_size, 1)

def r1_regularization(discriminator, real_points):
    # R1 regularization for point cloud discriminator
    real_points.requires_grad_(True)
    real_out = discriminator(real_points)
    grad = torch.autograd.grad(outputs=real_out.sum(), inputs=real_points, create_graph=True)[0]
    r1_loss = 0.5 * (grad.norm(2, dim=-1) ** 2).mean()
    return r1_loss