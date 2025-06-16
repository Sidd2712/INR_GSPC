import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.function_generator import FunctionGenerator, Discriminator, r1_regularization
from models.message_extractor import PointCloudMessageExtractor
from models.utils import random_fourier_features, compute_accuracy, compute_psnr, compute_ssim
import numpy as np
from scipy.optimize import minimize
import uuid

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = 256
hidden_dim = 512
message_length = 128
num_points = 1024
batch_size = 32
epochs = 10000
lr_generator = 1e-4
lr_discriminator = 1e-4

# Dataset (CelebA-HQ)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])
dataset = datasets.CelebA("./data/celeba_hq", split="train", transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Models
generator = FunctionGenerator(latent_dim, hidden_dim).to(device)
discriminator = Discriminator().to(device)
extractor = PointCloudMessageExtractor(message_length=message_length).to(device)

# Optimizers
opt_g = optim.Adam(generator.parameters(), lr=lr_generator)
opt_d = optim.Adam(discriminator.parameters(), lr=lr_discriminator)

# Fixed extractor (shared seed)
torch.manual_seed(42)
extractor.eval()

def generate_point_cloud(img, num_points=1024):
    # Convert image to point cloud
    b, c, h, w = img.shape
    coords = torch.stack(torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w)), dim=-1)
    coords = coords.view(-1, 2).to(device)
    values = img.permute(0, 2, 3, 1).view(b, -1, c)
    indices = torch.randperm(coords.shape[0])[:num_points]
    return coords[indices].expand(b, -1, -1), values[:, indices]

def optimize_point_cloud(point_cloud, secret_msg, extractor, max_iter=1000):
    # Optimize point cloud perturbations using L-BFGS
    point_cloud = point_cloud.clone().requires_grad_(True)
    optimizer = torch.optim.LBFGS([point_cloud], lr=0.03, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        ext_msg = extractor(point_cloud)
        loss = F.binary_cross_entropy(ext_msg, secret_msg)
        loss.backward()
        return loss

    optimizer.step(closure)
    return point_cloud.detach()

# Training Loop
for epoch in range(epochs):
    for real_imgs, _ in dataloader:
        real_imgs = real_imgs.to(device)
        batch_size = real_imgs.shape[0]

        # Generate point cloud
        coords, real_values = generate_point_cloud(real_imgs, num_points)

        # Train Discriminator
        opt_d.zero_grad()
        z = torch.randn(batch_size, latent_dim, device=device)
        gen_values = generator(z, random_fourier_features(coords))
        real_out = discriminator(real_values)
        gen_out = discriminator(gen_values)
        d_loss = F.binary_cross_entropy(real_out, torch.ones_like(real_out)) + \
                 F.binary_cross_entropy(gen_out, torch.zeros_like(gen_out))
        r1_loss = r1_regularization(discriminator, real_values)
        d_loss = d_loss + 10 * r1_loss
        d_loss.backward()
        opt_d.step()

        # Train Generator
        opt_g.zero_grad()
        z = torch.randn(batch_size, latent_dim, device=device)
        gen_values = generator(z, random_fourier_features(coords))
        gen_out = discriminator(gen_values)
        g_loss = F.binary_cross_entropy(gen_out, torch.ones_like(gen_out))
        g_loss.backward()
        opt_g.step()

        # Generate Stego Point Cloud
        secret_msg = (torch.rand(batch_size, message_length, device=device) > 0.5).float()
        stego_points = optimize_point_cloud(real_values, secret_msg, extractor)

        # Evaluate
        ext_msg = extractor(stego_points)
        accuracy = compute_accuracy(secret_msg, ext_msg)
        psnr_value = compute_psnr(real_imgs, stego_points.view(batch_size, 128, 128, 3).permute(0, 3, 1, 2))
        ssim_value = compute_ssim(real_imgs, stego_points.view(batch_size, 128, 128, 3).permute(0, 3, 1, 2))

        print(f"Epoch [{epoch+1}/{epochs}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}, "
              f"Accuracy: {accuracy.item():.4f}, PSNR: {psnr_value:.4f}, SSIM: {ssim_value:.4f}")

torch.save(generator.state_dict(), "generator.pth")
torch.save(discriminator.state_dict(), "discriminator.pth")