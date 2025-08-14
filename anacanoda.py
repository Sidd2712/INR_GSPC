import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import open3d as o3d
import os
import glob
import zipfile

# =========================
# 1. File and Directory Setup
# =========================
DATASET_DIR = "data/modelnet40"
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# =========================
# 2. Device + Hyperparameters
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = 256
hidden_dim = 512
message_length = 128
num_points = 1024
batch_size = 16
epochs = 10
lr_generator = 1e-4
lr_discriminator = 1e-4
stego_optimization_steps = 200
stego_lr = 0.03

# =========================
# 3. Model Architectures
# =========================
class FunctionGenerator(nn.Module):
    def __init__(self, latent_dim=256, hidden_dim=512, output_dim=3):
        super(FunctionGenerator, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * output_dim),
        )

    def forward(self, z, coords):
        weights = self.mlp(z).view(-1, self.hidden_dim, self.output_dim)
        out = torch.matmul(coords, weights)
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
        return self.mlp(point_cloud).mean(dim=1)

def r1_regularization(discriminator, real_points):
    if real_points.size(1) > 1024:
        indices = torch.randperm(real_points.size(1))[:1024].to(real_points.device)
        real_points = real_points[:, indices, :]
    
    real_points.requires_grad_(True)
    real_out = discriminator(real_points)
    grad = torch.autograd.grad(outputs=real_out.sum(), inputs=real_points, create_graph=True, retain_graph=True)[0]
    r1_loss = 0.5 * (grad.norm(2, dim=-1) ** 2).mean()
    return r1_loss

class PointConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PointConv, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, 1)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x)

class PointCloudMessageExtractor(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=512, message_length=128):
        super(PointCloudMessageExtractor, self).__init__()
        self.conv1 = PointConv(input_dim, hidden_dim)
        self.conv2 = PointConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, message_length)

    def forward(self, point_cloud):
        x = point_cloud.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.mean(dim=2)
        return torch.sigmoid(self.fc(x))

# =========================
# 4. Helper Functions
# =========================
def random_fourier_features(coords, num_features=256):
    B = torch.randn(coords.shape[-1], num_features // 2, device=coords.device) * 10
    coords = coords @ B
    return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

def compute_accuracy(real_msg, ext_msg):
    ber = (real_msg != (ext_msg > 0.5).float()).float().mean()
    return 1 - ber

def generate_point_cloud(data, num_points=1024):
    b, n, d = data.shape
    coords = data[:, :, :3]
    values = data[:, :, :3]
    indices = torch.randperm(coords.shape[1])[:num_points].to(coords.device)
    return coords[:, indices], values[:, indices]

def optimize_point_cloud(point_cloud, secret_msg, extractor, steps=200, lr=0.03):
    point_cloud = point_cloud.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([point_cloud], lr=lr)

    for _ in range(steps):
        optimizer.zero_grad()
        ext_msg = extractor(point_cloud)
        loss = F.binary_cross_entropy(ext_msg, secret_msg)
        loss.backward()
        optimizer.step()
    return point_cloud.detach()

# =========================
# 5. Dataset Loader
# =========================
class ModelNet40Dataset(Dataset):
    def __init__(self, root_dir, split='train'):
        self.files = glob.glob(os.path.join(root_dir, '*', split, '*.off'))
        if not self.files:
            raise FileNotFoundError(f"No .off files found in {root_dir}")
        self.num_files = len(self.files)

    def __len__(self):
        return self.num_files

    def __getitem__(self, idx):
        file_path = self.files[idx]
        attempts = 0
        while attempts < self.num_files:
            try:
                mesh = o3d.io.read_triangle_mesh(file_path)

                if not mesh.has_vertices() or not mesh.has_triangles():
                    raise ValueError("Mesh has no vertices or triangles.")

                points = mesh.sample_points_uniformly(number_of_points=num_points)
                points = np.asarray(points.points, dtype=np.float32)
                points_min = points.min(axis=0)
                points_max = points.max(axis=0)
                points = 2 * (points - points_min) / (points_max - points_min + 1e-8) - 1
                
                return torch.from_numpy(points)
            except Exception as e:
                print(f"Error loading file {file_path}: {e}. Skipping...")
                idx = (idx + 1) % self.num_files
                file_path = self.files[idx]
                attempts += 1
        
        # If all files fail after trying them all, raise a final error.
        raise RuntimeError("Could not find a single valid file in the dataset.")


# =========================
# 6. Data + Models
# =========================
try:
    dataset = ModelNet40Dataset(DATASET_DIR, split='train')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
except FileNotFoundError as e:
    print(e)
    exit()

generator = FunctionGenerator(latent_dim, hidden_dim).to(device)
discriminator = Discriminator().to(device)
extractor = PointCloudMessageExtractor(message_length=message_length).to(device)

opt_g = optim.Adam(generator.parameters(), lr=lr_generator)
opt_d = optim.Adam(discriminator.parameters(), lr=lr_discriminator)

torch.manual_seed(42)
extractor.eval()

start_epoch = 0
checkpoint_path = os.path.join(CHECKPOINT_DIR, 'checkpoint.pth')
if os.path.exists(checkpoint_path):
    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    opt_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
    opt_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Resuming training from epoch {start_epoch}")

# =========================
# 7. Training Loop
# =========================
for epoch in range(start_epoch, epochs):
    for i, real_points in enumerate(dataloader):
        real_points = real_points.to(device)
        if real_points.numel() == 0:
            continue

        current_batch_size = real_points.shape[0]
        coords, real_values = generate_point_cloud(real_points, num_points)
        
        opt_d.zero_grad()
        z = torch.randn(current_batch_size, latent_dim, device=device)
        gen_values = generator(z, random_fourier_features(coords.detach()))
        
        real_out = discriminator(real_values)
        gen_out = discriminator(gen_values.detach())
        d_loss = F.binary_cross_entropy(real_out, torch.ones_like(real_out)) + \
                 F.binary_cross_entropy(gen_out, torch.zeros_like(gen_out))
        
        r1_loss_val = r1_regularization(discriminator, real_values)
        d_loss = d_loss + 10 * r1_loss_val
        d_loss.backward()
        opt_d.step()

        opt_g.zero_grad()
        z = torch.randn(current_batch_size, latent_dim, device=device)
        gen_values = generator(z, random_fourier_features(coords))
        gen_out = discriminator(gen_values)
        g_loss = F.binary_cross_entropy(gen_out, torch.ones_like(gen_out))
        g_loss.backward()
        opt_g.step()

        secret_msg = (torch.rand(current_batch_size, message_length, device=device) > 0.5).float()
        stego_points = optimize_point_cloud(real_values.clone(), secret_msg, extractor, steps=stego_optimization_steps, lr=stego_lr)
        ext_msg = extractor(stego_points)
        accuracy = compute_accuracy(secret_msg, ext_msg)

        if (i + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(dataloader)}], "
                  f"D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}, Accuracy: {accuracy.item():.4f}")

    torch.save({
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_g_state_dict': opt_g.state_dict(),
        'optimizer_d_state_dict': opt_d.state_dict(),
    }, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch + 1}")

# =========================
# 8. Save Final Models
# =========================
torch.save(generator.state_dict(), "generator.pth")
torch.save(discriminator.state_dict(), "discriminator.pth")

# =========================
# 9. Evaluation
# =========================
print("Running final evaluation...")
try:
    test_dataset = ModelNet40Dataset(DATASET_DIR, split='test')
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    accuracies = []

    generator.eval()
    extractor.eval()
    with torch.no_grad():
        for real_points in test_dataloader:
            real_points = real_points.to(device)
            if real_points.numel() == 0:
                continue
            
            coords, real_values = generate_point_cloud(real_points, num_points)
            secret_msg = (torch.rand(real_points.shape[0], message_length, device=device) > 0.5).float()
            
            stego_points = optimize_point_cloud(real_values.clone(), secret_msg, extractor, steps=stego_optimization_steps, lr=stego_lr)
            ext_msg = extractor(stego_points)
            accuracy = compute_accuracy(secret_msg, ext_msg)
            accuracies.append(accuracy.item())

    print(f"Average Accuracy: {sum(accuracies) / len(accuracies):.4f}")
except FileNotFoundError as e:
    print(f"Skipping evaluation: {e}")

# =========================
# 10. Visualization (requires an interactive environment)
# =========================
def visualize_point_cloud(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.cpu().numpy())
    o3d.visualization.draw_geometries([pcd])

# The following visualization code will only work in a graphical environment,
# which is not available in most remote or terminal setups.
# if 'stego_points' in locals() and len(stego_points) > 0:
#     print("Visualizing a sample stego-point cloud...")
#     visualize_point_cloud(stego_points[0])
