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

# --- 1. File and Directory Setup ---
# Upload modelnet40.zip and unzip
uploaded = files.upload()

os.makedirs('/content/data/modelnet40', exist_ok=True)
with zipfile.ZipFile('modelnet40.zip', 'r') as zip_ref:
    zip_ref.extractall('/content/data')

# Verify structure
!ls /content/data/modelnet40

# --- 2. Install Dependencies and Hyperparameter Configuration ---
!pip install torch==2.5.1 torchvision==0.20.1 numpy scikit-image scipy
!pip install open3d

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = 256
hidden_dim = 512
message_length = 128
num_points = 1024
batch_size = 16  # Reduced batch size for 16GB GPU
epochs = 1000
lr_generator = 1e-4
lr_discriminator = 1e-4
max_iter_lbfgs = 200 # Reduced L-BFGS iterations for efficiency
checkpoint_dir = "/content/checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

# --- 3. Model Architecture and Helper Functions ---
class FunctionGenerator(nn.Module):
    def __init__(self, latent_dim=256, hidden_dim=512, output_dim=3):
        super(FunctionGenerator, self).__init__()
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
    # Select a random subset of points to reduce memory
    if real_points.size(1) > 1024:
        indices = torch.randperm(real_points.size(1))[:1024]
        real_points = real_points[:, indices, :]
    real_points.requires_grad_(True)
    real_out = discriminator(real_points)
    grad = torch.autograd.grad(outputs=real_out.sum(), inputs=real_points, create_graph=True)[0]
    r1_loss = 0.5 * (grad.norm(2, dim=-1) ** 2).mean()
    return r1_loss

class PointConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PointConv, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, 1)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x, coords):
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
        x = self.conv1(x, point_cloud)
        x = self.conv2(x, point_cloud)
        x = x.mean(dim=2)
        return torch.sigmoid(self.fc(x))

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
    indices = torch.randperm(coords.shape[1])[:num_points]
    return coords[:, indices], values[:, indices]

def optimize_point_cloud(point_cloud, secret_msg, extractor, max_iter=200):
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

# --- 4. Dataset, Dataloader and Model Instantiation ---
class ModelNet40Dataset(Dataset):
    def __init__(self, root_dir, split='train'):
        self.files = glob.glob(os.path.join(root_dir, '**', split, '*.off'), recursive=True)
        if not self.files:
            raise FileNotFoundError(f"No .off files found in {os.path.join(root_dir, '**', split)}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        point_cloud = o3d.io.read_point_cloud(path)
        points = np.asarray(point_cloud.points)
        if len(points) < 1024:
            # Pad with zeros if point cloud is too small
            padded_points = np.zeros((1024, 3), dtype=np.float32)
            padded_points[:len(points), :] = points
            points = padded_points
        points = (points - points.min()) / (points.max() - points.min()) * 2 - 1
        return torch.tensor(points, dtype=torch.float32)

try:
    dataset = ModelNet40Dataset('/content/data/modelnet40', split='train')
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

# Checkpoint loading logic
start_epoch = 0
checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint.pth')
if os.path.exists(checkpoint_path):
    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    opt_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
    opt_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Resuming training from epoch {start_epoch}")

# --- 5. Training Loop ---
for epoch in range(start_epoch, epochs):
    for i, real_points in enumerate(dataloader):
        real_points = real_points.to(device)
        batch_size = real_points.shape[0]

        if real_points.numel() == 0:
            continue

        coords, real_values = generate_point_cloud(real_points, num_points)

        # Train Discriminator
        opt_d.zero_grad()
        z = torch.randn(batch_size, latent_dim, device=device)
        gen_values = generator(z, random_fourier_features(coords.detach()))
        real_out = discriminator(real_values)
        gen_out = discriminator(gen_values.detach())
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

        # Steganography Process
        secret_msg = (torch.rand(batch_size, message_length, device=device) > 0.5).float()
        stego_points = optimize_point_cloud(real_values.clone(), secret_msg, extractor, max_iter=max_iter_lbfgs)
        ext_msg = extractor(stego_points)
        accuracy = compute_accuracy(secret_msg, ext_msg)

        if (i + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(dataloader)}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}, Accuracy: {accuracy.item():.4f}")

    # Save checkpoint at the end of each epoch
    torch.save({
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_g_state_dict': opt_g.state_dict(),
        'optimizer_d_state_dict': opt_d.state_dict(),
    }, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch + 1}")

# --- 6. Save and Download Final Models ---
torch.save(generator.state_dict(), "/content/generator.pth")
torch.save(discriminator.state_dict(), "/content/discriminator.pth")

files.download('/content/generator.pth')
files.download('/content/discriminator.pth')

# --- 7. Evaluation and Visualization (Optional) ---
def visualize_point_cloud(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.cpu().numpy())
    o3d.visualization.draw_geometries([pcd])

print("Running final evaluation...")
generator.eval()
extractor.eval()
test_dataset = ModelNet40Dataset('/content/data/modelnet40', split='test')
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
accuracies = []

with torch.no_grad():
    for real_points in test_dataloader:
        real_points = real_points.to(device)
        if real_points.numel() == 0:
            continue

        coords, real_values = generate_point_cloud(real_points, num_points)
        secret_msg = (torch.rand(real_points.shape[0], message_length, device=device) > 0.5).float()

        stego_points = optimize_point_cloud(real_values.clone(), secret_msg, extractor, max_iter=max_iter_lbfgs)
        ext_msg = extractor(stego_points)
        accuracy = compute_accuracy(secret_msg, ext_msg)
        accuracies.append(accuracy.item())

print(f"Average Accuracy: {sum(accuracies) / len(accuracies):.4f}")

if len(stego_points) > 0:
    print("Visualizing a sample stego-point cloud...")
    visualize_point_cloud(stego_points[0])