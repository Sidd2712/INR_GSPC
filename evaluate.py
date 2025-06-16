import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.function_generator import FunctionGenerator
from models.message_extractor import PointCloudMessageExtractor
from models.utils import generate_point_cloud, optimize_point_cloud, compute_accuracy, compute_psnr, compute_ssim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = 256
hidden_dim = 512
message_length = 128
num_points = 1024
batch_size = 32

# Load Models
generator = FunctionGenerator(latent_dim, hidden_dim).to(device)
generator.load_state_dict(torch.load("generator.pth"))
generator.eval()
extractor = PointCloudMessageExtractor(message_length=message_length).to(device)
torch.manual_seed(42)
extractor.eval()

# Dataset
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])
dataset = datasets.CelebA("./data/celeba_hq", split="test", transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# Evaluation
accuracies, psnrs, ssims = [], [], []
for real_imgs, _ in dataloader:
    real_imgs = real_imgs.to(device)
    coords, real_values = generate_point_cloud(real_imgs, num_points)
    secret_msg = (torch.rand(real_imgs.shape[0], message_length, device=device) > 0.5).float()
    stego_points = optimize_point_cloud(real_values, secret_msg, extractor)
    ext_msg = extractor(stego_points)
    accuracy = compute_accuracy(secret_msg, ext_msg)
    psnr_value = compute_psnr(real_imgs, stego_points.view(real_imgs.shape[0], 128, 128, 3).permute(0, 3, 1, 2))
    ssim_value = compute_ssim(real_imgs, stego_points.view(real_imgs.shape[0], 128, 128, 3).permute(0, 3, 1, 2))
    accuracies.append(accuracy.item())
    psnrs.append(psnr_value)
    ssims.append(ssim_value)

print(f"Average Accuracy: {sum(accuracies)/len(accuracies):.4f}")
print(f"Average PSNR: {sum(psnrs)/len(psnrs):.4f}")
print(f"Average SSIM: {sum(ssims)/len(ssims):.4f}")