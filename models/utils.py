import torch
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import numpy as np

def random_fourier_features(coords, num_features=256):
    # Random Fourier Feature encoding for high-frequency information
    B = torch.randn(coords.shape[-1], num_features // 2, device=coords.device) * 10
    coords = coords @ B
    coords = torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)
    return coords

def compute_accuracy(real_msg, ext_msg):
    # Compute message extraction accuracy
    ber = (real_msg != (ext_msg > 0.5).float()).float().mean()
    return 1 - ber

def compute_psnr(img1, img2):
    img1_np = img1.cpu().numpy()
    img2_np = img2.cpu().numpy()
    return psnr(img1_np, img2_np, data_range=1.0)

def compute_ssim(img1, img2):
    img1_np = img1.cpu().numpy()
    img2_np = img2.cpu().numpy()
    return ssim(img1_np, img2_np, data_range=1.0, multichannel=True)