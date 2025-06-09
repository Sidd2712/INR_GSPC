import torch

def calculate_accuracy(extracted_msg, secret_message):
    """Calculate extraction accuracy"""
    return (extracted_msg == secret_message).float().mean().item()

def calculate_psnr(original, stego):
    """Calculate PSNR between original and stego coordinates"""
    mse = torch.mean((original - stego) ** 2)
    max_val = torch.max(original)
    return (20 * torch.log10(max_val / torch.sqrt(mse))).item()