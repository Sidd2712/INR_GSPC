import torch
import torch.nn as nn
from utils.data_loader import load_and_sample_image
from utils.visualization import visualize_point_clouds
from utils.metrics import calculate_accuracy, calculate_psnr
from train_inr import train_inr_model
from create_stego import create_stego_point_cloud
from models.extractor import FixedMessageExtractor
from config import Config

def main():
    # Set random seed for reproducibility
    torch.manual_seed(Config.SEED)
    
    # 1. Load and sample image
    coords, colors, original_img = load_and_sample_image("example.jpg")
    
    # 2. Train INR model
    inr_model = train_inr_model(coords, colors)
    
    # 3. Generate random secret message
    secret_message = torch.randint(0, 2, (Config.NUM_POINTS,)).to(Config.DEVICE)
    
    # 4. Create stego point cloud
    stego_coords = create_stego_point_cloud(inr_model, coords, secret_message)
    
    # 5. Extract message
    extractor = FixedMessageExtractor().to(Config.DEVICE)
    extracted_msg = extract_message(inr_model, stego_coords, extractor)
    
    # 6. Evaluate
    accuracy = calculate_accuracy(extracted_msg, secret_message)
    psnr = calculate_psnr(coords, stego_coords)
    
    print(f"\nMessage extraction accuracy: {accuracy*100:.2f}%")
    print(f"PSNR between original and stego coordinates: {psnr:.2f} dB")
    
    # 7. Visualize
    visualize_point_clouds(coords, stego_coords, original_img)

def extract_message(model, stego_coords, extractor):
    """Extract message from stego point cloud"""
    with torch.no_grad():
        colors = model(stego_coords)
        extracted = extractor(colors).squeeze()
        return (extracted > 0.5).float()

if __name__ == "__main__":
    main()