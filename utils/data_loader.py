import numpy as np
import torch
from PIL import Image
from config import Config

def load_and_sample_image(image_path):
    """Load image and sample point cloud"""
    # Load and resize image
    img = Image.open(image_path).convert('RGB')
    img = img.resize(Config.IMG_SIZE)
    img_array = np.array(img) / 255.0  # Normalize to [0,1]
    
    # Sample points
    h, w = img_array.shape[:2]
    coords = np.random.rand(Config.NUM_POINTS, 2) * np.array([w-1, h-1])
    colors = []
    
    # Get colors with bilinear interpolation
    for x, y in coords:
        x0, y0 = int(x), int(y)
        x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)
        
        # Bilinear interpolation
        x_frac, y_frac = x - x0, y - y0
        top = img_array[y0, x0] * (1 - x_frac) + img_array[y0, x1] * x_frac
        bottom = img_array[y1, x0] * (1 - x_frac) + img_array[y1, x1] * x_frac
        color = top * (1 - y_frac) + bottom * y_frac
        colors.append(color)
    
    return (torch.FloatTensor(coords).to(Config.DEVICE), 
            torch.FloatTensor(np.array(colors)).to(Config.DEVICE),
            img_array)