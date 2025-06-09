import torch.optim as optim
from tqdm import tqdm
from models.inr_mlp import INR_MLP
from config import Config

def train_inr_model(coords, colors):
    """Train the INR model to represent the image"""
    model = INR_MLP().to(Config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=Config.INR_LR)
    criterion = nn.MSELoss()
    
    pbar = tqdm(range(Config.INR_EPOCHS), desc="Training INR Model")
    for epoch in pbar:
        optimizer.zero_grad()
        outputs = model(coords)
        loss = criterion(outputs, colors)
        loss.backward()
        optimizer.step()
        
        pbar.set_postfix({'loss': loss.item()})
    
    return model