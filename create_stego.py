from torch.optim import LBFGS
from tqdm import tqdm
from models.extractor import FixedMessageExtractor
from config import Config

def create_stego_point_cloud(model, coords, secret_message):
    """Create stego point cloud by adding perturbations"""
    extractor = FixedMessageExtractor().to(Config.DEVICE)
    stego_coords = coords.detach().clone().requires_grad_(True)
    
    optimizer = LBFGS([stego_coords], lr=Config.STEGO_LR, max_iter=10)
    
    pbar = tqdm(range(Config.STEGO_ITERATIONS), desc="Creating Stego Point Cloud")
    for i in pbar:
        def closure():
            optimizer.zero_grad()
            colors = model(stego_coords)
            extracted = extractor(colors).squeeze()
            loss = nn.BCELoss()(extracted, secret_message.float())
            loss.backward()
            return loss
        
        optimizer.step(closure)
        pbar.set_postfix({'loss': closure().item()})
    
    return stego_coords.detach()