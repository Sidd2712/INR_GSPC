# Configuration for the project
class Config:
    # Device configuration
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Model parameters
    INPUT_DIM = 2          # For 2D coordinates
    HIDDEN_DIM = 64        # Reduced for CPU
    OUTPUT_DIM = 3         # RGB output
    
    # Training parameters
    INR_EPOCHS = 1000
    INR_LR = 1e-4
    
    # Stego parameters
    STEGO_ITERATIONS = 50
    STEGO_LR = 0.01
    
    # Data parameters
    IMG_SIZE = (32, 32)    # Reduced for CPU
    NUM_POINTS = 500       # Reduced for CPU
    SEED = 42