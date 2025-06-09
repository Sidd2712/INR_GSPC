import matplotlib.pyplot as plt

def visualize_point_clouds(original_coords, stego_coords, original_img):
    """Visualize original and stego point clouds"""
    plt.figure(figsize=(12, 6))
    
    # Original
    plt.subplot(1, 2, 1)
    plt.scatter(original_coords[:, 0].cpu(), 
                original_coords[:, 1].cpu(), 
                c=original_img.reshape(-1, 3), 
                s=10)
    plt.title("Original Point Cloud")
    plt.gca().invert_yaxis()
    
    # Stego
    plt.subplot(1, 2, 2)
    plt.scatter(stego_coords[:, 0].cpu(), 
                stego_coords[:, 1].cpu(), 
                c=original_img.reshape(-1, 3), 
                s=10)
    plt.title("Stego Point Cloud")
    plt.gca().invert_yaxis()
    
    plt.tight_layout()
    plt.show()