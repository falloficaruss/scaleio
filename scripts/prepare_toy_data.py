import os
import torch
import numpy as np
from PIL import Image
from pathlib import Path

def create_toy_dataset(target_dir='data/toy_dataset', num_images=5, size=(256, 256)):
    """Creates a small set of synthetic images for testing."""
    os.makedirs(target_dir, exist_ok=True)
    print(f"Creating toy dataset at {target_dir}...")
    
    for i in range(num_images):
        # Create a random RGB image
        img_np = np.random.randint(0, 256, (*size, 3), dtype=np.uint8)
        # Add some structure (e.g., a rectangle) to make it slightly less "noise-like"
        img_np[50:200, 50:200, :] = [i * 40 % 255, (i * 80) % 255, (i * 120) % 255]
        
        img = Image.fromarray(img_np)
        img.save(os.path.join(target_dir, f'toy_{i}.png'))
    
    print(f"Generated {num_images} images.")

if __name__ == "__main__":
    create_toy_dataset()
