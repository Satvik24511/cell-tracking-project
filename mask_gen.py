import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
from torchvision.transforms import functional as F
from PIL import Image

from model import MaskRCNN
from utils import load_checkpoint
import config

def generate_detections(model, image_dir, output_dir, transform):
    
    if not os.path.isdir(image_dir):
        print(f"Error: Image directory '{image_dir}' not found.")
        return

    os.makedirs(output_dir, exist_ok=True)

    image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith('.tif')])

    if not image_files:
        print(f"No images found in '{image_dir}'. Skipping inference.")
        return

    model.eval() 
    device = config.DEVICE

    print(f"\nRunning inference on {len(image_files)} frames from '{image_dir}'...")

    for i, filename in tqdm(enumerate(image_files), total=len(image_files), desc="Processing Frames"):
        img_path = os.path.join(image_dir, filename)
        
        original_image = cv2.imread(img_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        
        preprocessed_image = F.to_tensor(Image.fromarray(original_image)).to(device)

        with torch.no_grad():
            detections = model([preprocessed_image])

        output_filename = f"frame_{i:04d}_detections.pth"
        output_path = os.path.join(output_dir, output_filename)
        
        torch.save(detections[0], output_path)
        
    print("\nInference complete. Raw detections saved.")

def main():
    num_classes = 2
    model = MaskRCNN(num_classes).to(config.DEVICE)
    
    checkpoint_path = os.path.join(os.getcwd(), config.CHECKPOINT)
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        return

    optimizer = torch.optim.Adam(model.parameters()) 
    load_checkpoint(checkpoint_path, model, optimizer)
    
    print(f"Model loaded from {config.CHECKPOINT}")

    image_dir = r'dataset\test\01'
    output_dir = r'dataset\test\01_detections'

    generate_detections(model, image_dir, output_dir, config.val_transforms)

if __name__ == "__main__":
    main()