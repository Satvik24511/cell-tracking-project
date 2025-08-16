import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision.transforms import functional as F
from PIL import Image
import torch.optim as optim

from model import MaskRCNN
from utils import load_checkpoint
import config

def get_test_images(test_dir, num_images=5):
    test_images_path = os.path.join(test_dir, '01')
    all_images = sorted(os.listdir(test_images_path))
    
    if len(all_images) > num_images:
        test_images_names = random.sample(all_images, num_images)
    else:
        test_images_names = all_images
        
    test_images = []
    for img_name in test_images_names:
        img_path = os.path.join(test_images_path, img_name)
        image = cv2.imread(img_path)
        
        if image is None:
            print(f"Warning: Could not load image {img_path}. Skipping.")
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        test_images.append((image, img_name))
        
    return test_images

def visualize_detections(image, detections, score_threshold=0.7):
    fig, ax = plt.subplots(1, 2, figsize=(10, 10))
    
    ax[0].imshow(image)
    ax[0].set_title("Original Image")
    ax[0].axis('off')
    
    ax[1].imshow(image)
    ax[1].set_title("Predicted Cell Segmentation")
    ax[1].axis('off')

    num_detections = len(detections['boxes'])
    if num_detections == 0:
        plt.show()
        return

    colors = plt.get_cmap('hsv', num_detections)
    
    for i in range(num_detections):
        score = detections['scores'][i].detach().item()
        if score > score_threshold:
            mask = detections['masks'][i].squeeze().detach().cpu().numpy()
            bbox = detections['boxes'][i].detach().cpu().numpy().astype(int)
            
            color = colors(i)
            ax[1].imshow(np.ma.masked_where(mask == 0, mask), cmap='hsv', alpha=0.5, vmin=0, vmax=num_detections)

            ax[1].add_patch(plt.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
                                       fill=False, edgecolor=color, linewidth=2))

            ax[1].text(bbox[0], bbox[1], f'{score:.2f}', bbox={'facecolor': 'white', 'alpha': 0.8},
                    fontsize=8, color='black')

    plt.tight_layout()
    plt.show()

def main():
    num_classes = 2
    model = MaskRCNN(num_classes).to(config.DEVICE)
    
    checkpoint_path = os.path.join(os.getcwd(), config.CHECKPOINT)
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        return

    optimizer = optim.Adam(model.parameters()) 
    load_checkpoint(checkpoint_path, model, optimizer)
    
    print(f"Model loaded from {config.CHECKPOINT}")
    
    model.eval()
    
    print("Loading test images...")
    test_images_data = get_test_images(test_dir=config.TEST_DIR, num_images=5)
    
    if not test_images_data:
        print("No images found in the test directory. Please check the path and file naming.")
        return

    print("\nGenerating and visualizing predictions...")
    for original_image, filename in tqdm(test_images_data, desc="Processing images"):
        image_pil = Image.fromarray(original_image)
        preprocessed_image_tensor = F.to_tensor(image_pil).to(config.DEVICE)
        
        detections = model([preprocessed_image_tensor])
        

        visualize_detections(image_pil, detections[0])

if __name__ == "__main__":
    main()
