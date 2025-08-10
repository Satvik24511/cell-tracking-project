import os
import numpy as np
import cv2  
import torch
from torch.utils.data import Dataset

class CellDataset(Dataset):
    def __init__(self, root_img, root_mask, transform=None):
        self.root_img = root_img
        self.root_mask = root_mask
        
        self.images = sorted(os.listdir(root_img))
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        filename = self.images[index]
        
        img_path = os.path.join(self.root_img, filename)
        mask_path = os.path.join(self.root_mask, filename)

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        

        if self.transform:
            augmentation = self.transform(image=image, mask=mask)
            image = augmentation['image']
            mask = augmentation['mask']
            
            
        obj_ids = np.unique(mask.numpy())
        obj_ids = obj_ids[1:]

        
        boxes = []
        masks = []
        labels = []

        for obj_id in obj_ids:
            instance_mask = (mask == obj_id)
            
            pos = torch.where(instance_mask)
            xmin = torch.min(pos[1])
            xmax = torch.max(pos[1])
            ymin = torch.min(pos[0])
            ymax = torch.max(pos[0])
            
            box_coords = torch.stack([xmin, ymin, xmax, ymax])
            
            if xmax - xmin > 0 and ymax - ymin > 0:
                boxes.append(box_coords)
                masks.append(instance_mask)
                labels.append(1)
            
        if not boxes:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            masks = torch.zeros((0, mask.shape[-2], mask.shape[-1]), dtype=torch.uint8)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.stack(boxes)
            masks = torch.stack(masks)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['masks'] = masks.to(torch.uint8)
        
        
        return image, target