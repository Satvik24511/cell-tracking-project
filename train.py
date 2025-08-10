import torch
from utils import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
import torch.optim as optim
import config
import os
from tqdm import tqdm
from model import MaskRCNN
from dataset import CellDataset
from torchmetrics.detection import MeanAveragePrecision

def collate_fn(batch):
    return tuple(zip(*batch))

def train_fn(model, loader, opt, scaler):
    model.train()
    loop = tqdm(loader, leave=True, desc="Training")
    for images, targets in loop:
        images = list(image.to(config.DEVICE) for image in images)
        targets = [{k: v.to(config.DEVICE) for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast():
            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())

        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        
        loop.set_postfix(loss=loss.item())
        
def val_fn(model, loader, metric):
    model.eval()
    loop = tqdm(loader, leave=True, desc="Validation")
    
    with torch.no_grad():
        for images, targets in loop:
            images = list(image.to(config.DEVICE) for image in images)
            
            targets_ = [{k: v.to(config.DEVICE) for k, v in t.items()} for t in targets]
            
            detections = model(images)
            
            metric.update(detections, targets_)
            
    computed_metrics = metric.compute()
    return computed_metrics

def main():
    model = MaskRCNN(num_classes=2).to(config.DEVICE)
    opt = optim.Adam(list(model.parameters()), lr=config.LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()

    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT, model, opt)
    
    map_metric = MeanAveragePrecision(box_format="xyxy", iou_type="segm")

    train_dataset = CellDataset(
        root_img=os.path.join(config.TRAIN_DIR, "images"),
        root_mask=os.path.join(config.TRAIN_DIR, "masks"),
        transform=config.train_transforms,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    val_dataset = CellDataset(
        root_img=os.path.join(config.VAL_DIR, "images"),
        root_mask=os.path.join(config.VAL_DIR, "masks"),
        transform=config.val_transforms,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    
    best_val_map = 0.0

    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{config.NUM_EPOCHS}")
        train_fn(model, train_loader, opt, scaler)
        
        metrics = val_fn(model, val_loader, map_metric)
        
        val_map = metrics['map'].item()
        
        print(f"Epoch {epoch + 1} | Val mAP: {val_map:.4f}")

        if config.SAVE_MODEL and val_map > best_val_map:
            best_val_map = val_map
            save_checkpoint(model, opt, filename=config.CHECKPOINT)
            print(f"Model saved with mAP: {best_val_map:.4f}")
        
        map_metric.reset()

if __name__ == "__main__":
    main()