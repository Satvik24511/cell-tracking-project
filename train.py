import torch
from utils import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
import torch.optim as optim
import config
import os
from tqdm import tqdm
from model import MaskRCNN
from dataset import CellDataset
import numpy as np

def collate_fn(batch):
    return tuple(zip(*batch))

def train_fn(model, loader, opt, scaler):
    model.train()
    loop = tqdm(loader, leave=True, desc="Training")
    for images, targets in loop:
        images = list(image.to(config.DEVICE) for image in images)
        targets = [{k: v.to(config.DEVICE) for k, v in t.items()} for t in targets]

        with torch.amp.autocast(device_type=config.DEVICE):
            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())

        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        
        loop.set_postfix(loss=loss.item())

def custom_map_calc(predictions, targets, iou_threshold=0.5):
    all_scores = []
    all_is_tp = []

    for img_preds, img_targets in zip(predictions, targets):
        
        pred_masks = img_preds['masks'].squeeze(1).cpu().numpy() if 'masks' in img_preds and img_preds['masks'].numel() > 0 else np.array([])
        pred_scores = img_preds['scores'].cpu().numpy() if 'scores' in img_preds and img_preds['scores'].numel() > 0 else np.array([])
        gt_masks = img_targets['masks'].squeeze(1).cpu().numpy() if 'masks' in img_targets and img_targets['masks'].numel() > 0 else np.array([])
        
        if gt_masks.size == 0 and pred_masks.size == 0:
            continue
        
        if gt_masks.size == 0:
            is_tp = np.zeros_like(pred_scores, dtype=bool)
            all_scores.extend(pred_scores.tolist())
            all_is_tp.extend(is_tp.tolist())
            continue

        if pred_masks.size == 0:
            continue

        ious = np.zeros((len(gt_masks), len(pred_masks)))
        for i, gt_mask in enumerate(gt_masks):
            for j, pred_mask in enumerate(pred_masks):
                intersection = np.sum(np.logical_and(pred_mask, gt_mask))
                union = np.sum(np.logical_or(pred_mask, gt_mask))
                if union > 0:
                    ious[i, j] = intersection / union
        
        is_tp = np.zeros_like(pred_scores, dtype=bool)
        gt_matched = np.zeros(len(gt_masks), dtype=bool)

        sorted_indices = np.argsort(pred_scores)[::-1]
        for idx in sorted_indices:
            iou_vector = ious[:, idx]
            best_gt_match_idx = np.argmax(iou_vector)
            
            if iou_vector[best_gt_match_idx] >= iou_threshold and not gt_matched[best_gt_match_idx]:
                is_tp[idx] = True
                gt_matched[best_gt_match_idx] = True

        all_scores.extend(pred_scores.tolist())
        all_is_tp.extend(is_tp.tolist())

    if not all_scores:
        return 0.0


    sorted_scores_indices = np.argsort(all_scores)[::-1]
    sorted_is_tp = np.array(all_is_tp)[sorted_scores_indices]

    num_gt = sum(len(t['masks']) for t in targets if 'masks' in t)
    
    if num_gt == 0:
        return 0.0
    
    tp_count = np.cumsum(sorted_is_tp)
    fp_count = np.cumsum(~sorted_is_tp)
    
    precision = tp_count / (tp_count + fp_count)
    recall = tp_count / num_gt

    ap = np.trapz(precision, recall)
    return ap

def val_fn(model, loader, iou_threshold=0.5):
    model.eval()
    loop = tqdm(loader, leave=True, desc="Validation")
    
    all_detections = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in loop:
            images = list(image.to(config.DEVICE) for image in images)
            
            targets_on_device = [{k: v.to(config.DEVICE) for k, v in t.items()} for t in targets]
            
            detections = model(images)
            
            all_detections.extend(detections)
            all_targets.extend(targets_on_device)
            
    map_score = custom_map_calc(all_detections, all_targets, iou_threshold)
    return map_score

def main():
    if torch.cuda.is_available():
        device = 'cuda'
        scaler = torch.amp.GradScaler('cuda')
        print("CUDA is available. Training on GPU.")
    else:
        device = 'cpu'
        scaler = torch.amp.GradScaler()
        print("CUDA not available. Training on CPU.")

    config.DEVICE = device
    model = MaskRCNN(num_classes=2).to(config.DEVICE)

    opt = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT, model, opt)
    
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
        
        val_map = val_fn(model, val_loader, iou_threshold=0.5)
        
        print(f"Epoch {epoch + 1} | Val mAP: {val_map:.4f}")

        if config.SAVE_MODEL and val_map > best_val_map:
            best_val_map = val_map
            save_checkpoint(model, opt, filename=config.CHECKPOINT)
            print(f"Model saved with mAP: {best_val_map:.4f}")


if __name__ == "__main__":
    main()