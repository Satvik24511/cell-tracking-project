import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

class MaskRCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
        
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, in_features_mask // 8, num_classes)

    def forward(self, images, targets=None):
        if not isinstance(images, list):
            images = [image for image in images]
            
        return self.model(images, targets)