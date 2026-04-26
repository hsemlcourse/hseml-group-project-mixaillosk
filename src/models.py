import torch
import torch.nn as nn
import torch.functional as F


class SimpleMobileNetSeg(nn.Module):
    def __init__(self, num_classes=19, is_frozen_backbone=True):
        super().__init__()
        
        self.backbone = torch.hub.load('pytorch/vision:v0.10.0', 
                                       'mobilenet_v2', 
                                       pretrained=True)
        
        if is_frozen_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        self.seg_head = nn.Sequential(
            nn.Conv2d(1280, 256, 3, padding=1, dilation=2, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Декодер
            nn.Conv2d(256, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=32, mode='bilinear', align_corners=False),
            
            # Классификатор
            nn.Conv2d(128, num_classes, kernel_size=1)
        )
    
    def forward(self, x):
        original_size = x.shape[2:]
        
        with torch.no_grad() if self.is_frozen_backbone else torch.enable_grad():
            features = self.backbone.features(x)
            out = features[-1]
        
        out = self.seg_head(out)
        
        out = F.interpolate(out, size=original_size, mode='bilinear', align_corners=False)
        
        return out
