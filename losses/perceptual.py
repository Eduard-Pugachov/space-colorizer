import torch
import torch.nn as nn
from torchvision import models


class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(vgg.features)[:9])

        for p in self.feature_extractor.parameters():
            p.requires_grad = False

    def forward(self, pred, target):
        pred_f = self.feature_extractor(pred)
        target_f = self.feature_extractor(target)
        return nn.L1Loss()(pred_f, target_f)
    
    