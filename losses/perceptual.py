import torch
import torch.nn as nn
from torchvision import models


class PerceptualLoss(nn.Module):
    def __init__(self):
        # calls parent nn.Module init so parameters are registered
        super().__init__()
        # pretrained vgg16 w/ ImageNet weights, this is a fixed
        # feature extractor that knows natural image structure
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        # vgg.features is list module of all conv/ReLU/pool layers
        # take only first 9 layers, capturing low/mid-level textures and edges
        # nn.Sequential() wraps them into one callable module that can be run
        self.feature_extractor = nn.Sequential(*list(vgg.features)[:9])

        # fixes all VGG parameters so gradients only flow back into the UNet
        for p in self.feature_extractor.parameters():
            p.requires_grad = False

    # pred_f and target_f are feature maps, tensors representing edge/texture responses
    def forward(self, pred, target):
        pred_f = self.feature_extractor(pred)
        target_f = self.feature_extractor(target)
        return nn.L1Loss()(pred_f, target_f)
    
