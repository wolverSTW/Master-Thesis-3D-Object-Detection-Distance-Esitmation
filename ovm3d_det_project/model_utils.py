import torch
import torch.nn as nn
import torchvision.models as models

class YOLO3DNetwork(nn.Module):
    def __init__(self):
        super(YOLO3DNetwork, self).__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        in_channels = 512
        
        self.head_2d = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(256, 5)
        )
        self.head_3d = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(256, 8)
        )

    def forward(self, x: torch.Tensor) -> dict:
        features = self.backbone(x)
        return {"pred_2d": self.head_2d(features), "pred_3d": self.head_3d(features)}