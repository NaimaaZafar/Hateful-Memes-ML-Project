# ================================
# 📄 models/image_models.py
# ================================
import torch.nn as nn
import torchvision.models as models

class ImageModel(nn.Module):
    def __init__(self):
        super(ImageModel, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 2)

    def forward(self, x):
        return self.resnet(x)