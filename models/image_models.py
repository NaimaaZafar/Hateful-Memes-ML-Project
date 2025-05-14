# ================================
# 📄 models/image_models.py
# ================================
import torch.nn as nn
import torchvision.models as models
import torch

class CNNImageModel(nn.Module):
    def __init__(self, num_classes=2, feature_extract=True):
        super(CNNImageModel, self).__init__()
        
        # Simple CNN architecture
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Fourth conv block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
        # Feature extraction size (for fusion models)
        self.feature_dim = 256
    
    def forward(self, x, return_features=False):
        features = self.features(x)
        
        if return_features:
            # Return flattened features for fusion
            return features.view(x.size(0), -1)
        else:
            # Return class predictions
            return self.classifier(features)


class ResNetImageModel(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, model_name='resnet50'):
        super(ResNetImageModel, self).__init__()
        
        # Choose ResNet variant
        if model_name == 'resnet18':
            base_model = models.resnet18(pretrained=pretrained)
            self.feature_dim = 512
        elif model_name == 'resnet34':
            base_model = models.resnet34(pretrained=pretrained)
            self.feature_dim = 512
        elif model_name == 'resnet50':
            base_model = models.resnet50(pretrained=pretrained)
            self.feature_dim = 2048
        elif model_name == 'resnet101':
            base_model = models.resnet101(pretrained=pretrained)
            self.feature_dim = 2048
        else:
            base_model = models.resnet50(pretrained=pretrained)
            self.feature_dim = 2048
        
        # Remove the final fully connected layer
        self.features = nn.Sequential(*list(base_model.children())[:-1])
        
        # Add custom classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x, return_features=False):
        features = self.features(x)
        
        if return_features:
            # Return flattened features for fusion
            return features.view(x.size(0), -1)
        else:
            # Return class predictions
            return self.classifier(features)


# Legacy model for compatibility
class ImageModel(ResNetImageModel):
    def __init__(self, num_classes=2):
        super(ImageModel, self).__init__(num_classes=num_classes, model_name='resnet18')