# ================================
# 📄 models/fusion_models.py
# ================================
import torch
import torch.nn as nn
import torch.nn.functional as F


class EarlyFusionModel(nn.Module):
    def __init__(self, image_model, text_model, fusion_output_size=512, dropout=0.5):
        super(EarlyFusionModel, self).__init__()
        self.image_model = image_model
        self.text_model = text_model
        
        # Get feature dimensions from each model
        self.image_feature_dim = getattr(image_model, 'feature_dim', 512)
        self.text_feature_dim = getattr(text_model, 'feature_dim', 768)
        self.combined_dim = self.image_feature_dim + self.text_feature_dim
        
        # Fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(self.combined_dim, fusion_output_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_output_size, 2)
        )

    def forward(self, image, input_ids, attention_mask):
        # Extract features from each modality
        img_features = self.image_model(image, return_features=True)
        text_features = self.text_model(input_ids, attention_mask, return_features=True)
        
        # Concatenate features
        combined = torch.cat((img_features, text_features), dim=1)
        
        # Pass through fusion layers
        return self.fusion(combined)


class AttentionFusionModel(nn.Module):
    def __init__(self, image_model, text_model, fusion_output_size=512, dropout=0.5):
        super(AttentionFusionModel, self).__init__()
        self.image_model = image_model
        self.text_model = text_model
        
        # Get feature dimensions from each model
        self.image_feature_dim = getattr(image_model, 'feature_dim', 512)
        self.text_feature_dim = getattr(text_model, 'feature_dim', 768)
        
        # Attention mechanism
        self.attention_image = nn.Linear(self.image_feature_dim, 1)
        self.attention_text = nn.Linear(self.text_feature_dim, 1)
        
        # Projection layers to common dimension
        common_dim = fusion_output_size // 2
        self.project_image = nn.Linear(self.image_feature_dim, common_dim)
        self.project_text = nn.Linear(self.text_feature_dim, common_dim)
        
        # Fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(common_dim * 2, fusion_output_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_output_size, 2)
        )

    def forward(self, image, input_ids, attention_mask):
        # Extract features from each modality
        img_features = self.image_model(image, return_features=True)
        text_features = self.text_model(input_ids, attention_mask, return_features=True)
        
        # Calculate attention weights
        img_attn = torch.sigmoid(self.attention_image(img_features))
        text_attn = torch.sigmoid(self.attention_text(text_features))
        
        # Apply attention and project to common dimension
        img_features = self.project_image(img_features * img_attn)
        text_features = self.project_text(text_features * text_attn)
        
        # Concatenate features
        combined = torch.cat((img_features, text_features), dim=1)
        
        # Pass through fusion layers
        return self.fusion(combined)


class LateFusionModel(nn.Module):
    def __init__(self, image_model, text_model, fusion_method='weighted_sum'):
        super(LateFusionModel, self).__init__()
        self.image_model = image_model
        self.text_model = text_model
        self.fusion_method = fusion_method
        
        if fusion_method == 'concat':
            self.fusion = nn.Linear(4, 2)  # 2 classes from each model
        elif fusion_method == 'weighted_sum':
            # Learnable weights for each modality
            self.alpha = nn.Parameter(torch.tensor(0.5))
        elif fusion_method == 'mlp':
            self.fusion = nn.Sequential(
                nn.Linear(4, 8),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(8, 2)
            )

    def forward(self, image, input_ids, attention_mask):
        # Get predictions from each model
        img_preds = self.image_model(image)
        text_preds = self.text_model(input_ids, attention_mask)
        
        # Apply fusion method
        if self.fusion_method == 'concat':
            combined = torch.cat((img_preds, text_preds), dim=1)
            return self.fusion(combined)
        elif self.fusion_method == 'weighted_sum':
            # Weighted sum of predictions
            alpha = torch.sigmoid(self.alpha)  # Constrain between 0 and 1
            return alpha * img_preds + (1 - alpha) * text_preds
        elif self.fusion_method == 'mlp':
            combined = torch.cat((img_preds, text_preds), dim=1)
            return self.fusion(combined)
        else:  # Default: average
            return (img_preds + text_preds) / 2
