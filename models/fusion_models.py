# ================================
# 📄 models/fusion_models.py
# ================================
import torch
import torch.nn as nn
import torch.nn.functional as F


class EarlyFusionModel(nn.Module):
    def __init__(self, image_model, text_model):
        super(EarlyFusionModel, self).__init__()
        self.image_model = nn.Sequential(*list(image_model.resnet.children())[:-1])
        self.text_model = text_model.bert
        self.fc = nn.Linear(512 + 768, 2)

    def forward(self, image, input_ids, attention_mask):
        img_feat = self.image_model(image).view(image.size(0), -1)
        text_feat = self.text_model(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        combined = torch.cat((img_feat, text_feat), dim=1)
        return self.fc(combined)
