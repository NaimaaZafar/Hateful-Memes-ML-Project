
# ================================
# 📄 models/text_models.py
# ================================
from transformers import BertModel
import torch.nn as nn

class TextModel(nn.Module):
    def __init__(self):
        super(TextModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.classifier = nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        return self.classifier(pooled_output)
