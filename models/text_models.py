# ================================
# 📄 models/text_models.py
# ================================
import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
from transformers import RobertaModel, RobertaConfig

class LSTMTextModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=300, hidden_dim=256, num_layers=2, 
                 bidirectional=True, dropout=0.5, num_classes=2, pretrained_embeddings=None):
        super(LSTMTextModel, self).__init__()
        
        # Embedding layer
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(
                pretrained_embeddings, freeze=False, padding_idx=0
            )
            embedding_dim = pretrained_embeddings.shape[1]
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output layer
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.feature_dim = lstm_output_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, lstm_output_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim // 2, num_classes)
        )

    def forward(self, input_ids, attention_mask=None, return_features=False):
        # Embed the input
        embedded = self.embedding(input_ids)
        
        # Pass through LSTM
        if attention_mask is not None:
            # Pack padded sequence if mask is provided
            lengths = attention_mask.sum(dim=1).cpu()
            packed_embedded = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths, batch_first=True, enforce_sorted=False
            )
            packed_output, (hidden, _) = self.lstm(packed_embedded)
            
            # Unpack sequence
            output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        else:
            output, (hidden, _) = self.lstm(embedded)
        
        # Get the final hidden state
        if self.lstm.bidirectional:
            # Concatenate the final forward and backward hidden states
            hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        else:
            hidden = hidden[-1, :, :]
        
        if return_features:
            return hidden
        else:
            return self.classifier(hidden)


class BERTTextModel(nn.Module):
    def __init__(self, num_classes=2, model_name='bert-base-uncased', freeze_bert=False):
        super(BERTTextModel, self).__init__()
        
        # Load pre-trained BERT model
        if 'roberta' in model_name:
            self.bert = RobertaModel.from_pretrained(model_name)
            config = RobertaConfig.from_pretrained(model_name)
        else:
            self.bert = BertModel.from_pretrained(model_name)
            config = BertConfig.from_pretrained(model_name)
            
        self.feature_dim = config.hidden_size
        
        # Freeze BERT parameters if specified
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )

    def forward(self, input_ids, attention_mask, return_features=False):
        # Get BERT outputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use the pooled output (CLS token representation)
        pooled_output = outputs.pooler_output
        
        if return_features:
            return pooled_output
        else:
            return self.classifier(pooled_output)


# Legacy model for compatibility
class TextModel(BERTTextModel):
    def __init__(self, num_classes=2):
        super(TextModel, self).__init__(num_classes=num_classes, model_name='bert-base-uncased')
