# ================================
# 📄 utils/dataset.py
# ================================
import torch
from torch.utils.data import Dataset
from PIL import Image
import json
import os

class HatefulMemesDataset(Dataset):
    def __init__(self, jsonl_file, img_dir, tokenizer, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.tokenizer = tokenizer
        self.data = []
        with open(jsonl_file, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        image_path = os.path.join(self.img_dir, sample['img'])
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        text = sample['text']
        label = sample.get('label', -1)  # -1 for test set
        encoding = self.tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=128)
        return {
            'image': image,
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }