# ================================
# 📄 utils/dataset.py
# ================================
import torch
from torch.utils.data import Dataset
from PIL import Image
import json
import os
import numpy as np
import random
from nltk.corpus import wordnet

class HatefulMemesDataset(Dataset):
    def __init__(self, jsonl_file, img_dir, tokenizer, transform=None, text_augmentation=False, apply_augmentation=False):
        self.img_dir = img_dir
        self.transform = transform
        self.tokenizer = tokenizer
        self.text_augmentation = text_augmentation
        self.apply_augmentation = apply_augmentation
        
        self.data = []
        with open(jsonl_file, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))
        
        # Count class distribution
        self.class_counts = self._count_classes()
        
    def _count_classes(self):
        """Count the number of samples in each class"""
        class_counts = {0: 0, 1: 0}
        for sample in self.data:
            if 'label' in sample:
                class_counts[sample['label']] += 1
        return class_counts
    
    def __len__(self):
        return len(self.data)
    
    def _get_synonyms(self, word):
        """Get synonyms for a word using WordNet"""
        synonyms = []
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.append(lemma.name())
        return list(set(synonyms))
    
    def _augment_text(self, text):
        """Augment text by replacing some words with synonyms"""
        words = text.split()
        num_to_replace = max(1, int(len(words) * 0.2))  # Replace up to 20% of words
        
        for _ in range(num_to_replace):
            idx = random.randint(0, len(words) - 1)
            word = words[idx]
            synonyms = self._get_synonyms(word)
            if synonyms:
                words[idx] = random.choice(synonyms)
        
        return ' '.join(words)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        image_path = os.path.join(self.img_dir, sample['img'])
        image = Image.open(image_path).convert("RGB")
        text = sample['text']
        label = sample.get('label', -1)  # -1 for test set
        
        # Apply augmentation if needed (for minority class or if always enabled)
        if self.apply_augmentation:
            # Apply text augmentation if enabled
            if self.text_augmentation and random.random() < 0.5:
                text = self._augment_text(text)

        # Apply image transformations
        if self.transform:
            image = self.transform(image)
            
        encoding = self.tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=128)
        return {
            'image': image,
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long),
            'text': text,  # Include original text for visualization
            'image_path': sample['img']  # Include image path for visualization
        }
    
    def get_class_weights(self):
        """Calculate class weights for weighted loss function"""
        total = sum(self.class_counts.values())
        weights = {}
        for cls, count in self.class_counts.items():
            weights[cls] = total / (count * len(self.class_counts))
        return torch.tensor([weights[0], weights[1]], dtype=torch.float)