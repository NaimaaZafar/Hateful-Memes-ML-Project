# ================================
# 📄 utils/preprocessing.py
# ================================
import torch
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# Download required NLTK resources
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')

def get_transform_train(include_augmentation=True):
    """
    Get transforms for training images with optional augmentation.
    
    Args:
        include_augmentation: Whether to include data augmentation
    """
    if include_augmentation:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def get_transform_eval():
    """Get transforms for evaluation/inference images."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def remove_stopwords(text):
    """
    Remove stopwords from text.
    
    Args:
        text: Input text string
    """
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text.lower())
    filtered_text = [word for word in word_tokens if word not in stop_words]
    return ' '.join(filtered_text)

def clean_text(text):
    """
    Basic text cleaning: remove special characters, extra spaces, and convert to lowercase.
    
    Args:
        text: Input text string
    """
    # Convert to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_text_for_lstm(text, remove_stops=True, clean=True):
    """
    Preprocess text for LSTM model.
    
    Args:
        text: Input text string
        remove_stops: Whether to remove stopwords
        clean: Whether to clean the text
    """
    if clean:
        text = clean_text(text)
    if remove_stops:
        text = remove_stopwords(text)
    return text

def prepare_lstm_batch(batch_texts, vocab, max_len=100):
    """
    Prepare a batch of texts for LSTM model.
    
    Args:
        batch_texts: List of text strings
        vocab: Vocabulary (word to index mapping)
        max_len: Maximum sequence length
    """
    tokenized = [word_tokenize(text.lower()) for text in batch_texts]
    # Convert words to indices
    indices = []
    for tokens in tokenized:
        seq = [vocab.get(token, vocab['<UNK>']) for token in tokens]
        # Pad or truncate to max_len
        if len(seq) < max_len:
            seq += [vocab['<PAD>']] * (max_len - len(seq))
        else:
            seq = seq[:max_len]
        indices.append(seq)
    return torch.tensor(indices, dtype=torch.long)

def build_vocab(texts, min_freq=2):
    """
    Build vocabulary from texts.
    
    Args:
        texts: List of text strings
        min_freq: Minimum frequency for a word to be included
    """
    word_counts = {}
    for text in texts:
        for word in word_tokenize(text.lower()):
            if word in word_counts:
                word_counts[word] += 1
            else:
                word_counts[word] = 1
    
    # Filter by minimum frequency
    vocab = {'<PAD>': 0, '<UNK>': 1}
    idx = 2
    for word, count in word_counts.items():
        if count >= min_freq:
            vocab[word] = idx
            idx += 1
    
    return vocab
