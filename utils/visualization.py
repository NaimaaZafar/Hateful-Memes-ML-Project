# ================================
# 📄 utils/visualization.py
# ================================
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import torch
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import torchvision.transforms as transforms

def display_sample_memes(dataset, num_samples=5, figsize=(15, 10)):
    """
    Display sample memes from the dataset with their labels and text.
    
    Args:
        dataset: HatefulMemesDataset instance
        num_samples: Number of samples to display
        figsize: Figure size for the plot
    """
    fig, axes = plt.subplots(num_samples, 1, figsize=figsize)
    
    # Get random indices
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    for i, idx in enumerate(indices):
        sample = dataset[idx]
        image = sample['image']
        text = sample['text']
        label = sample['label'].item()
        
        # Convert tensor to numpy for visualization
        if isinstance(image, torch.Tensor):
            img_np = image.permute(1, 2, 0).numpy()
            # Denormalize if needed
            if img_np.max() <= 1.0:
                img_np = (img_np * 255).astype(np.uint8)
        else:
            img_np = np.array(image)
        
        axes[i].imshow(img_np)
        axes[i].set_title(f"Label: {'Hateful' if label == 1 else 'Not Hateful'}")
        axes[i].text(10, image.shape[1] - 20, text, fontsize=12, 
                   color='white', backgroundcolor='black', alpha=0.7)
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig

def plot_class_distribution(dataset, figsize=(10, 6)):
    """
    Plot the class distribution in the dataset.
    
    Args:
        dataset: HatefulMemesDataset instance
        figsize: Figure size for the plot
    """
    class_counts = dataset.class_counts
    classes = ['Not Hateful', 'Hateful']
    counts = [class_counts[0], class_counts[1]]
    
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(classes, counts, color=['green', 'red'])
    
    # Add count labels on top of bars
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
               str(count), ha='center', va='bottom')
    
    ax.set_title('Class Distribution in Hateful Memes Dataset')
    ax.set_xlabel('Class')
    ax.set_ylabel('Number of Samples')
    
    # Add percentage labels
    total = sum(counts)
    for i, count in enumerate(counts):
        percentage = count / total * 100
        ax.text(i, count/2, f"{percentage:.1f}%", ha='center', va='center')
    
    return fig

def generate_word_cloud(dataset, figsize=(12, 8), max_words=100):
    """
    Generate a word cloud of the most frequent words in the dataset.
    
    Args:
        dataset: HatefulMemesDataset instance
        figsize: Figure size for the plot
        max_words: Maximum number of words to include in the word cloud
    """
    # Collect all text from the dataset
    all_text = []
    hateful_text = []
    non_hateful_text = []
    
    for i in range(len(dataset)):
        sample = dataset[i]
        text = sample['text']
        label = sample['label'].item()
        
        all_text.append(text)
        if label == 1:
            hateful_text.append(text)
        else:
            non_hateful_text.append(text)
    
    # Compute TF-IDF to identify important words
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(all_text)
    feature_names = vectorizer.get_feature_names_out()
    
    # Get average TF-IDF scores for each word
    tfidf_scores = tfidf_matrix.mean(axis=0).A1
    word_scores = {feature_names[i]: tfidf_scores[i] for i in range(len(feature_names))}
    
    # Create word clouds
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
    
    # All text word cloud
    wc_all = WordCloud(width=800, height=400, max_words=max_words, 
                       background_color='white').generate_from_frequencies(word_scores)
    ax1.imshow(wc_all, interpolation='bilinear')
    ax1.set_title('All Memes')
    ax1.axis('off')
    
    # Hateful text word cloud
    if hateful_text:
        hateful_text_combined = ' '.join(hateful_text)
        wc_hateful = WordCloud(width=800, height=400, max_words=max_words, 
                              background_color='white', colormap='Reds').generate(hateful_text_combined)
        ax2.imshow(wc_hateful, interpolation='bilinear')
        ax2.set_title('Hateful Memes')
        ax2.axis('off')
    
    # Non-hateful text word cloud
    if non_hateful_text:
        non_hateful_text_combined = ' '.join(non_hateful_text)
        wc_non_hateful = WordCloud(width=800, height=400, max_words=max_words, 
                                  background_color='white', colormap='Greens').generate(non_hateful_text_combined)
        ax3.imshow(wc_non_hateful, interpolation='bilinear')
        ax3.set_title('Non-Hateful Memes')
        ax3.axis('off')
    
    plt.tight_layout()
    return fig

def visualize_predictions(images, texts, true_labels, pred_probs, figsize=(15, 10), num_samples=5):
    """
    Visualize model predictions on sample memes.
    
    Args:
        images: List of images (as tensors or PIL Images)
        texts: List of text strings
        true_labels: List of true labels (0 or 1)
        pred_probs: List of prediction probabilities for class 1 (hateful)
        figsize: Figure size for the plot
        num_samples: Number of samples to display
    """
    if num_samples > len(images):
        num_samples = len(images)
    
    fig, axes = plt.subplots(num_samples, 1, figsize=figsize)
    if num_samples == 1:
        axes = [axes]
    
    # Get random indices
    indices = np.random.choice(len(images), num_samples, replace=False)
    
    for i, idx in enumerate(indices):
        image = images[idx]
        text = texts[idx]
        true_label = true_labels[idx]
        pred_prob = pred_probs[idx]
        
        # Convert tensor to numpy for visualization
        if isinstance(image, torch.Tensor):
            img_np = image.permute(1, 2, 0).cpu().numpy()
            # Denormalize if needed
            if img_np.max() <= 1.0:
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img_np = img_np * std + mean
                img_np = np.clip(img_np, 0, 1)
                img_np = (img_np * 255).astype(np.uint8)
        else:
            img_np = np.array(image)
        
        axes[i].imshow(img_np)
        
        # Format the title with prediction info
        pred_class = 'Hateful' if pred_prob > 0.5 else 'Not Hateful'
        true_class = 'Hateful' if true_label == 1 else 'Not Hateful'
        correct = pred_class == true_class
        color = 'green' if correct else 'red'
        
        axes[i].set_title(f"True: {true_class}, Pred: {pred_class} ({pred_prob:.2f})", 
                         color=color)
        axes[i].text(10, img_np.shape[0] - 20, text, fontsize=12, 
                   color='white', backgroundcolor='black', alpha=0.7)
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig
