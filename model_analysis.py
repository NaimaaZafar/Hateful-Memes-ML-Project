#!/usr/bin/env python
# ================================
# 📄 model_analysis.py
# ================================
import os
import json
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from torchviz import make_dot
from torchinfo import summary
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from torch.utils.data import DataLoader
from transformers import BertTokenizer

from models.image_models import CNNImageModel, ResNetImageModel
from models.text_models import LSTMTextModel, BERTTextModel
from models.fusion_models import EarlyFusionModel, LateFusionModel, AttentionFusionModel
from utils.dataset import HatefulMemesDataset
from utils.preprocessing import get_transform_eval

# Set device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

def load_model_from_checkpoint(checkpoint_path, config):
    """
    Load a model from a checkpoint
    
    Args:
        checkpoint_path: Path to the checkpoint file
        config: Configuration dictionary with model parameters
        
    Returns:
        Loaded PyTorch model
    """
    # Create tokenizer (needed for model initialization)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Create the corresponding model
    if config['model_type'] == 'early_fusion':
        # Create image model
        if config['image_model'] == 'cnn':
            image_model = CNNImageModel()
        else:  # Default: ResNet
            resnet_variant = config.get('resnet_variant', 'resnet50')
            image_model = ResNetImageModel(model_name=resnet_variant)
            
        # Create text model
        if config['text_model'] == 'lstm':
            # For LSTM, pass the tokenizer to properly set vocabulary size
            text_model = LSTMTextModel(vocab_size=30522, tokenizer=tokenizer)
        else:  # Default: BERT
            bert_variant = config.get('bert_variant', 'bert-base-uncased')
            text_model = BERTTextModel(model_name=bert_variant)
            
        # Create fusion model
        model = EarlyFusionModel(image_model, text_model)
        
    elif config['model_type'] == 'attention_fusion':
        # Create image model
        if config['image_model'] == 'cnn':
            image_model = CNNImageModel()
        else:  # Default: ResNet
            resnet_variant = config.get('resnet_variant', 'resnet50')
            image_model = ResNetImageModel(model_name=resnet_variant)
            
        # Create text model
        if config['text_model'] == 'lstm':
            # For LSTM, pass the tokenizer to properly set vocabulary size
            text_model = LSTMTextModel(vocab_size=30522, tokenizer=tokenizer)
        else:  # Default: BERT
            bert_variant = config.get('bert_variant', 'bert-base-uncased')
            text_model = BERTTextModel(model_name=bert_variant)
            
        # Create fusion model
        model = AttentionFusionModel(image_model, text_model)
        
    elif config['model_type'] == 'late_fusion':
        # Create image model
        if config['image_model'] == 'cnn':
            image_model = CNNImageModel()
        else:  # Default: ResNet
            resnet_variant = config.get('resnet_variant', 'resnet50')
            image_model = ResNetImageModel(model_name=resnet_variant)
            
        # Create text model
        if config['text_model'] == 'lstm':
            # For LSTM, pass the tokenizer to properly set vocabulary size
            text_model = LSTMTextModel(vocab_size=30522, tokenizer=tokenizer)
        else:  # Default: BERT
            bert_variant = config.get('bert_variant', 'bert-base-uncased')
            text_model = BERTTextModel(model_name=bert_variant)
            
        # Create fusion model
        fusion_method = config.get('fusion_method', 'weighted_sum')
        model = LateFusionModel(image_model, text_model, fusion_method=fusion_method)
    
    else:
        raise ValueError(f"Unknown model type: {config['model_type']}")
    
    # Check if checkpoint exists before loading
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        # Load model weights
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    else:
        print(f"WARNING: Checkpoint {checkpoint_path} not found. Using randomly initialized weights.")
    
    return model.to(DEVICE)

def evaluate_model(model, data_loader):
    """
    Evaluate model on the given data loader to get predictions for confusion matrix
    
    Args:
        model: PyTorch model
        data_loader: PyTorch DataLoader
    
    Returns:
        Dictionary with predictions and true labels
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in data_loader:
            images = batch['image'].to(DEVICE)
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['label'].to(DEVICE)
            
            outputs = model(images, input_ids, attention_mask)
            
            # For binary classification
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = (probs > 0.5).long()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return {
        'predictions': all_preds,
        'labels': all_labels,
        'probs': all_probs
    }

def plot_confusion_matrix(y_true, y_pred, title, figsize=(10, 8)):
    """
    Plot confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        title: Title for the plot
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    plt.figure(figsize=figsize)
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create heatmap with seaborn for better visualization
    ax = sns.heatmap(cm_norm, annot=cm, fmt='d', cmap='Blues',
                    xticklabels=['Not Hateful', 'Hateful'],
                    yticklabels=['Not Hateful', 'Hateful'])
    
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    ax.set_title(title)
    
    plt.tight_layout()
    return plt.gcf()

def visualize_model_architecture(model, input_shape, title, output_file):
    """
    Visualize model architecture using torchinfo
    
    Args:
        model: PyTorch model
        input_shape: Shape of the input tensor
        title: Title for the visualization
        output_file: Path to save the output
    """
    # Create model summary using torchinfo
    with open(output_file, 'w') as f:
        f.write(f"Model Architecture: {title}\n")
        f.write("=" * 80 + "\n")
        # Get model summary as string and write to file
        model_summary = summary(model, input_shape, verbose=0, depth=6)
        f.write(str(model_summary))

def main():
    # Set paths and parameters
    output_dir = "model_analysis_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Path to experiment directories
    experiment_dirs = [
        "experiments/20250514_173853",
        "experiments/20250514_173637",
        "experiments/debug_run/20250514_224704"
    ]
    
    # Data parameters
    val_jsonl = "data/dev.jsonl"
    img_dir = "data/img"  # Using the proper path for images
    batch_size = 16
    
    # Load tokenizer and transforms
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    transform = get_transform_eval()
    
    # Load validation dataset
    try:
        val_data = HatefulMemesDataset(val_jsonl, img_dir, tokenizer, transform)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        print(f"Loaded validation dataset with {len(val_data)} samples")
    except Exception as e:
        print(f"Error loading validation dataset: {e}")
        print("Using fallback img_dir...")
        # Fallback to just 'data' directory if 'data/img' doesn't exist
        img_dir = "data"
        val_data = HatefulMemesDataset(val_jsonl, img_dir, tokenizer, transform)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        print(f"Loaded validation dataset with {len(val_data)} samples")
    
    # Figure out what experiments we have
    all_experiments = []
    
    for exp_dir in experiment_dirs:
        # Check if directory exists
        if not os.path.exists(exp_dir):
            print(f"Experiment directory {exp_dir} not found. Skipping.")
            continue
        
        # Check for summary file
        summary_file = os.path.join(exp_dir, "experiments_summary.json")
        summary_partial = os.path.join(exp_dir, "experiments_summary_partial.json")
        
        if os.path.exists(summary_file):
            with open(summary_file, 'r') as f:
                experiments = json.load(f)
                all_experiments.extend(experiments)
        elif os.path.exists(summary_partial):
            with open(summary_partial, 'r') as f:
                experiments = json.load(f)
                all_experiments.extend(experiments)
        else:
            print(f"No experiment summary found in {exp_dir}. Skipping.")
    
    # Process each experiment
    for experiment in all_experiments:
        exp_id = experiment['experiment_id']
        exp_output_dir = experiment['output_dir']
        config = experiment['config']
        success = experiment.get('success', False)
        
        print(f"\nProcessing experiment: {exp_id}")
        print(f"Success: {success}")
        
        # Only process if experiment was successful or if we want to process all
        # For this example, we'll try to process all experiments
        
        # Checkpoint path
        checkpoint_path = os.path.join(exp_output_dir, f"{config['model_type']}_best.pth")
        
        # Try to load the model
        try:
            model = load_model_from_checkpoint(checkpoint_path, config)
            print(f"Successfully loaded model for {exp_id}")
            
            # Create a directory for this experiment's results
            exp_results_dir = os.path.join(output_dir, exp_id)
            os.makedirs(exp_results_dir, exist_ok=True)
            
            # Generate confusion matrix
            try:
                results = evaluate_model(model, val_loader)
                
                # Plot and save confusion matrix
                matrix_title = f"Confusion Matrix - {exp_id}"
                cm_fig = plot_confusion_matrix(results['labels'], results['predictions'], matrix_title)
                cm_path = os.path.join(exp_results_dir, "confusion_matrix.png")
                cm_fig.savefig(cm_path)
                plt.close(cm_fig)
                print(f"Saved confusion matrix to {cm_path}")
                
                # Visualize model architecture
                # For image models
                if config['model_type'] == "early_fusion":
                    # For early fusion, we need a batch of images and text inputs
                    model_arch_path = os.path.join(exp_results_dir, "model_architecture.txt")
                    # We use [1, 3, 224, 224] for image and [1, 128] for text inputs
                    # Note: This is just for visualization, real inference uses input_ids and attention_mask
                    visualize_model_architecture(
                        model, 
                        [(1, 3, 224, 224), (1, 128), (1, 128)],
                        f"Model Architecture - {exp_id}",
                        model_arch_path
                    )
                    print(f"Saved model architecture to {model_arch_path}")
                else:
                    # For other fusion types
                    model_arch_path = os.path.join(exp_results_dir, "model_architecture.txt")
                    visualize_model_architecture(
                        model, 
                        [(1, 3, 224, 224), (1, 128), (1, 128)],
                        f"Model Architecture - {exp_id}",
                        model_arch_path
                    )
                    print(f"Saved model architecture to {model_arch_path}")
                
            except Exception as e:
                print(f"Error evaluating model: {e}")
                
        except Exception as e:
            print(f"Error loading model for {exp_id}: {e}")

if __name__ == "__main__":
    main() 