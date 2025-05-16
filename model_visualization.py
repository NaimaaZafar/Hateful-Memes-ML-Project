#!/usr/bin/env python
# ================================
# 📄 model_visualization.py
# ================================
import os
import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageFont

# PyTorch imports
from torch.utils.data import DataLoader
from transformers import BertTokenizer

# Project imports
from models.image_models import CNNImageModel, ResNetImageModel
from models.text_models import LSTMTextModel, BERTTextModel
from models.fusion_models import EarlyFusionModel, LateFusionModel, AttentionFusionModel
from utils.dataset import HatefulMemesDataset
from utils.preprocessing import get_transform_eval

# Set device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

# Try to import visualkeras
try:
    import visualkeras
    import tensorflow as tf
    from tensorflow import keras
    VISUALKERAS_AVAILABLE = True
    print("visualkeras is available for model visualization")
except ImportError:
    VISUALKERAS_AVAILABLE = False
    print("visualkeras is not available. Please install with: pip install visualkeras")
    print("Falling back to text-based visualization")

def find_experiment_directories():
    """
    Find all experiment directories in the project
    
    Returns:
        List of experiment directory paths
    """
    experiment_dirs = []
    base_dir = "experiments"
    
    # Check if experiments directory exists
    if not os.path.exists(base_dir):
        print(f"Experiments directory '{base_dir}' not found.")
        return experiment_dirs
    
    # Walk through the experiments directory
    for root, dirs, files in os.walk(base_dir):
        # Look for directories that might be experiment runs (have summary files or experiment_ subdirectories)
        if "experiments_summary.json" in files or "experiments_summary_partial.json" in files:
            experiment_dirs.append(root)
            continue
            
        # Check if this directory contains experiment_* subdirectories
        has_experiment_dirs = any(d.startswith("experiment_") for d in dirs)
        if has_experiment_dirs:
            experiment_dirs.append(root)
    
    print(f"Found {len(experiment_dirs)} experiment directories:")
    for exp_dir in experiment_dirs:
        print(f"  - {exp_dir}")
        
    return experiment_dirs

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

def convert_to_keras_model(pytorch_model, model_type, config):
    """
    Convert PyTorch model to Keras model for visualization
    This is a simplified representation for visualization only
    
    Args:
        pytorch_model: PyTorch model
        model_type: Type of model (early_fusion, late_fusion, attention_fusion)
        config: Model configuration
        
    Returns:
        Keras model representation
    """
    if not VISUALKERAS_AVAILABLE:
        return None
        
    # Create a Keras Sequential model based on the PyTorch model structure
    keras_model = keras.Sequential(name=f"{model_type}")
    
    # Add image model representation
    if config['image_model'] == 'cnn':
        keras_model.add(keras.layers.InputLayer(input_shape=(224, 224, 3), name="Image_Input"))
        keras_model.add(keras.layers.Conv2D(64, kernel_size=3, activation='relu', name="Conv1"))
        keras_model.add(keras.layers.MaxPooling2D(pool_size=2, name="Pool1"))
        keras_model.add(keras.layers.Conv2D(128, kernel_size=3, activation='relu', name="Conv2"))
        keras_model.add(keras.layers.MaxPooling2D(pool_size=2, name="Pool2"))
        keras_model.add(keras.layers.Conv2D(256, kernel_size=3, activation='relu', name="Conv3"))
        keras_model.add(keras.layers.MaxPooling2D(pool_size=2, name="Pool3"))
        keras_model.add(keras.layers.Flatten(name="Flatten_Image"))
        keras_model.add(keras.layers.Dense(512, activation='relu', name="FC_Image"))
    else:  # ResNet
        resnet_variant = config.get('resnet_variant', 'resnet50')
        keras_model.add(keras.layers.InputLayer(input_shape=(224, 224, 3), name="Image_Input"))
        keras_model.add(keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='same', name="Conv1"))
        keras_model.add(keras.layers.BatchNormalization(name="BN1"))
        keras_model.add(keras.layers.Activation('relu', name="Relu1"))
        keras_model.add(keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same', name="MaxPool"))
        
        # Simplified ResNet blocks
        num_blocks = 4 if resnet_variant == 'resnet18' else 16
        for i in range(num_blocks):
            keras_model.add(keras.layers.Conv2D(64 * (2 ** (i//4)), kernel_size=3, padding='same', name=f"ResBlock_{i}_Conv1"))
            keras_model.add(keras.layers.BatchNormalization(name=f"ResBlock_{i}_BN1"))
            keras_model.add(keras.layers.Activation('relu', name=f"ResBlock_{i}_Relu1"))
        
        keras_model.add(keras.layers.GlobalAveragePooling2D(name="GAP"))
        keras_model.add(keras.layers.Dense(512, activation='relu', name="FC_Image"))
    
    # Add text model representation
    if config['text_model'] == 'lstm':
        keras_model.add(keras.layers.InputLayer(input_shape=(128,), name="Text_Input"))
        keras_model.add(keras.layers.Embedding(30522, 300, name="Embedding"))
        keras_model.add(keras.layers.Bidirectional(keras.layers.LSTM(256, return_sequences=True), name="Bi-LSTM1"))
        keras_model.add(keras.layers.Bidirectional(keras.layers.LSTM(256), name="Bi-LSTM2"))
        keras_model.add(keras.layers.Dense(512, activation='relu', name="FC_Text"))
    else:  # BERT
        keras_model.add(keras.layers.InputLayer(input_shape=(128,), name="Text_Input"))
        
        # Simplified BERT representation
        for i in range(6):  # Simplified 6-layer transformer
            keras_model.add(keras.layers.Dense(768, name=f"BERT_Layer_{i}"))
        
        keras_model.add(keras.layers.Dense(768, name="BERT_Pooler"))
        keras_model.add(keras.layers.Dense(512, activation='relu', name="FC_Text"))
    
    # Add fusion representation
    if model_type == 'early_fusion':
        keras_model.add(keras.layers.Dense(512, activation='relu', name="Fusion_FC1"))
        keras_model.add(keras.layers.Dropout(0.5, name="Dropout"))
        keras_model.add(keras.layers.Dense(2, activation='softmax', name="Output"))
    elif model_type == 'attention_fusion':
        keras_model.add(keras.layers.Attention(name="Cross_Attention"))
        keras_model.add(keras.layers.Dense(512, activation='relu', name="Fusion_FC1"))
        keras_model.add(keras.layers.Dropout(0.5, name="Dropout"))
        keras_model.add(keras.layers.Dense(2, activation='softmax', name="Output"))
    elif model_type == 'late_fusion':
        fusion_method = config.get('fusion_method', 'weighted_sum')
        if fusion_method == 'weighted_sum':
            keras_model.add(keras.layers.Dense(2, activation='softmax', name="Weighted_Sum"))
        elif fusion_method == 'concat':
            keras_model.add(keras.layers.Dense(2, activation='softmax', name="Concat_Fusion"))
        elif fusion_method == 'mlp':
            keras_model.add(keras.layers.Dense(8, activation='relu', name="MLP_Hidden"))
            keras_model.add(keras.layers.Dropout(0.5, name="Dropout"))
            keras_model.add(keras.layers.Dense(2, activation='softmax', name="Output"))
        else:  # Default: average
            keras_model.add(keras.layers.Dense(2, activation='softmax', name="Average_Fusion"))
    
    return keras_model

def visualize_model_architecture(pytorch_model, model_type, config, output_file):
    """
    Visualize model architecture using visualkeras
    
    Args:
        pytorch_model: PyTorch model
        model_type: Type of model
        config: Model configuration
        output_file: Path to save the output image
    """
    # Check if visualkeras is available
    if not VISUALKERAS_AVAILABLE:
        print("visualkeras not available. Using text-based visualization.")
        # Use string representation as fallback
        with open(output_file, 'w') as f:
            f.write(f"Model Architecture: {model_type}\n")
            f.write("=" * 80 + "\n")
            f.write(str(pytorch_model))
        return
    
    # Convert PyTorch model to Keras for visualization
    keras_model = convert_to_keras_model(pytorch_model, model_type, config)
    
    if keras_model is None:
        print("Failed to convert model for visualization.")
        return
    
    # Set up colors for different layer types
    color_map = {
        # Image model layers
        keras.layers.Conv2D: '#2ecc71',
        keras.layers.MaxPooling2D: '#3498db',
        keras.layers.GlobalAveragePooling2D: '#9b59b6',
        
        # Text model layers
        keras.layers.Embedding: '#e74c3c',
        keras.layers.LSTM: '#e67e22',
        keras.layers.Bidirectional: '#f1c40f',
        
        # Common layers
        keras.layers.Dense: '#1abc9c',
        keras.layers.Dropout: '#95a5a6',
        keras.layers.Flatten: '#7f8c8d',
        keras.layers.BatchNormalization: '#34495e',
        keras.layers.Activation: '#2c3e50',
        keras.layers.InputLayer: '#ffffff',
        keras.layers.Attention: '#ff00ff'
    }
    
    # Default color for layers not in the map
    default_color = '#cccccc'
    
    # Try to get a font
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except IOError:
        # If Arial is not available, try a default font or use None
        try:
            font = ImageFont.load_default()
        except:
            font = None
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Generate visualization
    visualkeras.layered_view(keras_model, to_file=output_file, 
                           color_map=color_map, 
                           default_color=default_color,
                           legend=True, 
                           font=font,
                           spacing=50,
                           scale_xy=1.0,
                           max_xy=2000,
                           background_fill='white')
    
    print(f"Saved model visualization to {output_file}")

def visualize_model_text(pytorch_model, model_type, config, output_file):
    """Fallback text-based visualization for when visualkeras is not available"""
    with open(output_file, 'w') as f:
        # Write header with model type and configuration
        f.write(f"Model Architecture: {model_type}\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Configuration:\n")
        for key, value in config.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n" + "=" * 80 + "\n\n")
        
        # Write the string representation of the model
        f.write(str(pytorch_model))
        
        # Add information about model components
        if hasattr(pytorch_model, 'image_model'):
            f.write("\n\n" + "-" * 40 + "\n")
            f.write("Image Model:\n")
            f.write("-" * 40 + "\n")
            f.write(str(pytorch_model.image_model))
            
        if hasattr(pytorch_model, 'text_model'):
            f.write("\n\n" + "-" * 40 + "\n")
            f.write("Text Model:\n")
            f.write("-" * 40 + "\n")
            f.write(str(pytorch_model.text_model))
            
        if hasattr(pytorch_model, 'fusion'):
            f.write("\n\n" + "-" * 40 + "\n")
            f.write("Fusion Layer(s):\n")
            f.write("-" * 40 + "\n")
            f.write(str(pytorch_model.fusion))

def main():
    # Set up output directory
    output_dir = "model_visualization_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Find experiment directories
    experiment_dirs = find_experiment_directories()
    
    if not experiment_dirs:
        print("No experiment directories found. Exiting.")
        return
    
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
        
        print(f"\nProcessing experiment: {exp_id}")
        
        # Checkpoint path
        checkpoint_path = os.path.join(exp_output_dir, f"{config['model_type']}_best.pth")
        
        # Create a directory for this experiment's results
        exp_results_dir = os.path.join(output_dir, exp_id)
        os.makedirs(exp_results_dir, exist_ok=True)
        
        # Try to load the model
        try:
            model = load_model_from_checkpoint(checkpoint_path, config)
            print(f"Successfully loaded model for {exp_id}")
            
            # Visualize model architecture
            model_type = config['model_type']
            
            # Create file paths for visualizations
            visual_output_file = os.path.join(exp_results_dir, "model_architecture.png")
            text_output_file = os.path.join(exp_results_dir, "model_architecture.txt")
            
            # Generate visualizations
            try:
                visualize_model_architecture(model, model_type, config, visual_output_file)
            except Exception as e:
                print(f"Error in visual model architecture visualization: {e}")
                
            # Always create text-based visualization as backup
            try:
                visualize_model_text(model, model_type, config, text_output_file)
                print(f"Saved text-based model description to {text_output_file}")
            except Exception as e:
                print(f"Error in text-based model description: {e}")
                
        except Exception as e:
            print(f"Error loading model for {exp_id}: {e}")
    
if __name__ == "__main__":
    main() 