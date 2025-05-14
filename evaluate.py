# ================================
# 📄 evaluate.py
# ================================
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from transformers import BertTokenizer
from models.image_models import CNNImageModel, ResNetImageModel
from models.text_models import LSTMTextModel, BERTTextModel
from models.fusion_models import EarlyFusionModel, LateFusionModel, AttentionFusionModel
from utils.dataset import HatefulMemesDataset
from utils.preprocessing import get_transform_eval
from utils.visualization import visualize_predictions
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import argparse
import os
from tqdm import tqdm
import json

# Set device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def evaluate_model(model, data_loader, criterion=None):
    """
    Evaluate model on the given data loader
    
    Args:
        model: PyTorch model
        data_loader: PyTorch DataLoader
        criterion: Loss function (optional)
    
    Returns:
        Dictionary of evaluation metrics and predictions
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    total_loss = 0.0
    all_images = []
    all_texts = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            images = batch['image'].to(DEVICE)
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['label'].to(DEVICE)
            
            outputs = model(images, input_ids, attention_mask)
            
            if criterion is not None:
                loss = criterion(outputs, labels)
                total_loss += loss.item() * images.size(0)
            
            # For binary classification
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = (probs > 0.5).long()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            # Store a subset of data for visualization
            if len(all_images) < 20:  # Store max 20 samples for visualization
                all_images.extend(images.cpu())
                all_texts.extend([t for t in batch['text']])
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )
    try:
        auroc = roc_auc_score(all_labels, all_probs)
    except:
        auroc = 0.0  # In case of single class
    
    avg_loss = total_loss / len(data_loader.dataset) if criterion is not None else None
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auroc': auroc,
        'loss': avg_loss,
        'predictions': all_preds,
        'labels': all_labels,
        'probs': all_probs,
        'images': all_images[:20],  # Limit number of stored images
        'texts': all_texts[:20]     # Limit number of stored texts
    }

def plot_confusion_matrix(y_true, y_pred, figsize=(10, 8)):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=figsize)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Not Hateful', 'Hateful'])
    disp.plot(ax=ax)
    plt.title('Confusion Matrix')
    return fig

def plot_roc_curve(y_true, y_scores, figsize=(10, 8)):
    """Plot ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc="lower right")
    return fig

def log_metrics_to_tensorboard(writer, metrics, model_name, split='val'):
    """Log metrics to TensorBoard"""
    writer.add_scalar(f'{split}/accuracy', metrics['accuracy'], 0)
    writer.add_scalar(f'{split}/precision', metrics['precision'], 0)
    writer.add_scalar(f'{split}/recall', metrics['recall'], 0)
    writer.add_scalar(f'{split}/f1', metrics['f1'], 0)
    writer.add_scalar(f'{split}/auroc', metrics['auroc'], 0)
    
    # Add confusion matrix
    cm_fig = plot_confusion_matrix(metrics['labels'], metrics['predictions'])
    writer.add_figure(f'{split}/{model_name}_confusion_matrix', cm_fig, 0)
    
    # Add ROC curve
    roc_fig = plot_roc_curve(metrics['labels'], metrics['probs'])
    writer.add_figure(f'{split}/{model_name}_roc_curve', roc_fig, 0)
    
    # Add prediction visualizations
    if 'images' in metrics and 'texts' in metrics:
        vis_fig = visualize_predictions(
            metrics['images'], 
            metrics['texts'], 
            metrics['labels'][:len(metrics['images'])],
            metrics['probs'][:len(metrics['images'])],
            num_samples=min(5, len(metrics['images']))
        )
        writer.add_figure(f'{split}/{model_name}_predictions', vis_fig, 0)

def main(args):
    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Set up transforms
    transform = get_transform_eval()
    
    # Load datasets
    val_data = HatefulMemesDataset(
        args.val_jsonl, 
        args.img_dir, 
        tokenizer, 
        transform
    )
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    
    # Create the corresponding model
    if args.model_type == 'early_fusion':
        # Create image model
        if args.image_model == 'cnn':
            image_model = CNNImageModel()
        else:  # Default: ResNet
            image_model = ResNetImageModel(model_name=args.resnet_variant)
            
        # Create text model
        if args.text_model == 'lstm':
            # For LSTM, pass the tokenizer to properly set vocabulary size
            text_model = LSTMTextModel(vocab_size=30522, tokenizer=tokenizer)  # Default BERT vocab size as fallback
        else:  # Default: BERT
            text_model = BERTTextModel(model_name=args.bert_variant)
            
        # Create fusion model
        model = EarlyFusionModel(image_model, text_model)
        
    elif args.model_type == 'attention_fusion':
        # Create image model
        if args.image_model == 'cnn':
            image_model = CNNImageModel()
        else:  # Default: ResNet
            image_model = ResNetImageModel(model_name=args.resnet_variant)
            
        # Create text model
        if args.text_model == 'lstm':
            # For LSTM, pass the tokenizer to properly set vocabulary size
            text_model = LSTMTextModel(vocab_size=30522, tokenizer=tokenizer)  # Default BERT vocab size as fallback
        else:  # Default: BERT
            text_model = BERTTextModel(model_name=args.bert_variant)
            
        # Create fusion model
        model = AttentionFusionModel(image_model, text_model)
        
    elif args.model_type == 'late_fusion':
        # Create image model
        if args.image_model == 'cnn':
            image_model = CNNImageModel()
        else:  # Default: ResNet
            image_model = ResNetImageModel(model_name=args.resnet_variant)
            
        # Create text model
        if args.text_model == 'lstm':
            # For LSTM, pass the tokenizer to properly set vocabulary size
            text_model = LSTMTextModel(vocab_size=30522, tokenizer=tokenizer)  # Default BERT vocab size as fallback
        else:  # Default: BERT
            text_model = BERTTextModel(model_name=args.bert_variant)
            
        # Create fusion model
        model = LateFusionModel(image_model, text_model, fusion_method=args.fusion_method)
    
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    # Load model weights
    model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE))
    model = model.to(DEVICE)
    
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    
    # Evaluate model
    metrics = evaluate_model(model, val_loader, criterion)
    
    # Print metrics
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"AUROC: {metrics['auroc']:.4f}")
    
    # Save metrics to file
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, f"{args.model_type}_metrics.json"), 'w') as f:
        json.dump({
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'auroc': metrics['auroc']
        }, f, indent=4)
    
    # Create visualizations
    cm_fig = plot_confusion_matrix(metrics['labels'], metrics['predictions'])
    cm_fig.savefig(os.path.join(args.output_dir, f"{args.model_type}_confusion_matrix.png"))
    
    roc_fig = plot_roc_curve(metrics['labels'], metrics['probs'])
    roc_fig.savefig(os.path.join(args.output_dir, f"{args.model_type}_roc_curve.png"))
    
    # Visualize predictions
    if len(metrics['images']) > 0:
        vis_fig = visualize_predictions(
            metrics['images'], 
            metrics['texts'], 
            metrics['labels'][:len(metrics['images'])],
            metrics['probs'][:len(metrics['images'])],
            num_samples=min(5, len(metrics['images']))
        )
        vis_fig.savefig(os.path.join(args.output_dir, f"{args.model_type}_predictions.png"))
    
    # Log to TensorBoard if requested
    if args.use_tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(os.path.join(args.output_dir, 'tensorboard'))
        log_metrics_to_tensorboard(writer, metrics, args.model_type)
        writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate hateful memes detection models")
    parser.add_argument("--val_jsonl", default="data/dev.jsonl", type=str, help="Path to validation JSON file")
    parser.add_argument("--img_dir", default="data/img", type=str, help="Path to image directory")
    parser.add_argument("--checkpoint", required=True, type=str, help="Path to model checkpoint")
    parser.add_argument("--output_dir", default="results", type=str, help="Directory to save results")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size")
    parser.add_argument("--model_type", default="early_fusion", choices=["early_fusion", "late_fusion", "attention_fusion"], help="Type of fusion model")
    parser.add_argument("--image_model", default="resnet", choices=["cnn", "resnet"], help="Type of image model")
    parser.add_argument("--text_model", default="bert", choices=["lstm", "bert"], help="Type of text model")
    parser.add_argument("--fusion_method", default="weighted_sum", choices=["weighted_sum", "concat", "mlp", "average"], help="Fusion method for late fusion")
    parser.add_argument("--resnet_variant", default="resnet50", help="ResNet variant to use")
    parser.add_argument("--bert_variant", default="bert-base-uncased", help="BERT variant to use")
    parser.add_argument("--use_tensorboard", action="store_true", help="Log metrics to TensorBoard")
    
    args = parser.parse_args()
    main(args)
