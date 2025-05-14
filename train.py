# ================================
# 📄 train.py
# ================================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from models.image_models import CNNImageModel, ResNetImageModel
from models.text_models import LSTMTextModel, BERTTextModel
from models.fusion_models import EarlyFusionModel, LateFusionModel, AttentionFusionModel
from utils.dataset import HatefulMemesDataset
from utils.preprocessing import get_transform_train, get_transform_eval
from utils.visualization import display_sample_memes, plot_class_distribution, generate_word_cloud
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import argparse
import os
from tqdm import tqdm
import numpy as np
import random

# Set seeds for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# Training function
def train(model, train_loader, val_loader, args):
    device = args.device
    model = model.to(device)
    
    # Define loss function
    if args.weighted_loss and hasattr(train_loader.dataset, 'get_class_weights'):
        class_weights = train_loader.dataset.get_class_weights().to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"Using weighted loss with weights: {class_weights}")
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Define optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps
    )
    
    # TensorBoard writer
    writer = SummaryWriter(os.path.join(args.output_dir, 'tensorboard'))
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Training loop
    best_val_auroc = 0.0
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        train_targets = []
        train_preds = []
        
        # Training
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in progress_bar:
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(images, input_ids, attention_mask)
            loss = criterion(outputs, labels)
            
            loss.backward()
            if args.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item() * labels.size(0)
            
            # Compute accuracy
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Store predictions and targets for AUROC
            train_targets.extend(labels.cpu().numpy())
            probs = torch.softmax(outputs, dim=1)[:, 1]
            train_preds.extend(probs.detach().cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss.item(),
                'acc': 100. * correct / total
            })
        
        # Calculate training metrics
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = 100. * correct / total
        train_auroc = roc_auc_score(train_targets, train_preds)
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        val_targets = []
        val_preds = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                images = batch['image'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(images, input_ids, attention_mask)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * labels.size(0)
                
                # Compute accuracy
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Store predictions and targets for AUROC
                val_targets.extend(labels.cpu().numpy())
                probs = torch.softmax(outputs, dim=1)[:, 1]
                val_preds.extend(probs.detach().cpu().numpy())
        
        # Calculate validation metrics
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = 100. * correct / total
        val_auroc = roc_auc_score(val_targets, val_preds)
        
        # Print metrics
        print(f"Epoch {epoch+1}/{args.epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Train AUROC: {train_auroc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val AUROC: {val_auroc:.4f}")
        
        # Log to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('AUROC/train', train_auroc, epoch)
        writer.add_scalar('AUROC/val', val_auroc, epoch)
        
        # Save best model
        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            torch.save(model.state_dict(), os.path.join(args.output_dir, f"{args.model_type}_best.pth"))
            print(f"  New best model saved with Val AUROC: {val_auroc:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_auroc': val_auroc,
            }, os.path.join(args.output_dir, f"{args.model_type}_checkpoint_epoch_{epoch+1}.pth"))
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(args.output_dir, f"{args.model_type}_final.pth"))
    
    # Close TensorBoard writer
    writer.close()
    
    return best_val_auroc

# Data visualization function
def visualize_data(train_dataset, output_dir):
    """Create and save visualizations of the dataset"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Display sample memes
    fig = display_sample_memes(train_dataset, num_samples=5)
    fig.savefig(os.path.join(output_dir, 'sample_memes.png'))
    plt.close(fig)
    
    # Plot class distribution
    fig = plot_class_distribution(train_dataset)
    fig.savefig(os.path.join(output_dir, 'class_distribution.png'))
    plt.close(fig)
    
    # Generate word cloud
    fig = generate_word_cloud(train_dataset)
    fig.savefig(os.path.join(output_dir, 'word_cloud.png'))
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Train hateful memes detection models")
    
    # Data paths
    parser.add_argument("--train_jsonl", default="data/train.jsonl", help="Path to training JSONL file")
    parser.add_argument("--val_jsonl", default="data/dev.jsonl", help="Path to validation JSONL file")
    parser.add_argument("--img_dir", default="data", help="Path to image directory")
    parser.add_argument("--output_dir", default="results", help="Output directory for models and results")
    
    # Model configuration
    parser.add_argument("--model_type", default="early_fusion", choices=["early_fusion", "late_fusion", "attention_fusion"],
                        help="Type of fusion model to use")
    parser.add_argument("--image_model", default="resnet", choices=["cnn", "resnet"],
                        help="Type of image model to use")
    parser.add_argument("--text_model", default="bert", choices=["lstm", "bert"],
                        help="Type of text model to use")
    parser.add_argument("--fusion_method", default="weighted_sum", choices=["weighted_sum", "concat", "mlp", "average"],
                        help="Fusion method for late fusion")
    parser.add_argument("--resnet_variant", default="resnet50", help="ResNet variant to use")
    parser.add_argument("--bert_variant", default="bert-base-uncased", help="BERT variant to use")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--clip_grad_norm", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--weighted_loss", action="store_true", help="Use weighted loss for class imbalance")
    parser.add_argument("--save_every", type=int, default=1, help="Save checkpoint every N epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    # Data augmentation settings
    parser.add_argument("--image_augmentation", action="store_true", help="Apply image augmentation")
    parser.add_argument("--text_augmentation", action="store_true", help="Apply text augmentation")
    
    # Device
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to use (cuda or cpu)")
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.bert_variant)
    
    # Set up transforms
    train_transform = get_transform_train(include_augmentation=args.image_augmentation)
    val_transform = get_transform_eval()
    
    # Load datasets
    train_data = HatefulMemesDataset(
        args.train_jsonl, 
        args.img_dir, 
        tokenizer, 
        train_transform, 
        text_augmentation=args.text_augmentation, 
        apply_augmentation=True
    )
    
    val_data = HatefulMemesDataset(
        args.val_jsonl, 
        args.img_dir, 
        tokenizer, 
        val_transform
    )
    
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    
    # Visualize data
    visualize_data(train_data, os.path.join(args.output_dir, 'visualizations'))
    
    # Create models
    print(f"Creating {args.model_type} model with {args.image_model} image model and {args.text_model} text model")
    
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
    if args.model_type == 'early_fusion':
        model = EarlyFusionModel(image_model, text_model)
    elif args.model_type == 'late_fusion':
        model = LateFusionModel(image_model, text_model, fusion_method=args.fusion_method)
    elif args.model_type == 'attention_fusion':
        model = AttentionFusionModel(image_model, text_model)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    # Train model
    best_val_auroc = train(model, train_loader, val_loader, args)
    print(f"Training completed. Best validation AUROC: {best_val_auroc:.4f}")

if __name__ == "__main__":
    main()