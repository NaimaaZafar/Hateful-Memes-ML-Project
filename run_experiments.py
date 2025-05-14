# ================================
# 📄 run_experiments.py
# ================================
import os
import argparse
import itertools
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

def run_experiment(args_dict, base_output_dir, experiment_id):
    """
    Run a single experiment with the given arguments
    
    Args:
        args_dict: Dictionary of arguments to pass to train.py
        base_output_dir: Base output directory
        experiment_id: Unique ID for this experiment
    
    Returns:
        The return code from the subprocess and experiment directory
    """
    # Create a unique directory for this experiment
    exp_dir = os.path.join(base_output_dir, f"experiment_{experiment_id}")
    os.makedirs(exp_dir, exist_ok=True)
    
    # Save the experiment configuration
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(args_dict, f, indent=4)
    
    # Build the command
    cmd = ["python", "train.py"]
    for key, value in args_dict.items():
        if value is True:
            cmd.append(f"--{key}")
        elif value is not False and value is not None:
            cmd.append(f"--{key}")
            cmd.append(str(value))
    
    # Add output directory
    cmd.append("--output_dir")
    cmd.append(exp_dir)
    
    # Log the command
    with open(os.path.join(exp_dir, "command.txt"), "w") as f:
        f.write(" ".join(cmd))
    
    # Run the command
    print(f"\n{'='*80}\nRunning experiment {experiment_id}: {' '.join(cmd)}\n{'='*80}")
    
    start_time = time.time()
    return_code = subprocess.call(cmd)
    end_time = time.time()
    
    # Log the execution time
    with open(os.path.join(exp_dir, "execution_time.txt"), "w") as f:
        f.write(f"Start time: {datetime.fromtimestamp(start_time)}\n")
        f.write(f"End time: {datetime.fromtimestamp(end_time)}\n")
        f.write(f"Duration: {end_time - start_time:.2f} seconds\n")
    
    # Evaluate the model using the best checkpoint
    if return_code == 0:
        # Find the best model checkpoint
        best_model_path = os.path.join(exp_dir, f"{args_dict['model_type']}_best.pth")
        if os.path.exists(best_model_path):
            eval_cmd = [
                "python", "evaluate.py",
                "--checkpoint", best_model_path,
                "--model_type", args_dict["model_type"],
                "--output_dir", os.path.join(exp_dir, "evaluation"),
                "--use_tensorboard"
            ]
            
            # Add model-specific arguments
            if "image_model" in args_dict:
                eval_cmd.extend(["--image_model", args_dict["image_model"]])
            if "text_model" in args_dict:
                eval_cmd.extend(["--text_model", args_dict["text_model"]])
            if "fusion_method" in args_dict and args_dict["model_type"] == "late_fusion":
                eval_cmd.extend(["--fusion_method", args_dict["fusion_method"]])
            if "resnet_variant" in args_dict and args_dict["image_model"] == "resnet":
                eval_cmd.extend(["--resnet_variant", args_dict["resnet_variant"]])
            if "bert_variant" in args_dict and args_dict["text_model"] == "bert":
                eval_cmd.extend(["--bert_variant", args_dict["bert_variant"]])
                
            # Add batch size for evaluation
            eval_cmd.extend(["--batch_size", str(args_dict["batch_size"])])
                
            print(f"\nEvaluating model: {' '.join(eval_cmd)}")
            subprocess.call(eval_cmd)
    
    return return_code, exp_dir

def run_all_experiments(base_args, experiments_dir, batch_sizes=[16, 32, 64]):
    """
    Run all combinations of experiments with different parameters
    
    Args:
        base_args: Base arguments to use for all experiments
        experiments_dir: Directory to save experiment results
        batch_sizes: List of batch sizes to test
    """
    # Create the experiments directory
    os.makedirs(experiments_dir, exist_ok=True)
    
    # Define parameter combinations to try
    all_configs = []
    
    # 1. Early fusion experiments
    early_fusion_configs = []
    
    # Early fusion with different image and text models
    for image_model in ["cnn", "resnet"]:
        for text_model in ["lstm", "bert"]:
            config = base_args.copy()
            config.update({
                "model_type": "early_fusion",
                "image_model": image_model,
                "text_model": text_model
            })
            
            # Add ResNet variants if using ResNet
            if image_model == "resnet":
                for resnet_variant in ["resnet18", "resnet50"]:
                    variant_config = config.copy()
                    variant_config["resnet_variant"] = resnet_variant
                    early_fusion_configs.append(variant_config)
            else:
                early_fusion_configs.append(config)
    
    # 2. Late fusion experiments
    late_fusion_configs = []
    
    # Late fusion with different image and text models and fusion methods
    for image_model in ["cnn", "resnet"]:
        for text_model in ["lstm", "bert"]:
            for fusion_method in ["weighted_sum", "concat", "mlp"]:
                config = base_args.copy()
                config.update({
                    "model_type": "late_fusion",
                    "image_model": image_model,
                    "text_model": text_model,
                    "fusion_method": fusion_method
                })
                
                # Add ResNet variants if using ResNet
                if image_model == "resnet":
                    for resnet_variant in ["resnet18", "resnet50"]:
                        variant_config = config.copy()
                        variant_config["resnet_variant"] = resnet_variant
                        late_fusion_configs.append(variant_config)
                else:
                    late_fusion_configs.append(config)
    
    # 3. Attention fusion experiments
    attention_fusion_configs = []
    
    # Attention fusion with different image and text models
    for image_model in ["cnn", "resnet"]:
        for text_model in ["lstm", "bert"]:
            config = base_args.copy()
            config.update({
                "model_type": "attention_fusion",
                "image_model": image_model,
                "text_model": text_model
            })
            
            # Add ResNet variants if using ResNet
            if image_model == "resnet":
                for resnet_variant in ["resnet18", "resnet50"]:
                    variant_config = config.copy()
                    variant_config["resnet_variant"] = resnet_variant
                    attention_fusion_configs.append(variant_config)
            else:
                attention_fusion_configs.append(config)
    
    # 4. Augmentation experiments (with best models only)
    augmentation_configs = []
    
    # Use the best model from early fusion for augmentation experiments
    best_model_config = {
        "model_type": "early_fusion",
        "image_model": "resnet",
        "text_model": "bert",
        "resnet_variant": "resnet50"
    }
    
    # Try different augmentation combinations
    for img_aug in [True, False]:
        for txt_aug in [True, False]:
            if not img_aug and not txt_aug:
                continue  # Skip the no augmentation case (covered in other experiments)
                
            config = base_args.copy()
            config.update(best_model_config)
            config["image_augmentation"] = img_aug
            config["text_augmentation"] = txt_aug
            augmentation_configs.append(config)
    
    # Combine all experiment configurations
    base_configs = early_fusion_configs + late_fusion_configs + attention_fusion_configs + augmentation_configs
    
    # Apply batch sizes to all configurations
    for batch_size in batch_sizes:
        for config in base_configs:
            new_config = config.copy()
            new_config["batch_size"] = batch_size
            all_configs.append(new_config)
    
    # Run each experiment
    results = []
    for i, config in enumerate(all_configs):
        experiment_id = f"{i+1:03d}_{config['model_type']}_{config['image_model']}_{config['text_model']}"
        if "fusion_method" in config:
            experiment_id += f"_{config['fusion_method']}"
        if "resnet_variant" in config:
            experiment_id += f"_{config['resnet_variant']}"
        if config.get("image_augmentation") or config.get("text_augmentation"):
            aug_str = ""
            if config.get("image_augmentation"):
                aug_str += "img"
            if config.get("text_augmentation"):
                aug_str += "txt"
            experiment_id += f"_aug_{aug_str}"
        experiment_id += f"_bs{config['batch_size']}"
        
        return_code, exp_dir = run_experiment(config, experiments_dir, experiment_id)
        
        results.append({
            "experiment_id": experiment_id,
            "config": config,
            "output_dir": exp_dir,
            "return_code": return_code,
            "success": return_code == 0
        })
    
    # Write summary of all experiments
    with open(os.path.join(experiments_dir, "experiments_summary.json"), "w") as f:
        json.dump(results, f, indent=4)
    
    # Generate a markdown report
    with open(os.path.join(experiments_dir, "experiments_report.md"), "w") as f:
        f.write("# Hateful Memes Experiments Report\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Experiment Results\n\n")
        f.write("| # | Model Type | Image Model | Text Model | Fusion Method | ResNet | Batch Size | Augmentation | Success |\n")
        f.write("|---|-----------|-------------|------------|---------------|--------|------------|--------------|--------|\n")
        
        for i, result in enumerate(results):
            config = result["config"]
            f.write(f"| {i+1} | {config.get('model_type', '-')} | {config.get('image_model', '-')} | {config.get('text_model', '-')} | {config.get('fusion_method', '-')} | {config.get('resnet_variant', '-')} | {config.get('batch_size', '-')} | ")
            
            # Augmentation info
            aug_info = []
            if config.get("image_augmentation"):
                aug_info.append("Image")
            if config.get("text_augmentation"):
                aug_info.append("Text")
            if aug_info:
                f.write(", ".join(aug_info))
            else:
                f.write("-")
            
            f.write(f" | {'✅' if result['success'] else '❌'} |\n")
    
    # Generate performance comparison table grouped by model type and batch size
    with open(os.path.join(experiments_dir, "performance_report.md"), "w") as f:
        f.write("# Hateful Memes Performance Report\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Group experiments by model type
        model_types = ["early_fusion", "late_fusion", "attention_fusion"]
        for model_type in model_types:
            f.write(f"## {model_type.replace('_', ' ').title()} Models\n\n")
            f.write("| Batch Size | Image Model | Text Model | Fusion Method | ResNet | AUROC | Accuracy | F1 Score |\n")
            f.write("|------------|-------------|------------|---------------|--------|-------|----------|----------|\n")
            
            # Filter experiments by model type
            type_results = [r for r in results if r["config"].get("model_type") == model_type and r["success"]]
            
            # Sort by batch size
            type_results.sort(key=lambda x: (x["config"].get("batch_size", 0), 
                                           x["config"].get("image_model", ""), 
                                           x["config"].get("text_model", "")))
            
            for result in type_results:
                config = result["config"]
                
                # Try to load metrics from the evaluation directory
                metrics_file = os.path.join(result["output_dir"], "evaluation", f"{model_type}_metrics.json")
                metrics = {}
                if os.path.exists(metrics_file):
                    try:
                        with open(metrics_file, "r") as mf:
                            metrics = json.load(mf)
                    except:
                        pass
                
                auroc = metrics.get("auroc", "-")
                accuracy = metrics.get("accuracy", "-")
                f1 = metrics.get("f1", "-")
                
                if isinstance(auroc, float):
                    auroc = f"{auroc:.4f}"
                if isinstance(accuracy, float):
                    accuracy = f"{accuracy:.4f}"
                if isinstance(f1, float):
                    f1 = f"{f1:.4f}"
                
                f.write(f"| {config.get('batch_size', '-')} | {config.get('image_model', '-')} | {config.get('text_model', '-')} | {config.get('fusion_method', '-')} | {config.get('resnet_variant', '-')} | {auroc} | {accuracy} | {f1} |\n")
            
            f.write("\n")
    
    print(f"\nAll experiments completed. See results in {experiments_dir}")
    # Count successful experiments
    successful = sum(1 for r in results if r["success"])
    print(f"Successful experiments: {successful}/{len(results)}")
    print(f"Total configurations tested: {len(all_configs)}")
    print(f"Model configurations: {len(base_configs)}")
    print(f"Batch sizes tested: {batch_sizes}")

def main():
    parser = argparse.ArgumentParser(description="Run a comprehensive set of experiments for hateful memes detection")
    
    # Base training parameters
    parser.add_argument("--experiments_dir", default="experiments", help="Directory to store experiment results")
    parser.add_argument("--train_jsonl", default="data/train.jsonl", help="Path to training data")
    parser.add_argument("--val_jsonl", default="data/dev.jsonl", help="Path to validation data")
    parser.add_argument("--img_dir", default="data/img", help="Path to image directory")
    parser.add_argument("--batch_sizes", nargs="+", type=int, default=[16, 32, 64], 
                      help="Batch sizes to test (default: 16, 32, 64)")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs to train")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", default="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu", 
                     help="Device to use (cuda or cpu)")
    parser.add_argument("--weighted_loss", action="store_true", help="Use weighted loss for class imbalance")
    
    args = parser.parse_args()
    
    # Create a timestamp for this experiment run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiments_dir = os.path.join(args.experiments_dir, timestamp)
    
    # Convert arguments to a dictionary for experiments
    base_args = {
        "train_jsonl": args.train_jsonl,
        "val_jsonl": args.val_jsonl,
        "img_dir": args.img_dir,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "device": args.device,
        "weighted_loss": args.weighted_loss
    }
    
    # Run all experiments with specified batch sizes
    run_all_experiments(base_args, experiments_dir, batch_sizes=args.batch_sizes)

if __name__ == "__main__":
    main() 