# Hateful Memes Detection Project

This repository contains the implementation of a multimodal machine learning system for detecting hateful content in memes using the Hateful Memes dataset from Meta AI.

## Project Overview

Memes often combine image and text in ways that require multimodal understanding. This project implements and compares different approaches for hateful meme classification:

- Text-only models (BERT, LSTM)
- Image-only models (CNN, ResNet)
- Multimodal fusion strategies (early fusion, late fusion, attention-based fusion)

## Dataset

The project uses the Hateful Memes dataset which consists of 10,000 multimodal examples. Each example is an image with text, with a label indicating whether it's hateful (1) or not (0).

The dataset is organized as follows:
- `data/img/` - Contains all meme images
- `data/train.jsonl` - Training set
- `data/dev.jsonl` - Development/validation set
- `data/test.jsonl` - Test set

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Hateful-Memes-ML-Project.git
cd Hateful-Memes-ML-Project
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
# For Windows
venv\Scripts\activate
# For Linux/Mac
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download NLTK resources:
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

## Project Structure

- `models/` - Implementation of image, text, and fusion models
- `utils/` - Utility functions for data loading, preprocessing, and visualization
- `train.py` - Script for training models
- `evaluate.py` - Script for evaluating trained models
- `run_experiments.py` - Script for running comprehensive model comparisons

## Usage

### Data Exploration and Visualization

To visualize the dataset, run the training script with visualization flags:

```bash
python train.py --output_dir results/exploration
```

This will generate visualizations in the `results/exploration/visualizations` directory, including:
- Sample memes from the dataset
- Class distribution
- Word clouds

### Training Models

The training script supports various configurations for different models and fusion strategies:

```bash
# Train early fusion model with ResNet50 and BERT
python train.py --model_type early_fusion --image_model resnet --text_model bert --resnet_variant resnet50 --output_dir results/early_fusion

# Train late fusion model with CNN and LSTM
python train.py --model_type late_fusion --image_model cnn --text_model lstm --fusion_method weighted_sum --output_dir results/late_fusion

# Train with image and text augmentation
python train.py --model_type early_fusion --image_augmentation --text_augmentation --output_dir results/augmented
```

### Evaluating Models

To evaluate a trained model on the validation set:

```bash
python evaluate.py --checkpoint results/early_fusion/early_fusion_best.pth --model_type early_fusion --output_dir results/evaluation
```

### Running Comprehensive Experiments

To automatically test all model combinations with different batch sizes (16, 32, 64):

```bash
python run_experiments.py --epochs 5
```

The script will test all possible combinations of:
- Model types: early fusion, late fusion, attention fusion
- Image models: CNN, ResNet (ResNet18, ResNet50)
- Text models: LSTM, BERT
- Fusion methods: weighted sum, concat, MLP (for late fusion)
- Batch sizes: 16, 32, 64

You can customize parameters:

```bash
# Test with different batch sizes
python run_experiments.py --batch_sizes 8 16 32 --epochs 3

# Test with specific data paths
python run_experiments.py --train_jsonl custom/train.jsonl --val_jsonl custom/dev.jsonl

# Test with weighted loss for class imbalance
python run_experiments.py --weighted_loss
```

Experiment results are saved in the `experiments/[timestamp]` directory, including:
- `experiments_summary.json` - Details of all experiments
- `experiments_report.md` - Overview of all experiments with success status
- `performance_report.md` - Detailed comparison of model performance
- Individual experiment directories with model checkpoints and evaluation metrics

## Model Architectures

### Image Models

1. **CNN**: A custom convolutional neural network with multiple conv-pool layers.
2. **ResNet**: Pre-trained ResNet model (ResNet18, ResNet50, etc.) with transfer learning.

### Text Models

1. **LSTM**: Bidirectional LSTM network with embeddings.
2. **BERT**: Pre-trained BERT model fine-tuned on the task.

### Fusion Strategies

1. **Early Fusion**: Features from both modalities are concatenated before classification.
2. **Late Fusion**: Predictions from separate models are combined using weighted sum or MLP.
3. **Attention Fusion**: Cross-modal attention mechanism to weigh features from different modalities.

## Results Visualization

Evaluation results are stored in the specified output directory and include:
- Confusion matrix
- ROC curve
- Sample predictions with visualization
- Performance metrics (AUROC, accuracy, precision, recall, F1-score)

TensorBoard logs are also available in the `tensorboard` subdirectory:

```bash
tensorboard --logdir results/early_fusion/tensorboard
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

This project uses the Hateful Memes dataset from Meta AI. Please cite:

```
@inproceedings{Kiela2020TheHM,
  title={The Hateful Memes Challenge: Detecting Hate Speech in Multimodal Memes},
  author={Douwe Kiela and Hamed Firooz and Aravind Mohan and Vedanuj Goswami and Amanpreet Singh and Pratik Ringshia and Davide Testuggine},
  year={2020}
}
```