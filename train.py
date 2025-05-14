# ================================
# 📄 train.py
# ================================
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from transformers import BertTokenizer
from models.image_models import ImageModel
from models.text_models import TextModel
from models.fusion_models import EarlyFusionModel
from utils.dataset import HatefulMemesDataset
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Config
BATCH_SIZE = 16
EPOCHS = 5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load tokenizer, dataset, models
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
train_data = HatefulMemesDataset('data/train.jsonl', 'hateful_memes/data', tokenizer, transform)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

image_model = ImageModel()
text_model = TextModel()
model = EarlyFusionModel(image_model, text_model).to(DEVICE)

# Train
optimizer = optim.Adam(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()

model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    for batch in tqdm(train_loader):
        image = batch['image'].to(DEVICE)
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['label'].to(DEVICE)

        optimizer.zero_grad()
        outputs = model(image, input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

# Save model
torch.save(model.state_dict(), 'fusion_model.pth')