import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
import pandas as pd
from PIL import Image
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torchmetrics import Accuracy, Precision
import numpy as np
from tqdm import tqdm

# Configuration - UPDATED PATHS
DATA_DIR = os.path.expanduser('~/Documents/EvasionProject/Dataset-GTSRB')
TRAIN_CSV = os.path.join(DATA_DIR, 'Train.csv')
TEST_CSV = os.path.join(DATA_DIR, 'Test.csv')
TRAIN_IMG_DIR = os.path.join(DATA_DIR, 'Train')  # Points to the Train folder
TEST_IMG_DIR = os.path.join(DATA_DIR, 'Test')    # Points to the Test folder
BATCH_SIZE = 32
EPOCHS = 4
LEARNING_RATE = 0.001
NUM_CLASSES = 43
IMG_SIZE = (224, 224)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Update the GTSRBDataset class with this corrected version
class GTSRBDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        
        # Preprocess paths to handle different formats
        self.image_paths = []
        for _, row in self.data.iterrows():
            path = row['Path']
            # Handle paths like "Train/11/00011_00025_00005.png" or "11/00011_00025_00005.png"
            if path.startswith('Train/'):
                # Remove duplicate 'Train' if present
                path = path.replace('Train/', '')
            self.image_paths.append(path)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        rel_path = self.image_paths[idx]
        class_id = str(self.data.iloc[idx]['ClassId'])
        
        # Build correct path - images are in structure: Dataset-GTSRB/Train/11/00011_...png
        img_path = os.path.join(self.img_dir, class_id, rel_path.split('/')[-1])
        
        if not os.path.exists(img_path):
            # Try alternative path structure if first attempt fails
            img_path = os.path.join(self.img_dir, rel_path)
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image not found at either: {os.path.join(self.img_dir, class_id, rel_path.split('/')[-1])} or {os.path.join(self.img_dir, rel_path)}")
        
        image = Image.open(img_path).convert('RGB')
        label = self.data.iloc[idx]['ClassId']
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# Data transformations
train_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets
print("Loading datasets...")
train_dataset = GTSRBDataset(TRAIN_CSV, TRAIN_IMG_DIR, transform=train_transform)
test_dataset = GTSRBDataset(TEST_CSV, TEST_IMG_DIR, transform=test_transform)

# Split train into train and validation
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# Model definition
print("Initializing EfficientNet...")
model = EfficientNet.from_pretrained('efficientnet-b0')
model._fc = nn.Linear(model._fc.in_features, NUM_CLASSES)
model = model.to(device)
print("Model loaded successfully!")

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)

# Metrics
accuracy = Accuracy(task="multiclass", num_classes=NUM_CLASSES).to(device)
precision = Precision(task="multiclass", num_classes=NUM_CLASSES, average='macro').to(device)

# Training and evaluation functions (same as before)

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for inputs, labels in tqdm(loader, desc="Training"):
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        _, preds = torch.max(outputs, 1)
        all_preds.append(preds)
        all_labels.append(labels)
        running_loss += loss.item() * inputs.size(0)
    
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy(torch.cat(all_preds), torch.cat(all_labels))
    return epoch_loss, epoch_acc

def validate_epoch(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Validating"):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            _, preds = torch.max(outputs, 1)
            all_preds.append(preds)
            all_labels.append(labels)
            running_loss += loss.item() * inputs.size(0)
    
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy(torch.cat(all_preds), torch.cat(all_labels))
    epoch_precision = precision(torch.cat(all_preds), torch.cat(all_labels))
    return epoch_loss, epoch_acc, epoch_precision

def train_model():
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'val_precision': []
    }
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        
        # Train phase
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc.cpu())
        
        # Validation phase
        val_loss, val_acc, val_precision = validate_epoch(model, val_loader, criterion)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc.cpu())
        history['val_precision'].append(val_precision.cpu())
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | Precision: {val_precision:.4f}")
    
    return history

# Train the model
print("Starting training...")
history = train_model()

# Plot training history
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history['train_acc'], label='Train Accuracy')
plt.plot(history['val_acc'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.savefig('training_history.png')
plt.show()

# Evaluate on test set
def evaluate_model():
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    
    test_loss, test_acc, test_precision = validate_epoch(model, test_loader, criterion)
    
    print("\nFinal Test Results:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Precision: {test_precision:.4f}")

evaluate_model()