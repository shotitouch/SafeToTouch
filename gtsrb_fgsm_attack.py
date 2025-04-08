import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
import pandas as pd
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Configuration
DATA_DIR = os.path.expanduser('~/Documents/EvasionProject/Dataset-GTSRB')
TEST_CSV = os.path.join(DATA_DIR, 'Test.csv')
TEST_IMG_DIR = os.path.join(DATA_DIR, 'test')  # Changed from 'Test' to 'test'
MODEL_PATH = 'best_model.pth'
OUTPUT_DIR = os.path.expanduser('~/Documents/EvasionProject/AttackedImages')
BATCH_SIZE = 32
NUM_CLASSES = 43
IMG_SIZE = (224, 224)
EPSILON = 0.05

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create output directories if they don't exist
os.makedirs(os.path.join(OUTPUT_DIR, 'test'), exist_ok=True)  # Changed to lowercase 'test'

# Dataset class with improved path handling
class GTSRBDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        
        # Preprocess paths to handle different formats
        self.image_paths = []
        for _, row in self.data.iterrows():
            path = row['Path']
            # Handle paths like "Test/00000.png" or "test/00000.png"
            path = path.replace('Test/', '').replace('test/', '')
            self.image_paths.append(path)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        rel_path = self.image_paths[idx]
        class_id = str(self.data.iloc[idx]['ClassId'])
        
        # Try multiple possible path combinations
        possible_paths = [
            os.path.join(self.img_dir, rel_path),  # Direct path
            os.path.join(self.img_dir, rel_path.split('/')[-1])  # Just filename
        ]
        
        img_path = None
        for path in possible_paths:
            if os.path.exists(path):
                img_path = path
                break
                
        if img_path is None:
            raise FileNotFoundError(f"Image not found at any of: {possible_paths}")
        
        image = Image.open(img_path).convert('RGB')
        label = self.data.iloc[idx]['ClassId']
        
        if self.transform:
            image = self.transform(image)
            
        return image, label, rel_path.split('/')[-1]  # Also return filename

# Data transformation
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load test dataset
print("Loading test dataset...")
test_dataset = GTSRBDataset(TEST_CSV, TEST_IMG_DIR, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# Load the trained model with weights_only=True for security
print("Loading model...")
model = EfficientNet.from_pretrained('efficientnet-b0')
model._fc = nn.Linear(model._fc.in_features, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))  # Added weights_only=True
model = model.to(device)
model.eval()
print("Model loaded successfully!")

# FGSM attack function (unchanged)
def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

# Function to save images (updated output path to lowercase 'test')
def save_image(tensor, filename, original_class, predicted_class, epsilon):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = tensor * std + mean
    tensor = tensor.clamp(0, 1)
    
    image = tensor.cpu().detach().numpy()
    image = np.transpose(image, (1, 2, 0)) * 255
    image = image.astype('uint8')
    
    output_path = os.path.join(OUTPUT_DIR, 'test', f"{epsilon}_{filename}")
    im = Image.fromarray(image)
    im.save(output_path)
    
    info_path = os.path.join(OUTPUT_DIR, 'test', 'attack_info.csv')
    if not os.path.exists(info_path):
        with open(info_path, 'w') as f:
            f.write("Filename,Epsilon,OriginalClass,PredictedClass\n")
    
    with open(info_path, 'a') as f:
        f.write(f"{filename},{epsilon},{original_class},{predicted_class}\n")

# Perform FGSM attack (unchanged)
def perform_attack(epsilon):
    correct = 0
    total = 0
    
    for images, labels, filenames in tqdm(test_loader, desc=f"Attacking (ε={epsilon})"):
        images = images.to(device)
        labels = labels.to(device)
        
        images.requires_grad = True
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        
        loss = nn.CrossEntropyLoss()(outputs, labels)
        model.zero_grad()
        loss.backward()
        data_grad = images.grad.data
        
        perturbed_images = fgsm_attack(images, epsilon, data_grad)
        outputs_perturbed = model(perturbed_images)
        _, predicted_perturbed = torch.max(outputs_perturbed.data, 1)
        
        for i in range(min(5, len(images))):
            save_image(perturbed_images[i], filenames[i], labels[i].item(), predicted_perturbed[i].item(), epsilon)
        
        total += labels.size(0)
        correct += (predicted_perturbed == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f"Epsilon: {epsilon}\tTest Accuracy = {correct} / {total} = {accuracy:.2f}%")
    return accuracy

# Run attack with different epsilon values
epsilons = [0, 0.01, 0.05, 0.1, 0.2, 0.3]
accuracies = []

for eps in epsilons:
    acc = perform_attack(eps)
    accuracies.append(acc)

# Plot results
plt.figure(figsize=(8, 6))
plt.plot(epsilons, accuracies, "*-")
plt.yticks(np.arange(0, 110, step=10))
plt.xticks(np.arange(0, 0.35, step=0.05))
plt.title("Accuracy vs Epsilon")
plt.xlabel("Epsilon")
plt.ylabel("Accuracy")
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, 'accuracy_vs_epsilon.png'))
plt.show()

print("FGSM attack completed. Adversarial examples saved to:", OUTPUT_DIR)