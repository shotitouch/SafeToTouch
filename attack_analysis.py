import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
from sklearn.metrics import confusion_matrix
from PIL import Image

# Configuration
DATA_DIR = os.path.expanduser('~/Documents/EvasionProject/Dataset-GTSRB')
TEST_CSV = os.path.join(DATA_DIR, 'Test.csv')
TEST_IMG_DIR = os.path.join(DATA_DIR, 'test')
MODEL_PATH = 'best_model.pth'
OUTPUT_DIR = os.path.expanduser('~/Documents/EvasionProject/AttackAnalysis')
os.makedirs(OUTPUT_DIR, exist_ok=True)
NUM_CLASSES = 43  # GTSRB has 43 classes
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset Class
class GTSRBDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.image_paths = [row['Path'].replace('Test/', '').replace('test/', '') 
                          for _, row in self.data.iterrows()]
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        rel_path = self.image_paths[idx]
        img_path = os.path.join(self.img_dir, rel_path.split('/')[-1])
        
        if not os.path.exists(img_path):
            img_path = os.path.join(self.img_dir, rel_path)
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image not found at: {img_path}")
        
        image = Image.open(img_path).convert('RGB')
        label = self.data.iloc[idx]['ClassId']
        
        if self.transform:
            image = self.transform(image)
            
        return image, label, rel_path.split('/')[-1]

# Metrics Tracking Class
class AttackMetrics:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()
        
    def reset(self):
        self.original_confidences = []
        self.perturbed_confidences = []
        self.original_preds = []
        self.perturbed_preds = []
        self.labels = []
        self.perturbations = []
        
    def update(self, orig_output, perturb_output, labels, perturbation):
        self.original_confidences.extend(F.softmax(orig_output, dim=1).cpu().detach().numpy())
        self.perturbed_confidences.extend(F.softmax(perturb_output, dim=1).cpu().detach().numpy())
        self.original_preds.extend(torch.argmax(orig_output, dim=1).cpu().numpy())
        self.perturbed_preds.extend(torch.argmax(perturb_output, dim=1).cpu().numpy())
        self.labels.extend(labels.cpu().numpy())
        self.perturbations.extend(perturbation.cpu().numpy())
    
    def calculate_metrics(self):
        metrics = {}
        labels = np.array(self.labels)
        orig_preds = np.array(self.original_preds)
        perturb_preds = np.array(self.perturbed_preds)
        
        metrics['asr'] = np.mean(perturb_preds != orig_preds)
        metrics['fooling_rate'] = np.mean(perturb_preds != labels)
        metrics['accuracy'] = np.mean(perturb_preds == labels)
        
        orig_conf = np.array(self.original_confidences)
        perturb_conf = np.array(self.perturbed_confidences)
        true_class_conf = orig_conf[np.arange(len(orig_conf)), labels]
        metrics['confidence_drop'] = np.mean(true_class_conf - perturb_conf[np.arange(len(perturb_conf)), labels])
        
        metrics['avg_perturbation'] = np.mean([np.linalg.norm(p.reshape(-1)) for p in self.perturbations])
        metrics['confusion_matrix'] = confusion_matrix(labels, perturb_preds)
        
        return metrics

# Attack Functions
def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    return torch.clamp(perturbed_image, 0, 1)

def targeted_attack(model, images, labels, target_class, epsilon):
    images.requires_grad = True
    outputs = model(images)
    loss = F.cross_entropy(outputs, target_class)  # Use target_class tensor directly
    model.zero_grad()
    loss.backward()
    perturbation = epsilon * images.grad.sign()
    return torch.clamp(images + perturbation, 0, 1), perturbation

# Main Analysis
def analyze_attack(model, test_loader, epsilons, num_classes):
    results = []
    metrics = AttackMetrics(num_classes)
    
    for epsilon in epsilons:
        metrics.reset()
        for images, labels, filenames in tqdm(test_loader, desc=f"ε={epsilon:.2f}"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            # Untargeted attack
            images.requires_grad = True
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            model.zero_grad()
            loss.backward()
            perturbation = epsilon * images.grad.sign()
            perturbed_images = fgsm_attack(images, epsilon, images.grad.data)
            
            # Targeted attack - get least likely class for each sample
            with torch.no_grad():
                probs = F.softmax(outputs, dim=1)
                target_class = torch.argmin(probs, dim=1)  # Tensor of class indices
            targeted_images, _ = targeted_attack(model, images, labels, target_class, epsilon)
            
            # Get outputs
            with torch.no_grad():
                orig_output = model(images)
                untargeted_output = model(perturbed_images)
                targeted_output = model(targeted_images)
            
            metrics.update(orig_output, untargeted_output, labels, perturbation)
        
        results.append(metrics.calculate_metrics() | {'epsilon': epsilon})
    
    return results

# Visualization
def plot_metrics(results, output_dir):
    epsilons = [r['epsilon'] for r in results]
    
    plt.figure(figsize=(15, 10))
    metrics = ['asr', 'fooling_rate', 'confidence_drop', 'accuracy']
    titles = ['Attack Success Rate', 'Fooling Rate', 'Confidence Drop', 'Accuracy']
    
    for i, (metric, title) in enumerate(zip(metrics, titles), 1):
        plt.subplot(2, 2, i)
        plt.plot(epsilons, [r[metric] for r in results], 'o-')
        plt.xlabel('Epsilon')
        plt.ylabel(title)
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'attack_metrics.png'))
    plt.close()
    
    # Confusion Matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(results[-1]['confusion_matrix'], cmap='Blues')
    plt.title(f'Confusion Matrix (ε={epsilons[-1]:.2f})')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

if __name__ == "__main__":
    # Load model
    model = EfficientNet.from_pretrained('efficientnet-b0')
    model._fc = nn.Linear(model._fc.in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model = model.to(DEVICE)
    model.eval()
    
    # Data pipeline
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
    ])
    test_dataset = GTSRBDataset(TEST_CSV, TEST_IMG_DIR, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Run analysis
    epsilons = [0, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3]
    results = analyze_attack(model, test_loader, epsilons, NUM_CLASSES)
    
    # Save results
    pd.DataFrame(results).to_csv(os.path.join(OUTPUT_DIR, 'attack_results.csv'), index=False)
    plot_metrics(results, OUTPUT_DIR)
    
    # Generate report
    report = f"""Attack Analysis Report:
    
Total samples: {len(test_dataset)}
Epsilon values: {epsilons}

Key Metrics:
- Max Attack Success Rate: {max(r['asr'] for r in results):.1%}
- Min Accuracy Under Attack: {min(r['accuracy'] for r in results):.1%}
- Avg Confidence Drop: {np.mean([r['confidence_drop'] for r in results]):.3f}
"""
    with open(os.path.join(OUTPUT_DIR, 'report.txt'), 'w') as f:
        f.write(report)
    
    print(f"Analysis complete! Results saved to {OUTPUT_DIR}")