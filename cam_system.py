import os
import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision import transforms, datasets
from PIL import Image

# =========================
# Paths and Setup
# =========================
base_dir = r"C:\Users\dylan\downloads"
model_path = os.path.join(base_dir, "simple_cnn_model.pth")
train_dir = os.path.join(base_dir, "archive", "train")

# =========================
# Load class names from train folder
# =========================
train_dataset = datasets.ImageFolder(train_dir)
class_to_idx = train_dataset.class_to_idx
idx_to_class = {v: k for k, v in class_to_idx.items()}
num_classes = len(idx_to_class)

# =========================
# Model Definition
# =========================
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# =========================
# Load Trained Model
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN(num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# =========================
# Image Transform (Same as training)
# =========================
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# =========================
# Webcam Prediction Loop
# =========================
def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("❌ Could not open webcam.")
        return

    print("🔴 Webcam started. Press 'Q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❗ Failed to grab frame")
            break

        # Convert frame to PIL image
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted = torch.max(outputs, 1)
            predicted_label = idx_to_class[predicted.item()]

        # Display prediction on frame
        cv2.putText(frame, f"Predicted: {predicted_label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Street Sign Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("🛑 Exiting...")
            break

    cap.release()
    cv2.destroyAllWindows()

# =========================
# Run the main function
# =========================
if __name__ == "__main__":
    main()
