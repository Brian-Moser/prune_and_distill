import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
import os
import json
import numpy as np
from collections import defaultdict

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load pre-trained ResNet18 model
model = models.resnet18(pretrained=True)
model = model.to(device)
model.eval()

# Loss function
criterion = nn.CrossEntropyLoss()

# ImageNet normalization
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load ImageNet dataset
data_dir = '/ds-sds/images/imagenet/train'
dataset = datasets.ImageFolder(root=data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# Store losses class-wise
class_losses = defaultdict(list)

# Iterate over the dataset
for i, (inputs, labels) in enumerate(dataloader):
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = model(inputs)
    
    loss = criterion(outputs, labels).item()
    
    img_path = dataset.imgs[i][0]
    class_losses[labels.item()].append((img_path, loss))

# Sort and save the top percentages
def save_top_losses(class_losses, percentages, file_prefix):
    os.makedirs(file_prefix, exist_ok=True)
    for cls, losses in class_losses.items():
        losses.sort(key=lambda x: x[1])  # Sort by loss value ascending
        total_images = len(losses)

        for percent in percentages:
            n = int(percent * total_images / 100)
            top_losses_asc = losses[:n]
            top_losses_desc = losses[-n:]

            # Save to JSON (ascending order)
            with open(f'{file_prefix}/class_{cls}_top_{percent}_asc.json', 'w') as f:
                json.dump([img[0] for img in top_losses_asc], f)
            
            # Save to JSON (descending order)
            with open(f'{file_prefix}/class_{cls}_top_{percent}_desc.json', 'w') as f:
                json.dump([img[0] for img in top_losses_desc], f)

# Define percentages to save
percentages = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90]
save_top_losses(class_losses, percentages, 'imagenet_losses')