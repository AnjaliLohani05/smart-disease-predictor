"""
Pneumonia detection training script using PyTorch.
Supports CUDA for GPU acceleration.

Usage:
    python src/training/train_pneumonia.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import logging
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler()]
)

def get_config():
    config_path = "config/train_config.yaml"
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            full_config = yaml.safe_load(f)
            return full_config.get("image", {}).get("pneumonia", {})
    return {
        "epochs": 15,
        "batch_size": 32,
        "learning_rate": 0.001,
        "img_size": 128
    }

def train():
    cfg = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Data directory
    data_dir = "chest_xray"
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    test_dir = os.path.join(data_dir, "test")

    if not os.path.exists(train_dir):
        logging.error(f"Training directory {train_dir} not found.")
        return

    # Transforms (Gray-scale images in original, but VGG/ResNet expect 3-channel)
    train_transform = transforms.Compose([
        transforms.Resize((cfg["img_size"], cfg["img_size"])),
        transforms.Grayscale(num_output_channels=3), # Convert to 3 channel for pretrained models
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((cfg["img_size"], cfg["img_size"])),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Datasets
    train_ds = datasets.ImageFolder(train_dir, transform=train_transform)
    val_ds = datasets.ImageFolder(val_dir, transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=2)

    logging.info(f"Classes: {train_ds.class_to_idx}")

    # Model: Transfer learning with ResNet18
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    # Freeze base
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace final fully connected layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, len(train_ds.classes))
    )
    
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=cfg["learning_rate"])

    # Training Loop
    best_acc = 0.0
    for epoch in range(cfg["epochs"]):
        logging.info(f"Epoch {epoch+1}/{cfg['epochs']}")
        
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            
        train_loss = running_loss / len(train_ds)
        train_acc = running_corrects.double() / len(train_ds)
        
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
        
        val_loss = val_loss / len(val_ds)
        val_acc = val_corrects.double() / len(val_ds)
        
        logging.info(f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs("models", exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'class_to_idx': train_ds.class_to_idx
            }, "models/pneumonia.pth")
            logging.info(f"Best model saved with accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    train()
