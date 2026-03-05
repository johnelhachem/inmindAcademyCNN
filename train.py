# -------------------------------
#  CNN Training Script for CIFAR-10
#  Using ResNet9 (Improved from SimpleNet)
# -------------------------------

import os
import yaml
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# Import our model (ResNet9 architecture)
from model import ResNet9

# -------------------------------
#  Load configuration
# -------------------------------
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

hp    = config['hyperparameters']   # batch size, learning rate, etc.
paths = config['paths']             # where to save model, data, etc.

# CIFAR-10 dataset stats (used to normalize images)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

# -------------------------------
#  Data Augmentation for training
# -------------------------------
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),      # zoom/crop images slightly
    transforms.RandomHorizontalFlip(),         # flip images horizontally
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # color changes
    transforms.ToTensor(),                      # convert to PyTorch tensor
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),  # standardize
    transforms.RandomErasing(p=0.1),           # randomly hide part of image
])

# -------------------------------
#  Data Processing for eval/test
# -------------------------------
transform_eval = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

# -------------------------------
#  Load CIFAR-10 datasets and create DataLoaders
# -------------------------------
def get_loaders():
    os.makedirs(paths['train_dir'], exist_ok=True)
    os.makedirs(paths['test_dir'],  exist_ok=True)

    # Training dataset with augmentation
    dataset_train_aug   = datasets.CIFAR10(paths['train_dir'], train=True,
                                           download=True,  transform=transform_train)
    # Training dataset without augmentation (for validation)
    dataset_train_clean = datasets.CIFAR10(paths['train_dir'], train=True,
                                           download=False, transform=transform_eval)
    # Test dataset
    dataset_test        = datasets.CIFAR10(paths['test_dir'],  train=False,
                                           download=True,  transform=transform_eval)

    # Split training data into train and validation
    val_split = hp.get('val_split', 0.1)
    n_total   = len(dataset_train_aug)
    n_val     = int(n_total * val_split)
    n_train   = n_total - n_val

    generator = torch.Generator().manual_seed(42)  # reproducible split
    train_indices, val_indices = random_split(
        range(n_total), [n_train, n_val], generator=generator
    )

    from torch.utils.data import Subset
    dataset_train = Subset(dataset_train_aug,   list(train_indices))
    dataset_val   = Subset(dataset_train_clean, list(val_indices))

    bs  = hp['batch_size']
    nw  = hp['num_workers']
    pin = torch.cuda.is_available()

    loader_train = DataLoader(dataset_train, batch_size=bs, shuffle=True,
                              num_workers=nw, pin_memory=pin)
    loader_val   = DataLoader(dataset_val,   batch_size=bs, shuffle=False,
                              num_workers=nw, pin_memory=pin)
    loader_test  = DataLoader(dataset_test,  batch_size=bs, shuffle=False,
                              num_workers=nw, pin_memory=pin)
    return loader_train, loader_val, loader_test

# -------------------------------
#  Evaluation function
# -------------------------------
def evaluate(model, loader, criterion, device):
    model.eval()  # evaluation mode (disables dropout)
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():  # don't compute gradients
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)                 # forward pass
            loss    = criterion(outputs, labels)   # compute loss
            total_loss += loss.item() * labels.size(0)
            _, preds = torch.max(outputs, 1)       # predicted class
            total   += labels.size(0)
            correct += (preds == labels).sum().item()

    return total_loss / total, 100.0 * correct / total

# -------------------------------
#  Training function
# -------------------------------
def train(model, loader_train, loader_val, criterion, optimizer, scheduler, device):
    epochs      = hp['epochs']
    best_acc    = 0.0
    model_path  = paths['model_path']

    for epoch in range(epochs):
        model.train()  
        running_loss = 0.0

        # Progress bar for batches
        with tqdm(loader_train, desc=f"Epoch {epoch+1:>2}/{epochs}", leave=True, unit="batch") as pbar:
            for i, (inputs, labels) in enumerate(pbar):
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()       # clear previous gradients
                outputs = model(inputs)     # forward pass
                loss    = criterion(outputs, labels)
                loss.backward()             # backpropagation

                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # stabilize gradients

                optimizer.step()            # update weights
                scheduler.step()            # update learning rate

                running_loss += loss.item()
                pbar.set_postfix(loss=f"{running_loss/(i+1):.4f}")

        # Validation
        avg_train_loss = running_loss / len(loader_train)
        val_loss, val_acc = evaluate(model, loader_val, criterion, device)
        lr_now = optimizer.param_groups[0]['lr']

        print(f"  Train loss: {avg_train_loss:.4f} | "
              f"Val loss: {val_loss:.4f} | Val acc: {val_acc:.2f}% | "
              f"LR: {lr_now:.6f}")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(model.state_dict(), model_path)
            print(f"  ★ Best so far ({best_acc:.2f}%) — saved to {model_path}")

    print("\nTraining done!")

# -------------------------------
#  Main entry point
# -------------------------------
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load datasets
    loader_train, loader_val, loader_test = get_loaders()
    print(f"Train batches: {len(loader_train)} | "
          f"Val batches: {len(loader_val)} | "
          f"Test batches: {len(loader_test)}")

    # Build model
    model = ResNet9().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: ResNet9  |  Parameters: {n_params:,}")

    # Loss function with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Optimizer (AdamW)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=hp['lr'],
        weight_decay=hp['weight_decay']
    )

    # Learning rate scheduler (OneCycleLR)
    total_steps = len(loader_train) * hp['epochs']
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr       = hp['max_lr'],
        total_steps  = total_steps,
        pct_start    = 0.3,           # warm-up period
        anneal_strategy = 'cos',      # cosine decay
        div_factor   = 25,            
        final_div_factor = 1e4,       
    )

    # Train model
    train(model, loader_train, loader_val, criterion, optimizer, scheduler, device)

    # Load best model and evaluate on test set
    print(f"\nLoading best checkpoint from {paths['model_path']} ...")
    model.load_state_dict(torch.load(paths['model_path'], map_location=device))
    test_loss, test_acc = evaluate(model, loader_test, criterion, device)
    print(f"\nFinal result  →  Test loss: {test_loss:.3f} | Test acc: {test_acc:.2f}%")

    # Save final model
    torch.save(model.state_dict(), paths['model_path'])
    print(f"Model saved to {paths['model_path']}")

# Run the main function
if __name__ == '__main__':
    main()
