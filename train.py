import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import OneCycleLR
from model import ResNet9


# ── Config ────────────────────────────────────────────────────
with open("config.yaml") as f:
    config = yaml.safe_load(f)
hp = config["hyperparameters"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ── Normalisation (real CIFAR-10 stats, not generic 0.5) ──────
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)


# ── Transforms ────────────────────────────────────────────────
# Training: augmented. Validation/test: clean original photos.
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    transforms.RandomErasing(p=0.1),
])

transform_eval = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])


# ── Dataset — load twice so val split gets clean photos ───────
dataset_aug   = datasets.CIFAR10(root=config["paths"]["train_dir"], train=True,  download=True,  transform=transform_train)
dataset_clean = datasets.CIFAR10(root=config["paths"]["train_dir"], train=True,  download=False, transform=transform_eval)

total      = len(dataset_aug)
val_size   = int(total * hp["val_split"])
train_size = total - val_size

generator = torch.Generator().manual_seed(42)
train_idx, val_idx = random_split(range(total), [train_size, val_size], generator=generator)

dataset_train = Subset(dataset_aug,   train_idx)
dataset_val   = Subset(dataset_clean, val_idx)


# ── Loaders ───────────────────────────────────────────────────
loader_train = DataLoader(dataset_train, batch_size=hp["batch_size"], shuffle=True,  num_workers=hp["num_workers"], pin_memory=True)
loader_val   = DataLoader(dataset_val,   batch_size=256,              shuffle=False, num_workers=hp["num_workers"], pin_memory=True)

dataset_test = datasets.CIFAR10(root=config["paths"]["test_dir"], train=False, download=True, transform=transform_eval)
loader_test  = DataLoader(dataset_test, batch_size=256, shuffle=False, num_workers=hp["num_workers"], pin_memory=True)


# ── Model ─────────────────────────────────────────────────────
model = ResNet9(num_classes=10, dropout=0.2).to(DEVICE)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")


# ── Loss, Optimizer, Scheduler ────────────────────────────────
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

optimizer = optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])

scheduler = OneCycleLR(
    optimizer,
    max_lr=hp["max_lr"],
    total_steps=hp["epochs"] * len(loader_train),
    pct_start=0.3,
    anneal_strategy="cos",
    div_factor=25,
    final_div_factor=1e4,
)


# ── Training Loop ─────────────────────────────────────────────
os.makedirs(os.path.dirname(config["paths"]["model_path"]), exist_ok=True)
model_path  = config["paths"]["model_path"]
best_val_acc = 0.0

for epoch in range(hp["epochs"]):

    # Training phase
    model.train()
    train_loss, train_correct, train_total = 0.0, 0, 0

    for images, labels in loader_train:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()  # must be inside batch loop for OneCycleLR

        train_loss    += loss.item() * images.size(0)
        _, preds       = torch.max(outputs, 1)
        train_correct += (preds == labels).sum().item()
        train_total   += labels.size(0)

    train_acc = 100.0 * train_correct / train_total

    # Validation phase
    model.eval()
    val_correct, val_total = 0, 0

    with torch.no_grad():
        for images, labels in loader_val:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            _, preds = torch.max(model(images), 1)
            val_correct += (preds == labels).sum().item()
            val_total   += labels.size(0)

    val_acc = 100.0 * val_correct / val_total

    # Save best checkpoint
    saved = ""
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), model_path)
        saved = " ← saved"

    print(f"Epoch {epoch+1:2d}/{hp['epochs']} | Train: {train_acc:.2f}% | Val: {val_acc:.2f}%{saved}")


# ── Final Test ────────────────────────────────────────────────
model.load_state_dict(torch.load(model_path, map_location=DEVICE))
model.eval()

test_correct, test_total = 0, 0
with torch.no_grad():
    for images, labels in loader_test:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        _, preds = torch.max(model(images), 1)
        test_correct += (preds == labels).sum().item()
        test_total   += labels.size(0)

print(f"\nFinal Test Accuracy: {100.0 * test_correct / test_total:.2f}%")
