import argparse
from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from dataset import get_dataloader, get_dataset
from model import ToxicityClassifier, save_model


def evaluate_validation(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Computes loss and accuracy on a validation DataLoader.

    Args:
        model: PyTorch classification model.
        val_loader: DataLoader for validation data.
        criterion: Loss function.
        device: Device to run evaluation on.

    Returns:
        Tuple of (avg_val_loss, val_accuracy).
    """
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * labels.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = val_loss / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy


def train(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epochs: int = 3,
    scheduler: Optional[object] = None,
    val_loader: Optional[DataLoader] = None,
) -> Dict[str, List[float]]:
    """Trains the model with optional per-epoch validation and learning rate scheduling.

    Args:
        model: PyTorch classifier to train.
        train_loader: DataLoader for training set.
        optimizer: Optimizer for backprop.
        criterion: Loss function.
        device: Device to place tensors on.
        epochs: Number of training epochs.
        scheduler: Optional learning rate scheduler.
        val_loader: Optional validation DataLoader for monitoring generalization.

    Returns:
        Dictionary containing epoch-level history for losses and accuracies.
    """
    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    for epoch in range(epochs):
        model.train()
        loop = tqdm(train_loader, leave=True)
        running_loss = 0.0
        correct = 0
        total = 0

        for batch in loop:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            running_loss += loss.item() * labels.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            loop.set_description(f"Epoch [{epoch + 1}/{epochs}]")
            loop.set_postfix(
                loss=running_loss / total,
                accuracy=correct / total,
            )

        epoch_train_loss = running_loss / total
        epoch_train_acc = correct / total
        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)

        # Validation phase at the end of each epoch
        if val_loader is not None:
            val_loss, val_acc = evaluate_validation(model, val_loader, criterion, device)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            print(
                f"Epoch [{epoch + 1}/{epochs}] -> "
                f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )
        else:
            print(
                f"Epoch [{epoch + 1}/{epochs}] -> "
                f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}"
            )

    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a toxicity classifier")
    parser.add_argument("--data_directory", type=str, required=True, help="Path to dataset directory.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to save trained weights.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs (default: 3).")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (default: 32).")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate (default: 2e-5).")
    parser.add_argument("--train_size", type=int, default=5000, help="Number of training samples (default: 5000).")
    parser.add_argument("--val_size", type=int, default=1000, help="Number of validation samples (default: 1000).")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load balanced train and validation datasets
    train_dataset = get_dataset(args.data_directory, dataset_size=args.train_size, split="train", class_balancing=True)
    val_dataset = get_dataset(args.data_directory, dataset_size=args.val_size, split="test", class_balancing=True)

    train_loader = get_dataloader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = get_dataloader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = ToxicityClassifier().to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # Learning rate schedule: linear warmup over 10% of total training steps, then linear decay
    total_training_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_training_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    history = train(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=args.epochs,
        scheduler=scheduler,
        val_loader=val_loader,
    )

    save_model(model, args.model_path)