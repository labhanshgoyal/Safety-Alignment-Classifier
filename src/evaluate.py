import argparse
import json
import os
from typing import Any, Dict, Optional
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import get_adversarial_dataloader, get_dataloader, get_dataset
from model import load_model


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    description: str = "Evaluating",
) -> Dict[str, Any]:
    """Evaluates a classifier on a DataLoader and computes comprehensive metrics.

    Args:
        model: PyTorch model to evaluate.
        data_loader: DataLoader yielding batches with 'input_ids', 'attention_mask', and 'label'.
        device: Device to place tensors on.
        description: Label for the tqdm progress bar.

    Returns:
        Dictionary containing accuracy, precision, recall, f1, and confusion matrix breakdown.
    """
    model.eval()
    preds_list = []
    labels_list = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc=description, leave=True):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask)
            preds = torch.argmax(outputs, dim=1)

            preds_list.extend(preds.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())

    accuracy = float(accuracy_score(labels_list, preds_list))
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels_list, preds_list, average="binary", zero_division=0
    )

    cm = confusion_matrix(labels_list, preds_list)
    # Binary classification confusion matrix unpacking: [[TN, FP], [FN, TP]]
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0

    metrics: Dict[str, Any] = {
        "accuracy": accuracy,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
        "total_samples": len(labels_list),
    }

    print(f"\n--- {description} Results ---")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1-score : {metrics['f1']:.4f}")
    print(
        f"TN: {tn} | FP: {fp} | FN: {fn} | TP: {tp}"
    )

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a toxicity classifier with optional adversarial testing.")
    parser.add_argument("--data_directory", type=str, required=True, help="Path to dataset directory.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model weights.")
    parser.add_argument("--dataset_size", type=int, default=1000, help="Test set size (default: 1000).")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size (default: 64).")
    parser.add_argument("--adversarial", action="store_true", help="Also evaluate on adversarial test set.")
    parser.add_argument("--output_path", type=str, default=None, help="Optional JSON path to save evaluation metrics.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Use the centralized load_model utility
    model = load_model(args.model_path, device=device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # Clean test dataset & evaluation
    test_dataset = get_dataset(args.data_directory, dataset_size=args.dataset_size, split="test", class_balancing=True)
    test_loader = get_dataloader(test_dataset, batch_size=args.batch_size, shuffle=False)
    results: Dict[str, Any] = {
        "standard_evaluation": evaluate(model, test_loader, device, description="Standard Evaluation")
    }

    # Optional adversarial evaluation
    if args.adversarial:
        _, adv_dataloader = get_adversarial_dataloader(
            test_dataset, batch_size=args.batch_size, transform_prob=0.4, fully_adversarial=True
        )
        results["adversarial_evaluation"] = evaluate(
            model, adv_dataloader, device, description="Adversarial Evaluation"
        )

    # Save to JSON file if path specified
    if args.output_path:
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        with open(args.output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)
        print(f"\nResults successfully saved to {args.output_path}")