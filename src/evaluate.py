import argparse
import json
import os
from typing import Any, Callable, Dict, Optional
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset as TorchDataset
from tqdm import tqdm

from config import AppConfig
from data.adversarial import (
    homophone_substitution,
    keyboard_typo,
    random_char_insertion,
    synonym_replacement,
    word_order_shuffle,
)
from dataset import get_adversarial_dataloader, get_dataloader, get_dataset
from model import load_model


class SingleAttackDataset(TorchDataset):
    """Wrapper that applies a single specific perturbation function to evaluate vulnerability."""

    def __init__(self, base_dataset: Any, attack_fn: Callable[[str, float], str], prob: float = 0.4):
        self.base_dataset = base_dataset
        self.attack_fn = attack_fn
        self.prob = prob
        self.tokenizer = base_dataset.tokenizer
        self.max_length = getattr(base_dataset, "max_length", 256)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.base_dataset[idx]
        text = item["text"]
        label = item["label"]

        # Apply single attack function
        perturbed_text = self.attack_fn(text, self.prob)

        encoding = self.tokenizer(
            perturbed_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "text": perturbed_text,
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": label,
            "is_adversarial": True,
        }


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    description: str = "Evaluating",
) -> Dict[str, Any]:
    """Evaluates a classifier on a DataLoader and computes comprehensive metrics."""
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
    print(f"TN: {tn} | FP: {fp} | FN: {fn} | TP: {tp}")

    return metrics


def evaluate_per_attack(
    model: nn.Module,
    clean_dataset: Any,
    device: torch.device,
    batch_size: int = 64,
    transform_prob: float = 0.4,
) -> Dict[str, Dict[str, Any]]:
    """Evaluates the model against each individual perturbation vector separately."""
    attacks = {
        "Keyboard Typo": lambda t, p: keyboard_typo(t, p=p),
        "Char Insertion": lambda t, p: random_char_insertion(t, p=p),
        "Synonym Replace": lambda t, p: synonym_replacement(t, p=p),
        "Homophone Sub": lambda t, p: homophone_substitution(t, p=p),
        "Word Shuffle": lambda t, p: word_order_shuffle(t, window_size=2),
    }

    per_attack_results: Dict[str, Dict[str, Any]] = {}
    for attack_name, attack_fn in attacks.items():
        attack_dataset = SingleAttackDataset(clean_dataset, attack_fn=attack_fn, prob=transform_prob)
        attack_loader = DataLoader(attack_dataset, batch_size=batch_size, shuffle=False)
        per_attack_results[attack_name] = evaluate(
            model, attack_loader, device, description=f"Attack: {attack_name}"
        )

    return per_attack_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a toxicity classifier with optional adversarial testing.")
    parser.add_argument("--config", type=str, default="config/defaults.yaml", help="Path to YAML config file.")
    parser.add_argument("--data_directory", type=str, default=None, help="Override path to dataset directory.")
    parser.add_argument("--model_path", type=str, default=None, help="Override path to trained model weights.")
    parser.add_argument("--dataset_size", type=int, default=None, help="Override test set size.")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--adversarial", action="store_true", help="Also evaluate on combined adversarial test set.")
    parser.add_argument("--per_attack", action="store_true", help="Run per-attack breakdown analysis.")
    parser.add_argument("--output_path", type=str, default=None, help="Optional JSON path to save evaluation metrics.")
    args = parser.parse_args()

    cfg = AppConfig.from_yaml(args.config)

    data_dir = args.data_directory or cfg.data.directory
    model_path = args.model_path or cfg.paths.classifier_model
    dataset_size = args.dataset_size if args.dataset_size is not None else cfg.data.test_size
    batch_size = args.batch_size if args.batch_size is not None else cfg.training.batch_size
    transform_prob = cfg.adversarial.transform_prob

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Evaluating model: {model_path}")

    model = load_model(model_path, device=device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    test_dataset = get_dataset(data_dir, dataset_size=dataset_size, split="test", class_balancing=cfg.data.class_balancing)
    test_loader = get_dataloader(test_dataset, batch_size=batch_size, shuffle=False)

    results: Dict[str, Any] = {
        "standard_evaluation": evaluate(model, test_loader, device, description="Clean Baseline Evaluation")
    }

    if args.adversarial or args.per_attack:
        _, adv_dataloader = get_adversarial_dataloader(
            test_dataset,
            batch_size=batch_size,
            transform_prob=transform_prob,
            fully_adversarial=True,
        )
        results["combined_adversarial_evaluation"] = evaluate(
            model, adv_dataloader, device, description="All Attacks Combined"
        )

    if args.per_attack:
        results["per_attack_breakdown"] = evaluate_per_attack(
            model, test_dataset, device, batch_size=batch_size, transform_prob=transform_prob
        )

    output_path = args.output_path or os.path.join(cfg.paths.results, "evaluation_report.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"\nEvaluation metrics successfully saved to {output_path}")