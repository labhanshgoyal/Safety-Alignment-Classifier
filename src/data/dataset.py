from typing import Any, Dict, Optional, Tuple
from datasets import Dataset
import pandas as pd
import torch
from torch.utils.data import Dataloader, Dataset as TorchDataset
from transformers import RobertaTokenizer

from data.adversarial import apply_adversarial_perturbations
from data.preprocessing import balance_classes, load_toxicity_dataset

class ToxicityDataset(TorchDataset):
    """PyTorch Dataset wrapper for clean (non-adversarial) toxicity comments."""

    def __init__(
        self,
        dataset: Any,
        tokenizer_name: str = "roberta-base",
        max_length: int = 256,
    ):
        self.dataset = dataset
        self.tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.is_adversarial = torch.zeros(len(dataset), dtype=torch.bool)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item_data = self.dataset[idx]
        text = item_data.get("text", item_data.get("comment_text", ""))
        label = torch.tensor(item_data["label"], dtype=torch.long)
        is_adversarial = self.is_adversarial[idx]

        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "text": text,
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": label,
            "is_adversarial": is_adversarial,
        }


class AdversarialTextDataset(TorchDataset):
    """PyTorch Dataset wrapper applying stochastic adversarial transformations."""

    def __init__(
        self,
        original_dataset: Any,
        transform_prob: float = 0.3,
        apply_back_translation: bool = False,
        fully_adversarial: bool = False,
        tokenizer_name: str = "roberta-base",
        max_length: int = 256,
    ):
        self.original_dataset = original_dataset
        self.apply_back_translation = apply_back_translation
        self.tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

        n_samples = len(original_dataset)
        if fully_adversarial:
            self.transform_prob = torch.full((n_samples,), transform_prob)
        else:
            random_indices = torch.randint(2, (n_samples,))
            prob_values = torch.tensor([0.0, transform_prob])
            self.transform_prob = prob_values[random_indices]

        self.is_adversarial = self.transform_prob != 0

    def __len__(self) -> int:
        return len(self.original_dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item_data = self.original_dataset[idx]
        text = item_data.get("text", item_data.get("comment_text", ""))
        label = torch.tensor(item_data["label"], dtype=torch.long)

        prob = float(self.transform_prob[idx].item())
        if prob > 0:
            perturbed_text = apply_adversarial_perturbations(
                text=text,
                transform_prob=prob,
                apply_back_translation_flag=self.apply_back_translation,
            )
        else:
            perturbed_text = text

        is_adversarial = self.is_adversarial[idx]

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
            "is_adversarial": is_adversarial,
        }


def get_dataset(
    directory_path: str,
    dataset_size: int,
    split: str = "train",
    class_balancing: bool = True,
    tokenizer_name: str = "roberta-base",
    max_length: int = 256,
) -> ToxicityDataset:
    """Loads, balances, and wraps the Jigsaw dataset into a ToxicityDataset."""
    train_dataset, test_dataset = load_toxicity_dataset(directory_path)
    dataset = train_dataset if split == "train" else test_dataset

    if class_balancing:
        df = pd.DataFrame(dataset)
        df_balanced = balance_classes(df, target_size=dataset_size)
        dataset = Dataset.from_pandas(df_balanced)
    else:
        dataset = dataset.shuffle(seed=42).select(range(min(len(dataset), dataset_size)))

    print(f"Loaded {split} dataset size: {len(dataset)}")
    return ToxicityDataset(dataset, tokenizer_name=tokenizer_name, max_length=max_length)


def get_dataloader(
    dataset: Optional[TorchDataset],
    batch_size: int = 32,
    shuffle: bool = True,
    drop_last: bool = False,
) -> Optional[DataLoader]:
    """Creates a standard DataLoader for a ToxicityDataset."""
    if dataset is None:
        return None
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)


def get_adversarial_dataloader(
    original_dataset: Any,
    batch_size: int = 32,
    shuffle: bool = True,
    drop_last: bool = False,
    transform_prob: float = 0.4,
    apply_back_translation: bool = False,
    fully_adversarial: bool = False,
    tokenizer_name: str = "roberta-base",
    max_length: int = 256,
) -> Tuple[AdversarialTextDataset, DataLoader]:
    """Wraps an existing dataset with adversarial perturbations and returns a DataLoader."""
    adv_dataset = AdversarialTextDataset(
        original_dataset=original_dataset,
        transform_prob=transform_prob,
        apply_back_translation=apply_back_translation,
        fully_adversarial=fully_adversarial,
        tokenizer_name=tokenizer_name,
        max_length=max_length,
    )
    adv_dataloader = DataLoader(adv_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    return adv_dataset, adv_dataloader