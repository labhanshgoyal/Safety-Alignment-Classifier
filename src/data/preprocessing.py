import re
from typing import Any, Dict, Tuple
from datasets import Dataset, load_dataset
import pandas as pd


def clean_text(text: str) -> str:
    """Standardizes text by normalizing whitespaces and stripping unprintable characters.

    Args:
        text: Raw input string.

    Returns:
        Cleaned, normalized string.
    """
    if not isinstance(text, str):
        return ""
    # Replace multiple spaces, tabs, and newlines with a single space
    text = re.sub(r"\s+", " ", text)
    # Strip leading/trailing whitespace
    return text.strip()


def preprocess_toxicity_label(example: Dict[str, Any]) -> Dict[str, Any]:
    """Binarizes the continuous toxicity score into binary 0 (safe) or 1 (toxic).

    Args:
        example: Dataset record dictionary containing 'toxic' score.

    Returns:
        Dictionary with added integer 'label' key.
    """
    example["label"] = 1 if example.get("toxic", 0.0) >= 0.5 else 0
    if "comment_text" in example:
        example["text"] = clean_text(example["comment_text"])
    elif "text" in example:
        example["text"] = clean_text(example["text"])
    return example


def balance_classes(df: pd.DataFrame, target_size: int, random_state: int = 42) -> pd.DataFrame:
    """Balances dataset 50/50 between toxic (1) and non-toxic (0) samples.

    Args:
        df: Pandas DataFrame containing a 'label' column.
        target_size: Total number of samples desired in the balanced output.
        random_state: Seed for reproducible sampling.

    Returns:
        Shuffled DataFrame containing equal number of class 0 and class 1 samples.
    """
    df_label_0 = df[df["label"] == 0]
    df_label_1 = df[df["label"] == 1]

    samples_per_class = target_size // 2
    n_0 = min(len(df_label_0), samples_per_class)
    n_1 = min(len(df_label_1), samples_per_class)

    df_balanced = pd.concat([
        df_label_0.sample(n=n_0, random_state=random_state),
        df_label_1.sample(n=n_1, random_state=random_state),
    ])

    return df_balanced.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def load_toxicity_dataset(directory_path: str) -> Tuple[Any, Any]:
    """Loads and preprocesses train/test splits for the Jigsaw toxicity dataset.

    Args:
        directory_path: Directory path where the Jigsaw data is stored.

    Returns:
        Tuple of (train_dataset, test_dataset) mapped with binary labels.
    """
    dataset = load_dataset("jigsaw_toxicity_pred", data_dir=directory_path, trust_remote_code=True)
    dataset = dataset.map(preprocess_toxicity_label)
    return dataset["train"], dataset["test"]