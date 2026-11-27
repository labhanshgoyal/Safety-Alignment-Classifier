from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
import yaml

@dataclass
class DataConfig:
    directory: str = "./jigsaw_toxicity_data"
    train_size: int = 5000
    test_size: int = 1000
    class_balancing: bool = True
    max_seq_length: int = 256

@dataclass
class ModelConfig:
    name: str = "roberta-base"
    dropout: float = 0.3
    num_classes: int = 2

@dataclass
class TrainingConfig:
    epochs: int = 3
    batch_size: int = 64
    learning_rate: float = 2.0e-5
    optimizer: str = "adamw"

@dataclass
class AdversarialConfig:
    transform_prob: float = 0.4
    apply_back_translation: bool = False
    fully_adversarial: bool = False

@dataclass
class RLConfig:
    epochs: int = 4
    learning_rate: float=3.0e-4
    epsilon: float=0.2
    policy_hidden_dim: int =128

@dataclass
class PathsConfig:
    classifier_model: str="./models/classifier.pt"
    policy_model: str="./models/policy.pt"
    results: str = "./results/"

@dataclass
class AppConfig:
    """Master configuration class aggregating all sub-configurations."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    adversarial: AdversarialConfig = field(default_factory=AdversarialConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    @classmethod
    def from_yaml(cls, yaml_path: str = "config/defaults.yaml") -> "AppConfig":
        """Loads configuration from a YAML file, falling back to defaults for any missing keys.
        Args:
            yaml_path: File path to the YAML configuration file.
        Returns:
            Instantiated AppConfig object with typed configuration values.
        """
        path = Path(yaml_path)
        if not path.is_file():
            print(f"Warning: Config file '{yaml_path}' not found. Using default configurations.")
            return cls()
        with open(path, "r", encoding="utf-8") as f:
            raw_dict = yaml.safe_load(f) or {}
        return cls(
            data=DataConfig(**raw_dict.get("data", {})),
            model=ModelConfig(**raw_dict.get("model", {})),
            training=TrainingConfig(**raw_dict.get("training", {})),
            adversarial=AdversarialConfig(**raw_dict.get("adversarial", {})),
            rl=RLConfig(**raw_dict.get("rl", {})),
            paths=PathsConfig(**raw_dict.get("paths", {})),
        )