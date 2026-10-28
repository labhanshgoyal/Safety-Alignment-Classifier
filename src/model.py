import os
from typing import Optional
import torch
import torch.nn as nn
from transformers import RobertaModel


class ToxicityClassifier(nn.Module):
    """RoBERTa-based binary toxicity classification model.

    Passes text through a pretrained RoBERTa backbone, extracts the CLS pooled output,
    applies dropout regularization, and projects to classification logits.
    """

    def __init__(
        self,
        pretrained_model_name: str = "roberta-base",
        dropout_rate: float = 0.3,
        num_classes: int = 2,
    ):
        super(ToxicityClassifier, self).__init__()
        self.roberta = RobertaModel.from_pretrained(pretrained_model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(self.roberta.config.hidden_size, num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass through RoBERTa and linear classification head.

        Args:
            input_ids: Token ID tensor of shape (batch_size, sequence_length).
            attention_mask: Attention mask tensor of shape (batch_size, sequence_length).

        Returns:
            Raw logits tensor of shape (batch_size, num_classes).
        """
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        logits = self.fc(self.dropout(pooled_output))
        return logits


def save_model(model: nn.Module, model_path: str) -> None:
    """Saves model weights, unwrapping DataParallel if necessary.

    Args:
        model: PyTorch model or DataParallel-wrapped model.
        model_path: Destination file path for saved weights.
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    # If model is wrapped in DataParallel, unwrap it to avoid 'module.' key prefix
    state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    torch.save(state_dict, model_path)
    print(f"Model saved to {model_path}")


def load_model(
    model_path: str,
    device: Optional[torch.device] = None,
    **model_kwargs,
) -> ToxicityClassifier:
    """Loads a ToxicityClassifier model from saved weights.

    Args:
        model_path: Path to the .pt or .pth weights file.
        device: Torch device to load the model onto. Defaults to CPU.
        **model_kwargs: Additional arguments passed to ToxicityClassifier constructor.

    Returns:
        ToxicityClassifier with loaded weights placed on target device.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ToxicityClassifier(**model_kwargs)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)

    # Clean 'module.' prefix if saved from DataParallel in older versions
    cleaned_state_dict = {
        key.replace("module.", ""): value for key, value in state_dict.items()
    }
    model.load_state_dict(cleaned_state_dict)
    model.to(device)
    return model