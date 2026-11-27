import argparse
from typing import Any, Dict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from config import AppConfig
from dataset import get_adversarial_dataloader, get_dataset
from model import ToxicityClassifier, load_model, save_model


class PolicyNetwork(nn.Module):
    """PPO policy network that outputs adjustment logits for the base classifier."""

    def __init__(self, input_dim: int = 2, hidden_dim: int = 128, num_classes: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        return logits


class PPOAgent:
    """Proximal Policy Optimization (PPO) agent for robust policy tuning."""

    def __init__(self, classifier: nn.Module, policy_network: nn.Module, epsilon: float = 0.2):
        self.classifier = classifier
        self.policy = policy_network
        self.epsilon = epsilon

    def select_action(self, state: torch.Tensor):
        logits = self.policy(state)
        probs = F.softmax(logits, dim=-1)
        action = torch.multinomial(probs, num_samples=1)
        return action, probs.gather(1, action)

    def compute_loss(self, old_probs: torch.Tensor, new_probs: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
        rewards = rewards.unsqueeze(1)
        ratio = new_probs / (old_probs + 1e-8)
        clipped_ratio = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
        loss = -torch.min(ratio * rewards, clipped_ratio * rewards).mean()
        return loss

    def compute_reward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor, adversarial: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logits = self.classifier(input_ids, attention_mask)
            predictions = logits.argmax(dim=-1)

        correct = (predictions == labels).float()
        reward = torch.where(adversarial, torch.where(correct.bool(), 1.5, -1.0), correct)
        return reward


def train_rl_classifier(
    model: nn.Module,
    policy_network: nn.Module,
    adv_dataloader: Any,
    device: torch.device,
    epochs: int = 4,
    lr: float = 3e-4,
    epsilon: float = 0.2,
) -> PPOAgent:
    """Trains the PPO policy network against adversarial dataloader batches."""
    policy_network.train()
    agent = PPOAgent(model, policy_network, epsilon=epsilon)
    optimizer = optim.Adam(policy_network.parameters(), lr=lr)

    for epoch in range(epochs):
        loop = tqdm(adv_dataloader, leave=True)
        total_loss = 0.0
        correct = 0
        total = 0
        total_reward = 0.0

        for batch in loop:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            is_adversarial = batch["is_adversarial"].to(device)

            optimizer.zero_grad()
            states = model(input_ids, attention_mask).detach()
            actions, old_probs = agent.select_action(states)

            rewards = agent.compute_reward(input_ids, attention_mask, labels, is_adversarial)
            new_logits = agent.policy(states)
            new_probs = F.softmax(new_logits, dim=-1).gather(1, actions)
            loss = agent.compute_loss(old_probs, new_probs, rewards)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_reward += rewards.sum().item()
            preds = torch.argmax(states, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            loop.set_description(f"RL Epoch [{epoch + 1}/{epochs}]")
            loop.set_postfix(
                policyloss=total_loss / total,
                accuracy=correct / total,
                rewards=total_reward / total,
            )

    return agent


def evaluate_rl_classifier(
    classifier: nn.Module,
    policy_network: nn.Module,
    dataloader: Any,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluates the composite base classifier + policy network."""
    classifier.eval()
    policy_network.eval()

    preds_list = []
    labels_list = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = classifier(input_ids, attention_mask)
            policy_adjustments = policy_network(logits)
            adjusted_logits = logits + policy_adjustments
            predictions = adjusted_logits.argmax(dim=-1)

            preds_list.extend(predictions.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())

    accuracy = float(accuracy_score(labels_list, preds_list))
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels_list, preds_list, average="binary", zero_division=0
    )

    metrics = {
        "accuracy": accuracy,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }

    print(f"\n--- RL Robustness Evaluation Results ---")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Precision    : {precision:.4f}")
    print(f"Recall       : {recall:.4f}")
    print(f"F1-score     : {f1:.4f}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a RL policy network for robust toxicity classification")
    parser.add_argument("--config", type=str, default="config/defaults.yaml", help="Path to YAML config file.")
    parser.add_argument("--data_directory", type=str, default=None, help="Override path to dataset directory.")
    parser.add_argument("--classifier_model_path", type=str, default=None, help="Override path to load classifier model.")
    parser.add_argument("--policy_model_path", type=str, default=None, help="Override path to save policy model.")
    parser.add_argument("--epochs", type=int, default=None, help="Override RL training epochs.")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate.")
    args = parser.parse_args()

    cfg = AppConfig.from_yaml(args.config)

    data_dir = args.data_directory or cfg.data.directory
    classifier_model_path = args.classifier_model_path or cfg.paths.classifier_model
    policy_model_path = args.policy_model_path or cfg.paths.policy_model
    epochs = args.epochs if args.epochs is not None else cfg.rl.epochs
    lr = args.lr if args.lr is not None else cfg.rl.learning_rate

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loaded config from {args.config} (RL epochs={epochs}, lr={lr}, epsilon={cfg.rl.epsilon})")

    model = load_model(classifier_model_path, device=device)

    policy_network = PolicyNetwork(
        input_dim=cfg.model.num_classes,
        hidden_dim=cfg.rl.policy_hidden_dim,
        num_classes=cfg.model.num_classes,
    )
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        policy_network = torch.nn.DataParallel(policy_network)
    model = model.to(device)
    policy_network = policy_network.to(device)

    # Train RL policy
    train_dataset = get_dataset(data_dir, dataset_size=cfg.data.train_size, split="train", class_balancing=cfg.data.class_balancing)
    _, adv_train_dataloader = get_adversarial_dataloader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        transform_prob=cfg.adversarial.transform_prob,
    )
    agent = train_rl_classifier(
        model,
        policy_network,
        adv_train_dataloader,
        device=device,
        epochs=epochs,
        lr=lr,
        epsilon=cfg.rl.epsilon,
    )
    save_model(agent.policy, policy_model_path)

    # Evaluate RL policy robustness
    test_dataset = get_dataset(data_dir, dataset_size=cfg.data.test_size, split="test", class_balancing=cfg.data.class_balancing)
    _, adv_test_dataloader = get_adversarial_dataloader(
        test_dataset,
        batch_size=cfg.training.batch_size,
        transform_prob=cfg.adversarial.transform_prob,
    )
    evaluate_rl_classifier(model, agent.policy, adv_test_dataloader, device)