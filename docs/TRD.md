# Technical Requirements Document (TRD)
# Safety Alignment Classifier — Adversarially Robust Toxicity Detection

**Version:** 1.0  
**Date:** August 27, 2026  
**Author:** Lavish Bansal  
**Status:** Draft  

---

## 1. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SAFETY ALIGNMENT CLASSIFIER                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐    ┌──────────────────┐    ┌──────────────────┐   │
│  │  Data Layer   │───▶│  Training Layer   │───▶│  Evaluation     │   │
│  │              │    │                  │    │  Layer           │   │
│  │ • Jigsaw DS  │    │ • Baseline Train │    │ • Clean Eval    │   │
│  │ • Tokenizer  │    │ • Adversarial DS │    │ • Adversarial   │   │
│  │ • Balancing  │    │ • RL Policy Train│    │ • RL-Enhanced   │   │
│  │ • Adversarial│    │ • PPO Agent      │    │ • Comparative   │   │
│  └──────────────┘    └──────────────────┘    └──────────────────┘   │
│         │                     │                       │              │
│         ▼                     ▼                       ▼              │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                     Model Layer                               │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │   │
│  │  │ RoBERTa-base │  │ Policy Net   │  │ Combined         │   │   │
│  │  │ Classifier   │  │ (2-layer MLP)│  │ Inference        │   │   │
│  │  │ (768→2)      │  │ (2→128→2)    │  │ logits + adjust  │   │   │
│  │  └──────────────┘  └──────────────┘  └──────────────────┘   │   │
│  └──────────────────────────────────────────────────────────────┘   │
│         │                     │                       │              │
│         ▼                     ▼                       ▼              │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                   Presentation Layer                           │   │
│  │  ┌──────────┐  ┌──────────────┐  ┌────────────────────┐     │   │
│  │  │ CLI      │  │ Streamlit    │  │ FastAPI            │     │   │
│  │  │ Interface│  │ Dashboard    │  │ REST API           │     │   │
│  │  └──────────┘  └──────────────┘  └────────────────────┘     │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Technology Stack

### 2.1 Core ML Stack

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| Language | Python | 3.9+ | Primary development language |
| Deep Learning Framework | PyTorch | 2.0+ | Model training & inference |
| Transformer Models | HuggingFace Transformers | 4.30+ | RoBERTa-base, MarianMT |
| NLP | SpaCy | 3.5+ | Synonym replacement via word vectors |
| Datasets | HuggingFace Datasets | 2.14+ | Jigsaw dataset loading |
| Tokenization | RobertaTokenizer | (via transformers) | Text tokenization |

### 2.2 Data & Metrics Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Data Manipulation | Pandas | DataFrame operations for class balancing |
| Metrics | scikit-learn | accuracy, precision, recall, F1, confusion matrix |
| Metrics (alt) | torchmetrics | PyTorch-native metrics |
| Visualization | matplotlib, seaborn | Plots, confusion matrices, ROC curves |

### 2.3 Infrastructure Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Experiment Tracking | TensorBoard / W&B | Loss curves, metrics logging |
| Web Demo | Streamlit or Gradio | Interactive demo |
| API (future) | FastAPI | REST API for inference |
| Environment | pip + requirements.txt | Dependency management |
| Version Control | Git + GitHub | Code versioning |
| CI/CD | GitHub Actions | Automated testing, linting |

---

## 3. Detailed Component Design

### 3.1 Data Layer

#### 3.1.1 Dataset Loading (`src/data/dataset.py`)

```python
# Current flow:
# 1. load_dataset("jigsaw_toxicity_pred") from HuggingFace
# 2. Binarize labels: toxic >= 0.5 → 1, else → 0
# 3. Class balancing: equal samples of toxic/non-toxic
# 4. Wrap in ToxicityDataset (PyTorch Dataset)
# 5. RoBERTa tokenization: max_length=256, padding, truncation
```

**Key Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `dataset_size` | 5000 (train), 1000 (test) | Number of samples after balancing |
| `max_length` | 256 | RoBERTa tokenizer max sequence length |
| `class_balancing` | True | Equal class distribution |
| `batch_size` | 32-64 | DataLoader batch size |

#### 3.1.2 Adversarial Perturbation Engine (`src/data/adversarial.py`)

**Attack Taxonomy:**

| Attack Type | Category | Probability | Complexity |
|-------------|----------|-------------|------------|
| Keyboard Typo | Character-level | 0.1 | O(n) per word |
| Random Char Insertion | Character-level | 0.1 | O(n) per word |
| Synonym Replacement | Word-level | 0.2 | O(V) per word (SpaCy vocab scan) |
| Homophone Substitution | Word-level | 0.2 | O(1) per word (dictionary lookup) |
| Word Order Shuffle | Sentence-level | N/A | O(n) per sentence |
| Back Translation | Sentence-level | Optional | O(1) (model inference, slow) |

**Current Bugs to Fix:**
1. `AdversarialTextDataset.adversarial_transform()` uses `self.transform_prob` as both a float and a tensor — runtime error
2. `__getitem__` calls `self.adversarial_transform(text, transform_prob)` but method only takes `self, text` — signature mismatch
3. `fully_adversarial` mode uses `torch.full(len(dataset), ...)` instead of `torch.full((len(dataset),), ...)`

### 3.2 Model Layer

#### 3.2.1 Toxicity Classifier (`src/models/classifier.py`)

```
Architecture:
┌──────────────────────────────────────┐
│          RoBERTa-base                │
│  (12 layers, 768 hidden, 125M params)│
├──────────────────────────────────────┤
│         [CLS] Pooler Output          │
│            (768-dim)                 │
├──────────────────────────────────────┤
│          Dropout (p=0.3)             │
├──────────────────────────────────────┤
│        Linear(768 → 2)              │
│         (logits output)              │
└──────────────────────────────────────┘
```

**Training Configuration:**
| Hyperparameter | Value |
|---------------|-------|
| Optimizer | AdamW |
| Learning Rate | 2e-5 |
| Epochs | 3 |
| Loss Function | CrossEntropyLoss |
| Dropout | 0.3 |
| Max Seq Length | 256 |
| Batch Size | 32 |
| Multi-GPU | DataParallel (if available) |

#### 3.2.2 Policy Network (`src/models/policy.py`)

```
Architecture:
┌──────────────────────────────────────┐
│    Classifier Logits (2-dim)         │
├──────────────────────────────────────┤
│        Linear(2 → 128)              │
│            ReLU                      │
├──────────────────────────────────────┤
│        Linear(128 → 2)              │
│     (policy adjustments)             │
└──────────────────────────────────────┘

Combined Inference:
  final_logits = classifier_logits + policy_adjustments
  prediction = argmax(final_logits)
```

#### 3.2.3 PPO Agent (`src/rl/ppo_agent.py`)

**Algorithm:**
```
For each epoch:
  For each batch of (clean + adversarial) examples:
    1. Get classifier logits (frozen): states = classifier(input_ids, attn_mask)
    2. Select action via policy: action, old_probs = policy.select_action(states)
    3. Compute reward:
       - Correct clean prediction:       reward = 1.0
       - Correct adversarial prediction:  reward = 1.5  (incentivize robustness)
       - Incorrect prediction:            reward = -1.0
    4. Compute new probabilities: new_probs = softmax(policy(states))
    5. PPO clipped loss: L = -min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)
    6. Update policy network
```

**PPO Hyperparameters:**
| Parameter | Value | Description |
|-----------|-------|-------------|
| `epsilon` | 0.2 | PPO clipping parameter |
| `lr` | 3e-4 | Policy network learning rate |
| `epochs` | 4 | RL training epochs |
| `optimizer` | Adam | Policy optimizer |
| `reward_adversarial_correct` | 1.5 | Bonus for correct adversarial classification |
| `reward_correct` | 1.0 | Reward for correct clean classification |
| `reward_incorrect` | -1.0 | Penalty for misclassification |

**Current Bugs to Fix:**
1. `compute_reward()` takes `self, input_ids, attention_mask, labels, adversarial` but is called with `self, model, input_ids, attention_mask, labels, is_adversarial` — extra `model` argument
2. `train_rl_classifier()` uses global `device` instead of passing it as a parameter
3. `evaluate_rl_classifier()` accesses `batch["labels"]` but dataset uses key `"label"` — KeyError
4. `get_adversarial_dataloader()` returns `(dataset, dataloader)` tuple but `rl_policy.py` line 149 only captures one return value

### 3.3 Evaluation Layer

#### 3.3.1 Metrics Suite

| Metric | Formula | Use Case |
|--------|---------|----------|
| Accuracy | (TP+TN)/(TP+TN+FP+FN) | Overall correctness |
| Precision | TP/(TP+FP) | "Of predicted toxic, how many actually are?" |
| Recall | TP/(TP+FN) | "Of actually toxic, how many were caught?" |
| F1 Score | 2 × (P×R)/(P+R) | Harmonic mean of precision and recall |
| AUC-ROC | Area under ROC curve | Threshold-independent performance |
| Confusion Matrix | 2×2 matrix | Visual classification breakdown |

#### 3.3.2 Evaluation Scenarios

```
Scenario 1: Clean Evaluation
  Input: Clean test data → Classifier → Metrics

Scenario 2: Adversarial Evaluation
  Input: Perturbed test data → Classifier → Metrics
  (Expected: significant accuracy drop)

Scenario 3: RL-Enhanced Evaluation
  Input: Perturbed test data → Classifier → Policy Network → Adjusted → Metrics
  (Expected: recovered accuracy)

Scenario 4: Per-Attack Evaluation (NEW)
  For each attack type:
    Input: Specific attack test data → Classifier → Metrics
    Output: Attack-specific robustness breakdown
```

---

## 4. Project Structure (Proposed Refactored)

```
Safety-Alignment-Classifier/
├── docs/                              # Documentation
│   ├── PRD.md                         # Product Requirements
│   ├── TRD.md                         # Technical Requirements (this file)
│   ├── FLOW.md                        # Web/App Flow
│   ├── DESIGN.md                      # Design / UI-UX Document
│   ├── SCHEMA.md                      # Data Schema
│   └── IMPLEMENTATION_PLAN.md         # Implementation Plan
│
├── src/                               # Source code
│   ├── __init__.py
│   ├── config.py                      # Configuration management (YAML/dataclass)
│   │
│   ├── data/                          # Data layer
│   │   ├── __init__.py
│   │   ├── dataset.py                 # ToxicityDataset, data loading
│   │   ├── adversarial.py             # Adversarial perturbation engine
│   │   ├── preprocessing.py           # Text preprocessing utilities
│   │   └── dataloader.py              # DataLoader factory functions
│   │
│   ├── models/                        # Model layer
│   │   ├── __init__.py
│   │   ├── classifier.py              # RoBERTa-based ToxicityClassifier
│   │   └── policy.py                  # PolicyNetwork for RL
│   │
│   ├── rl/                            # Reinforcement Learning layer
│   │   ├── __init__.py
│   │   ├── ppo_agent.py               # PPOAgent with reward shaping
│   │   └── trainer.py                 # RL training loop
│   │
│   ├── evaluation/                    # Evaluation layer
│   │   ├── __init__.py
│   │   ├── evaluator.py               # Main evaluation logic
│   │   ├── metrics.py                 # Metrics computation
│   │   └── visualization.py           # Confusion matrices, plots
│   │
│   └── utils/                         # Utilities
│       ├── __init__.py
│       ├── logger.py                  # Logging configuration
│       ├── checkpoint.py              # Model save/load utilities
│       └── seed.py                    # Reproducibility (seed setting)
│
├── scripts/                           # Entry point scripts
│   ├── train.py                       # Baseline classifier training
│   ├── evaluate.py                    # Model evaluation
│   ├── train_rl.py                    # RL policy training
│   ├── inference.py                   # Single-sample inference
│   └── run_experiments.py             # Full experiment pipeline
│
├── app/                               # Web application
│   ├── streamlit_app.py               # Streamlit demo
│   └── api.py                         # FastAPI endpoints (future)
│
├── configs/                           # Configuration files
│   ├── default.yaml                   # Default hyperparameters
│   ├── experiment_baseline.yaml       # Baseline experiment config
│   └── experiment_rl.yaml             # RL experiment config
│
├── notebooks/                         # Jupyter notebooks
│   ├── 01_data_exploration.ipynb      # Dataset analysis
│   ├── 02_training_analysis.ipynb     # Training results analysis
│   └── 03_adversarial_analysis.ipynb  # Adversarial robustness analysis
│
├── tests/                             # Unit tests
│   ├── test_dataset.py
│   ├── test_model.py
│   ├── test_adversarial.py
│   └── test_rl.py
│
├── results/                           # Experiment results (git-tracked)
│   ├── figures/                       # Generated plots
│   ├── tables/                        # Results tables
│   └── logs/                          # TensorBoard logs
│
├── models/                            # Saved model checkpoints (git-ignored)
│   ├── classifier.pt
│   └── policy.pt
│
├── .github/                           # GitHub configuration
│   └── workflows/
│       └── ci.yml                     # CI/CD pipeline
│
├── requirements.txt                   # Python dependencies
├── setup.py                           # Package setup (optional)
├── Makefile                           # Common commands
├── README.md                          # Project README
├── LICENSE                            # Apache 2.0
└── .gitignore                         # Git ignore rules
```

---

## 5. Existing Code Bugs & Technical Debt

### 5.1 Critical Bugs (Must Fix)

| # | Bug | File | Line(s) | Impact | Fix |
|---|-----|------|---------|--------|-----|
| B1 | `adversarial_transform()` signature mismatch | `dataset.py` | L143-167 | Method takes `(self, text)` but `__getitem__` calls `(text, transform_prob)` | Add `transform_prob` parameter or pass via `self` |
| B2 | `torch.full()` missing tuple for size | `dataset.py` | L136 | `torch.full(len(dataset), ...)` → should be `torch.full((len(dataset),), ...)` | Add parentheses |
| B3 | `compute_reward()` extra `model` arg | `rl_policy.py` | L73 | Called with `model` as first arg, but method doesn't accept it | Remove extra arg from call site |
| B4 | Global `device` in `train_rl_classifier()` | `rl_policy.py` | L63-66 | Uses `device` from global scope — will crash if not defined | Add `device` parameter |
| B5 | `batch["labels"]` vs `batch["label"]` | `rl_policy.py` | L104 | `evaluate_rl_classifier` uses "labels" but dataset provides "label" | Change to "label" |
| B6 | `get_adversarial_dataloader()` returns tuple | `rl_policy.py` | L149,155 | Function returns `(dataset, dataloader)` but call site captures only one value | Unpack tuple properly |
| B7 | `get_dataset()` called with wrong args | `dataset.py` | L253-254 | `get_dataset(trainset, ...)` passes dataset object instead of directory path | Fix function call |

### 5.2 Code Quality Issues

| # | Issue | Severity | Fix |
|---|-------|----------|-----|
| Q1 | No `__init__.py` in `src/` | Medium | Add package init files |
| Q2 | No type hints | Low | Add throughout |
| Q3 | No docstrings | Medium | Add Google-style docstrings |
| Q4 | No logging (uses `print()` everywhere) | Medium | Replace with `logging` module |
| Q5 | No configuration management | High | Add YAML config + dataclass |
| Q6 | No error handling | Medium | Add try/except with meaningful messages |
| Q7 | No reproducibility controls | Medium | Add seed setting utility |
| Q8 | Hardcoded hyperparameters | High | Extract to config files |
| Q9 | No `requirements.txt` | High | Create with pinned versions |
| Q10 | Model tokenizer initialized per-sample in Dataset | High | Initialize once in `__init__` (already done, but verify) |

### 5.3 Missing Features (Technical Gaps)

| # | Missing Feature | Priority | Effort |
|---|----------------|----------|--------|
| M1 | requirements.txt | P0 | 1 hour |
| M2 | Configuration system | P0 | 2 hours |
| M3 | Proper logging | P1 | 2 hours |
| M4 | Unit tests | P1 | 4 hours |
| M5 | Experiment tracking (TensorBoard) | P1 | 3 hours |
| M6 | Confusion matrix visualization | P1 | 2 hours |
| M7 | Per-attack-type evaluation | P1 | 3 hours |
| M8 | Inference script | P1 | 2 hours |
| M9 | Streamlit demo | P1 | 6 hours |
| M10 | Model export (ONNX) | P2 | 2 hours |
| M11 | FastAPI endpoint | P2 | 4 hours |
| M12 | GitHub Actions CI | P2 | 2 hours |

---

## 6. Dependencies

### 6.1 Required Dependencies (requirements.txt)

```
# Core ML
torch>=2.0.0
transformers>=4.30.0
datasets>=2.14.0
accelerate>=0.20.0
torchmetrics>=1.0.0

# NLP
spacy>=3.5.0

# Data Processing
pandas>=2.0.0
numpy>=1.24.0

# Metrics & Evaluation
scikit-learn>=1.3.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0

# Training Utilities
tqdm>=4.65.0

# Web Demo
streamlit>=1.25.0
# gradio>=3.40.0  # alternative

# API (optional)
# fastapi>=0.100.0
# uvicorn>=0.23.0

# Experiment Tracking
# tensorboard>=2.13.0
# wandb>=0.15.0

# Development
# pytest>=7.4.0
# black>=23.0.0
# flake8>=6.0.0
# mypy>=1.4.0

# Kaggle API
kaggle>=1.5.0
```

### 6.2 SpaCy Model Downloads
```bash
python -m spacy download en_core_web_md    # For synonym replacement
python -m spacy download en_core_web_lg    # Optional, better vectors
```

---

## 7. Hardware Requirements

### 7.1 Development
| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 4 cores | 8+ cores |
| RAM | 8 GB | 16+ GB |
| GPU | None (CPU fallback) | NVIDIA GPU with 6+ GB VRAM |
| Storage | 5 GB | 10+ GB (models + data) |

### 7.2 Training Estimates
| Task | GPU (RTX 3060) | CPU (i7) |
|------|---------------|----------|
| Baseline Training (5K, 3 epochs) | ~15 min | ~3 hours |
| Adversarial Evaluation (1K) | ~5 min | ~45 min |
| RL Training (5K, 4 epochs) | ~20 min | ~4 hours |
| Full Pipeline | ~45 min | ~8 hours |

---

## 8. Security Considerations

| Concern | Mitigation |
|---------|------------|
| Kaggle API key exposure | Use `~/.kaggle/kaggle.json` with `chmod 600`; never commit to repo |
| Model weights contain learned patterns from toxic text | Add disclaimer; don't expose raw model internals |
| Adversarial techniques could be misused | Focus on defense; document responsible use |
| Dataset contains harmful language | Required for toxicity detection research; add content warnings |

---

## 9. Testing Strategy

### 9.1 Unit Tests
- **Data Layer**: Test dataset loading, tokenization, class balancing, adversarial perturbations
- **Model Layer**: Test forward pass shapes, save/load consistency
- **RL Layer**: Test reward computation, PPO loss computation
- **Metrics**: Test metric computation against known values

### 9.2 Integration Tests
- End-to-end training pipeline (small dataset, 1 epoch)
- End-to-end evaluation pipeline
- RL training + evaluation pipeline

### 9.3 Performance Tests
- Training throughput (samples/sec)
- Inference latency (ms/sample)
- Memory footprint (peak GPU memory)

---

## 10. Monitoring & Observability (Future)

| Component | Tool | Metrics |
|-----------|------|---------|
| Training | TensorBoard | Loss, accuracy, learning rate |
| Inference | Custom logging | Latency, throughput, error rate |
| Model Drift | Periodic evaluation | Accuracy on held-out set over time |
| System | psutil / GPUtil | CPU, GPU, memory utilization |
