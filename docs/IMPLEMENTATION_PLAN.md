# Implementation Plan
# Safety Alignment Classifier — From Cloned Repo to Resume-Ready Project

**Version:** 1.0  
**Date:** August 27, 2026  
**Author:** Lavish Bansal  

---

## Overview

This plan transforms the cloned `Safety-Alignment-Classifier` repo (which has **7 critical bugs**, **10 code quality issues**, and **12 missing features**) into a polished, resume-worthy project that demonstrates expertise in **NLP, adversarial robustness, reinforcement learning, and MLOps**.

---

## Current State Analysis

### What Exists (5 files, ~540 lines)
| File | Lines | Status |
|------|-------|--------|
| `src/dataset.py` | 257 | ⚠️ Has 3 bugs, works partially |
| `src/model.py` | 25 | ✅ Works (simple) |
| `src/train.py` | 63 | ✅ Works (basic) |
| `src/evaluate.py` | 61 | ✅ Works (basic) |
| `src/rl_policy.py` | 163 | ❌ Has 4 bugs, won't run |

### Critical Bugs Found

| # | Bug | File | Severity |
|---|-----|------|----------|
| B1 | `adversarial_transform()` — method signature takes `(self, text)` but `__getitem__` calls `(text, transform_prob)` | `dataset.py:167` | 🔴 Crash |
| B2 | `torch.full(len(dataset), ...)` — missing tuple wrapper for size arg | `dataset.py:136` | 🔴 Crash |
| B3 | `compute_reward()` called with extra `model` argument it doesn't accept | `rl_policy.py:73` | 🔴 Crash |
| B4 | Global `device` variable used in `train_rl_classifier()` — undefined | `rl_policy.py:63-66` | 🔴 Crash |
| B5 | `batch["labels"]` — key is `"label"` not `"labels"` | `rl_policy.py:104` | 🔴 KeyError |
| B6 | `get_adversarial_dataloader()` returns `(dataset, dataloader)` tuple but only one value captured | `rl_policy.py:149,155` | 🔴 Crash |
| B7 | `get_dataset()` at `__main__` called with dataset object instead of directory path | `dataset.py:253-254` | 🔴 Crash |

### What's Missing
- ❌ `requirements.txt`
- ❌ Configuration system
- ❌ Logging (uses `print()`)
- ❌ Unit tests
- ❌ Experiment tracking
- ❌ Visualization / plots
- ❌ Web demo
- ❌ Inference script
- ❌ Comprehensive README
- ❌ Per-attack evaluation
- ❌ Confusion matrices
- ❌ Model export

---

## Phase 1: Foundation (Week 1)

> **Goal**: Fix all bugs, establish proper project structure, get everything running end-to-end.

### 1.1 Project Structure Setup

#### [NEW] `requirements.txt`
Create pinned dependency file:
```
torch>=2.0.0
transformers>=4.30.0
datasets>=2.14.0
spacy>=3.5.0
pandas>=2.0.0
scikit-learn>=1.3.0
tqdm>=4.65.0
matplotlib>=3.7.0
seaborn>=0.12.0
kaggle>=1.5.0
streamlit>=1.25.0
tensorboard>=2.13.0
```

#### [NEW] `src/__init__.py`
Make `src/` a proper Python package.

#### [NEW] `configs/default.yaml`
Centralized configuration with all hyperparameters.

#### [NEW] `Makefile`
Common development commands:
```makefile
train:     python scripts/train.py --config configs/default.yaml
evaluate:  python scripts/evaluate.py --model models/classifier.pt
train-rl:  python scripts/train_rl.py --classifier models/classifier.pt
demo:      streamlit run app/streamlit_app.py
test:      pytest tests/ -v
```

### 1.2 Bug Fixes

#### [MODIFY] `src/dataset.py`
- **B1**: Fix `adversarial_transform()` to accept `transform_prob` parameter
- **B2**: Fix `torch.full(len(dataset), ...)` → `torch.full((len(dataset),), ...)`
- **B7**: Fix `__main__` section: `get_dataset()` should receive `directory_path`, not dataset object
- Add type hints and docstrings throughout
- Add `__init__.py` to make proper package

#### [MODIFY] `src/rl_policy.py`
- **B3**: Fix `compute_reward()` call — remove extra `model` argument
- **B4**: Add `device` parameter to `train_rl_classifier()`
- **B5**: Fix `batch["labels"]` → `batch["label"]`
- **B6**: Fix `get_adversarial_dataloader()` — unpack `(dataset, dataloader)` tuple
- Add type hints and docstrings

#### [MODIFY] `src/model.py`
- Add type hints
- Add comprehensive docstrings
- Add `load_model()` utility function

#### [MODIFY] `src/train.py`
- Add proper logging (replace `print()`)
- Add model validation after each epoch
- Add learning rate scheduler option
- Save training history to JSON

#### [MODIFY] `src/evaluate.py`
- Add confusion matrix generation
- Add ROC/AUC computation
- Add results saving to file
- Return metrics dictionary (not just print)

### 1.3 Verification
- [ ] `python src/train.py --data_directory ./jigsaw_toxicity_data --model_path ./models/classifier.pt` → completes without errors
- [ ] `python src/evaluate.py --data_directory ./jigsaw_toxicity_data --model_path ./models/classifier.pt --adversarial` → prints metrics
- [ ] `python src/rl_policy.py --data_directory ./jigsaw_toxicity_data --classifier_model_path ./models/classifier.pt --policy_model_path ./models/policy.pt` → completes without errors

---

## Phase 2: Core Enhancement (Week 2)

> **Goal**: Refactor into modular architecture, add config system, implement comprehensive evaluation.

### 2.1 Code Restructuring

#### [NEW] `src/config.py`
```python
@dataclass
class TrainingConfig:
    epochs: int = 3
    learning_rate: float = 2e-5
    batch_size: int = 32
    # ... (load from YAML)
```

#### [NEW] `src/data/adversarial.py`
Extract adversarial perturbation functions from `dataset.py` into dedicated module:
- All 6 attack functions
- `KEYBOARD_MAP`, `HOMOPHONE_MAP`
- `AdversarialTextDataset` class
- Configurable per-attack probabilities

#### [NEW] `src/data/preprocessing.py`
- Text cleaning utilities
- Label preprocessing
- Data validation

#### [NEW] `src/evaluation/metrics.py`
- Comprehensive metrics computation
- Per-class breakdown
- Confidence calibration analysis

#### [NEW] `src/evaluation/visualization.py`
- Confusion matrix plots (clean, adversarial, RL)
- ROC/AUC curves
- Per-attack robustness radar chart
- Training loss/accuracy curves
- Metrics comparison bar charts

#### [NEW] `src/utils/logger.py`
- Structured logging configuration
- Console + file handlers
- Training progress formatting

#### [NEW] `src/utils/seed.py`
- Global seed setting (torch, numpy, random)
- Deterministic DataLoader workers

#### [NEW] `src/utils/checkpoint.py`
- Comprehensive checkpoint save/load
- Include config, metrics, epoch info
- Model versioning support

### 2.2 Enhanced Training

#### [MODIFY] `src/train.py` → `scripts/train.py`
- YAML config loading
- TensorBoard logging
- Validation after each epoch
- Best model checkpointing by F1 score
- Early stopping
- Learning rate scheduling (cosine)
- Mixed precision training (AMP) support

#### [MODIFY] `src/rl_policy.py` → `scripts/train_rl.py`
- Fix all bugs (Phase 1)
- Add proper reward logging
- Add adversarial accuracy tracking per epoch
- Add policy network complexity tuning

### 2.3 Per-Attack Evaluation

#### [NEW] `scripts/evaluate_attacks.py`
```python
# For each attack type individually:
#   1. Apply only that attack to test set
#   2. Evaluate classifier accuracy
#   3. Evaluate RL-corrected accuracy
#   4. Log per-attack metrics
# Output: comparative table + radar chart
```

### 2.4 Verification
- [ ] Config loading works from YAML
- [ ] TensorBoard logs are generated
- [ ] Confusion matrices are saved as PNG
- [ ] Per-attack evaluation produces breakdown table
- [ ] All unit tests pass

---

## Phase 3: Visualization & Demo (Week 3)

> **Goal**: Build the interactive Streamlit dashboard and generate all results/visualizations.

### 3.1 Streamlit Dashboard

#### [NEW] `app/streamlit_app.py`
Main application with multi-page navigation:
- 🏠 **Overview**: Hero section, metric cards, architecture diagram
- 📝 **Live Demo**: Real-time text → prediction with adversarial toggle
- 📊 **Results Dashboard**: Metrics comparison table, confusion matrices
- 🔬 **Adversarial Explorer**: Apply all 6 attacks to any text, see results
- 📈 **Training Curves**: Embedded TensorBoard or matplotlib plots
- ℹ️ **About**: Project description, methodology, links

#### [NEW] `app/components/`
Reusable Streamlit components:
- `metric_card.py` — Styled metric display
- `prediction_panel.py` — 3-column clean/attack/RL comparison
- `text_diff.py` — Highlighted text diff display
- `attack_selector.py` — Attack type dropdown + strength slider

### 3.2 Results Generation

#### [NEW] `scripts/run_experiments.py`
Automated experiment pipeline:
```python
# 1. Train baseline classifier
# 2. Evaluate on clean data → save metrics
# 3. Evaluate on adversarial data (all attacks) → save metrics
# 4. Train RL policy
# 5. Evaluate RL-enhanced on adversarial data → save metrics
# 6. Generate all plots and tables
# 7. Generate comparative report
```

#### [NEW] `results/figures/`
Generate all visualization assets:
- `confusion_matrix_clean.png`
- `confusion_matrix_adversarial.png`
- `confusion_matrix_rl_enhanced.png`
- `metrics_comparison.png`
- `training_curves.png`
- `per_attack_robustness.png`
- `roc_curves.png`
- `confidence_distribution.png`

### 3.3 Verification
- [ ] `streamlit run app/streamlit_app.py` launches without errors
- [ ] Live demo accepts text and shows predictions
- [ ] All 6 attack types work in the adversarial explorer
- [ ] Results dashboard displays all pre-computed metrics
- [ ] All plots render correctly

---

## Phase 4: Polish & Documentation (Week 4)

> **Goal**: Make everything resume-ready with comprehensive documentation, tests, and deployment.

### 4.1 README Rewrite

#### [MODIFY] `README.md`
Complete rewrite with hero image, results table, architecture diagram, demo GIF, quick start, and citation.

### 4.2 Technical Report

#### [NEW] `docs/TECHNICAL_REPORT.md`
2-3 page technical writeup covering: Introduction, Related Work, Methodology, Experimental Setup, Results & Analysis, Ablation Studies, Conclusions & Future Work.

### 4.3 Unit Tests

#### [NEW] `tests/test_dataset.py`
- Test data loading and preprocessing
- Test class balancing
- Test tokenization output shapes
- Test adversarial perturbation functions

#### [NEW] `tests/test_model.py`
- Test classifier forward pass shapes
- Test policy network forward pass shapes
- Test model save/load roundtrip

#### [NEW] `tests/test_adversarial.py`
- Test each attack function individually
- Test that attacks modify text
- Test that labels are preserved after attack

#### [NEW] `tests/test_rl.py`
- Test reward computation
- Test PPO loss computation
- Test action selection

### 4.4 CI/CD

#### [NEW] `.github/workflows/ci.yml`
GitHub Actions pipeline for automated testing and linting.

### 4.5 Deployment
- Deploy Streamlit app to Streamlit Cloud
- Upload model weights to HuggingFace Hub
- Add "Live Demo" badge to README
- Create social preview image for GitHub repo

### 4.6 Verification
- [ ] All unit tests pass (`pytest tests/ -v`)
- [ ] CI/CD pipeline runs green
- [ ] README renders correctly on GitHub
- [ ] Streamlit Cloud deployment works
- [ ] All docs are complete and cross-linked

---

## Resume Impact Strategy

### What This Project Demonstrates

| Skill | Evidence |
|-------|----------|
| **NLP / Transformers** | Fine-tuned RoBERTa for binary text classification |
| **Adversarial ML** | Implemented 6 adversarial attack types, measured robustness |
| **Reinforcement Learning** | PPO-based policy network for adaptive prediction correction |
| **MLOps** | Config management, experiment tracking, model checkpointing, CI/CD |
| **Data Engineering** | Imbalanced dataset handling, on-the-fly augmentation, DataLoader optimization |
| **Evaluation** | Comprehensive metrics suite, confusion matrices, per-attack breakdown |
| **Full-Stack ML** | Streamlit demo, FastAPI (optional), end-to-end pipeline |
| **Software Engineering** | Modular architecture, type hints, tests, documentation |

### Resume Bullet Points (Draft — update with real numbers)

> **Safety Alignment Classifier — Adversarially Robust Toxicity Detection** &nbsp;&nbsp; [Live Demo](link)
> - Engineered a **robust toxicity classification system** using **RoBERTa** and **PPO-based reinforcement learning** that maintains **XX% F1-score** on adversarially perturbed text, recovering **XX% of accuracy** lost to 6 types of adversarial attacks.
> - Designed an **adversarial perturbation engine** with 6 attack vectors (keyboard typos, synonym replacement, homophone substitution, word-order shuffling, back-translation) and a **PPO policy network** that adaptively corrects classifier outputs at inference time.
> - Built an interactive **Streamlit dashboard** with real-time toxicity detection, adversarial attack visualization, and RL correction comparison; deployed with **TensorBoard** experiment tracking and comprehensive evaluation pipeline.
> - **Tech Stack**: PyTorch, HuggingFace Transformers, SpaCy, scikit-learn, Streamlit, TensorBoard

---

## Timeline Summary

| Phase | Duration | Key Deliverables |
|-------|----------|-----------------|
| **Phase 1: Foundation** | Week 1 | All bugs fixed, project structure, requirements, end-to-end run |
| **Phase 2: Core Enhancement** | Week 2 | Modular codebase, config system, comprehensive evaluation |
| **Phase 3: Visualization & Demo** | Week 3 | Streamlit app, all results/figures, experiment tracking |
| **Phase 4: Polish & Documentation** | Week 4 | README, technical report, tests, CI/CD, deployment |

---

## Open Questions

1. **Compute Environment**: Do you have access to a GPU for training? CPU training will take ~8 hours for the full pipeline.
2. **Deployment Target**: Should we deploy to Streamlit Cloud (free, easy) or HuggingFace Spaces (better for ML projects)?
3. **Priority**: Should we focus on getting Phase 1-2 done quickly (functional project) or spend more time on Phase 3-4 (polished demo)?
4. **Resume**: Once you share your resume, I can tailor the bullet points and align the project focus with your target roles.

---

## Next Steps

Once you approve this plan, I'll start with **Phase 1** — fixing all 7 bugs and establishing the project foundation. Ready to begin?
