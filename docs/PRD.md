# Product Requirements Document (PRD)
# Safety Alignment Classifier — Adversarially Robust Toxicity Detection

**Version:** 1.0  
**Date:** August 27, 2026  
**Author:** Lavish Bansal  
**Status:** Draft  

---

## 1. Executive Summary

The **Safety Alignment Classifier** is a research-grade NLP system that detects toxic text content with resilience against adversarial attacks. It combines a fine-tuned RoBERTa-based classifier with a PPO-based Reinforcement Learning policy network to maintain high accuracy even when inputs are deliberately perturbed (typos, synonym substitutions, word-order shuffling, etc.). This project bridges the gap between standard toxicity classifiers (which fail under adversarial manipulation) and the real-world need for robust content moderation at scale.

---

## 2. Problem Statement

### 2.1 The Core Problem
Standard text toxicity classifiers are **brittle** — they achieve high accuracy on clean test sets but degrade dramatically when adversarial actors intentionally modify toxic text to bypass detection. Techniques like keyboard typos, homophone substitution, synonym replacement, and word-order shuffling can cause accuracy drops of **20–40%** on leading models.

### 2.2 Why It Matters
- **Content Moderation at Scale**: Platforms like social media, forums, and chat applications need classifiers that can't be easily tricked.
- **AI Safety & Alignment**: As LLMs become more powerful, ensuring they don't generate or pass through toxic content is critical. This project directly addresses the "safety alignment" dimension.
- **Real-World Adversarial Behavior**: Users actively try to circumvent content filters using leet-speak, typos, homophones, and other perturbation strategies.

### 2.3 Current Landscape
| Approach | Strengths | Weaknesses |
|----------|-----------|------------|
| Rule-based filters | Fast, interpretable | Easily bypassed, poor recall |
| Standard fine-tuned BERT/RoBERTa | Good clean accuracy | Fragile to adversarial attacks |
| Adversarial training (augmentation only) | Better robustness | Still limited; no adaptive correction |
| **Our approach (RL + adversarial)** | **Adaptive robustness via policy correction** | **Novel, more complex** |

---

## 3. Target Users / Personas

### 3.1 Primary Persona — ML Engineer / Researcher
- **Goal**: Build or evaluate robust toxicity classifiers for production or research.
- **Pain Point**: Existing models fail under adversarial perturbation; need a systematic approach combining adversarial training with RL-based correction.
- **How They Use It**: Train the classifier, generate adversarial examples, train the RL policy, evaluate comparative robustness.

### 3.2 Secondary Persona — Platform/Trust & Safety Engineer
- **Goal**: Deploy a content moderation classifier that is resistant to user-driven evasion tactics.
- **Pain Point**: Toxic users constantly evolve their evasion strategies; the classifier needs to be adaptable.
- **How They Use It**: Deploy the trained model + policy network as an inference pipeline. Periodically retrain with new adversarial strategies.

### 3.3 Tertiary Persona — Recruiter / Hiring Manager (Resume Audience)
- **Goal**: Evaluate technical depth and breadth of the candidate's ML engineering skills.
- **What Impresses**: End-to-end pipeline, adversarial robustness, RL integration, clear metrics, interactive demo, clean code.

---

## 4. Product Goals & Success Metrics

### 4.1 Goals

| # | Goal | Priority |
|---|------|----------|
| G1 | Train a baseline toxicity classifier with ≥ 90% accuracy on clean data | P0 |
| G2 | Implement 6+ adversarial perturbation techniques | P0 |
| G3 | Train a PPO-based RL policy that recovers ≥ 80% of accuracy lost to adversarial attacks | P0 |
| G4 | Provide comprehensive evaluation with metrics, visualizations, and comparative analysis | P1 |
| G5 | Build an interactive web demo for real-time toxicity detection with adversarial robustness | P1 |
| G6 | Package as a clean, well-documented, resume-ready project | P1 |
| G7 | Publish a short technical report / blog post summarizing findings | P2 |

### 4.2 Key Performance Indicators (KPIs)

| Metric | Baseline Target | With RL Target |
|--------|----------------|----------------|
| Clean Accuracy | ≥ 92% | ≥ 90% (no degradation) |
| Adversarial Accuracy | ~55-65% (expected drop) | ≥ 78% (recovered) |
| F1 Score (clean) | ≥ 0.91 | ≥ 0.89 |
| F1 Score (adversarial) | ~0.50-0.60 | ≥ 0.75 |
| Precision (adversarial) | ≥ 0.70 | ≥ 0.75 |
| Recall (adversarial) | ≥ 0.65 | ≥ 0.72 |

---

## 5. Feature Requirements

### 5.1 Core Features (MVP — P0)

#### F1: Baseline Toxicity Classifier
- Fine-tuned RoBERTa-base on Jigsaw Toxic Comment Classification dataset
- Binary classification: toxic (1) vs non-toxic (0)
- Configurable training hyperparameters via CLI
- Model checkpoint saving/loading

#### F2: Adversarial Text Perturbation Engine
- 6 perturbation techniques:
  1. **Keyboard Typos** — adjacent key substitution
  2. **Random Character Insertion** — random letter insertion
  3. **Synonym Replacement** — SpaCy word-vector based replacement
  4. **Homophone Substitution** — common homophone swaps
  5. **Word Order Shuffling** — local word position swaps
  6. **Back Translation** — English → French → English via MarianMT
- Configurable perturbation probability per technique
- Support for partial and full adversarial datasets

#### F3: RL-Based Robustness Policy (PPO)
- Policy Network: 2-layer MLP that adjusts classifier logits
- PPO (Proximal Policy Optimization) training with clipped surrogate objective
- Reward shaping: higher reward for correct adversarial classification (1.5x vs 1.0x)
- Combined inference: classifier logits + policy adjustments → final prediction

#### F4: Evaluation Pipeline
- Clean data evaluation: accuracy, precision, recall, F1
- Adversarial data evaluation: same metrics on perturbed data
- RL-enhanced evaluation: metrics after policy adjustment
- Comparative analysis: clean vs adversarial vs RL-corrected

### 5.2 Enhanced Features (P1)

#### F5: Interactive Web Dashboard (Streamlit/Gradio)
- Real-time text input → toxicity prediction
- Toggle adversarial perturbation on/off
- Show adversarial vs RL-corrected predictions side-by-side
- Visualize confidence scores and perturbation effects
- Model comparison dashboard

#### F6: Experiment Tracking & Visualization
- TensorBoard / Weights & Biases integration
- Training loss curves, accuracy progression
- Per-attack-type robustness breakdown
- Confusion matrices (clean vs adversarial)
- ROC/AUC curves

#### F7: Comprehensive Reporting
- Automated metrics report generation
- Per-perturbation-type accuracy breakdown table
- LaTeX-ready figures and tables for potential paper submission

### 5.3 Nice-to-Have Features (P2)

#### F8: Multi-Model Comparison
- Compare RoBERTa-base vs DistilBERT vs BERT-base
- Speed vs accuracy tradeoff analysis

#### F9: API Endpoint
- FastAPI-based REST API for inference
- Batch prediction endpoint
- Model versioning support

#### F10: Explainability
- Attention visualization (which tokens influence prediction)
- SHAP/LIME integration for model interpretability
- GradCAM-style analysis for transformer attention heads

---

## 6. Dataset

### 6.1 Primary Dataset
- **Name**: Jigsaw Toxic Comment Classification Challenge
- **Source**: Kaggle
- **Size**: ~160K comments (training), ~150K (test)
- **Labels**: Binary (toxic/non-toxic, derived from `toxic` score ≥ 0.5 threshold)
- **Original Labels**: Multi-label (toxic, severe_toxic, obscene, threat, insult, identity_hate)

### 6.2 Data Pipeline
1. Download from Kaggle via API
2. Load with HuggingFace `datasets` library
3. Preprocess: binarize labels, tokenize with RoBERTa tokenizer
4. Balance classes: 50/50 split for training (configurable)
5. Generate adversarial variants on-the-fly during training/evaluation

---

## 7. Non-Functional Requirements

| Requirement | Specification |
|-------------|---------------|
| **Training Hardware** | CUDA GPU recommended (supports CPU fallback) |
| **Training Time** | < 30 min for baseline (5K samples, 3 epochs on GPU) |
| **Inference Latency** | < 100ms per prediction (single sample, GPU) |
| **Model Size** | ~500MB (RoBERTa-base) + ~1MB (Policy Network) |
| **Python Version** | 3.8+ |
| **Reproducibility** | Fixed random seeds, deterministic data splits |
| **Code Quality** | Type hints, docstrings, logging, modular design |

---

## 8. Out of Scope (v1.0)

- Multi-label toxicity classification (only binary for v1.0)
- Multilingual support (English only for v1.0)
- Real-time production deployment with load balancing
- User feedback loop / active learning
- LLM-based adversarial generation (using GPT to generate adversarial paraphrases)
- Fine-tuning on custom datasets (only Jigsaw dataset for v1.0)

---

## 9. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| RL policy doesn't significantly improve adversarial accuracy | Medium | High | Tune reward shaping; try DQN/A2C as alternatives; increase training data |
| Overfitting on small balanced dataset (5K samples) | Medium | Medium | Increase dataset size; add regularization; use cross-validation |
| Adversarial perturbations are too simple to be realistic | Low | Medium | Add more sophisticated attacks (TextFooler, BERT-Attack); compare with attack benchmarks |
| SpaCy synonym replacement is slow | High | Low | Cache embeddings; pre-compute synonyms; use faster alternatives |
| Back translation model downloads are large | Medium | Low | Make back translation optional (already implemented) |

---

## 10. Roadmap

### Phase 1: Foundation (Week 1)
- Fix existing bugs in cloned repo
- Set up project structure, CI/CD, requirements
- Get baseline training running end-to-end

### Phase 2: Core Enhancement (Week 2)
- Refactor codebase (modular, clean, documented)
- Add configuration system (YAML/argparse)
- Implement proper evaluation pipeline with all metrics
- Fix RL training bugs

### Phase 3: Visualization & Demo (Week 3)
- Build Streamlit/Gradio interactive demo
- Add experiment tracking (TensorBoard)
- Generate all results and visualizations

### Phase 4: Polish & Documentation (Week 4)
- Write comprehensive README
- Create technical report
- Optimize code, add tests
- Final resume-ready packaging

---

## 11. Appendix

### 11.1 Reference Papers
1. **Jigsaw Toxic Comment Classification** — Kaggle Competition (2018)
2. **RoBERTa: A Robustly Optimized BERT Pretraining Approach** — Liu et al. (2019)
3. **Proximal Policy Optimization Algorithms** — Schulman et al. (2017)
4. **Adversarial Examples in NLP: A Survey** — Zhang et al. (2020)
5. **Is BERT Really Robust? A Strong Baseline for Natural Language Attack on Text Classification and Entailment** — Jin et al. (2020)

### 11.2 Glossary
| Term | Definition |
|------|-----------|
| PPO | Proximal Policy Optimization — an RL algorithm that constrains policy updates |
| Adversarial Example | An input intentionally modified to cause misclassification |
| Safety Alignment | Ensuring AI systems behave safely and as intended |
| Toxicity | Content that is rude, disrespectful, or likely to cause someone to leave a conversation |
