# Web / App Flow Document
# Safety Alignment Classifier — User Flows & Interaction Design

**Version:** 1.0  
**Date:** August 27, 2026  
**Author:** Lavish Bansal  

---

## 1. Application Overview

The Safety Alignment Classifier has **two interfaces**:
1. **CLI Pipeline** — For training, evaluation, and experiment management (developer-facing)
2. **Web Dashboard** — Streamlit-based interactive demo (user-facing, portfolio/resume showcase)

---

## 2. CLI Pipeline Flows

### 2.1 Complete Pipeline Flow

```mermaid
flowchart TD
    A["🚀 START"] --> B["Setup Environment"]
    B --> B1["pip install -r requirements.txt"]
    B1 --> B2["python -m spacy download en_core_web_md"]
    B2 --> B3["Configure Kaggle API"]
    B3 --> C["Download Dataset"]
    
    C --> D{"Choose Workflow"}
    
    D -->|"Train"| E["Baseline Training"]
    D -->|"Evaluate"| F["Evaluation"]
    D -->|"RL Train"| G["RL Policy Training"]
    D -->|"Inference"| H["Single Inference"]
    D -->|"Full Experiment"| I["Run All"]
    
    E --> E1["python scripts/train.py --config configs/default.yaml"]
    E1 --> E2["Load & Balance Dataset"]
    E2 --> E3["Fine-tune RoBERTa"]
    E3 --> E4["Save Checkpoint → models/classifier.pt"]
    E4 --> E5["Log to TensorBoard"]
    
    F --> F1["python scripts/evaluate.py --model models/classifier.pt"]
    F1 --> F2{"Adversarial?"}
    F2 -->|"Yes"| F3["Generate Adversarial Examples"]
    F3 --> F4["Evaluate Both Clean & Adversarial"]
    F2 -->|"No"| F5["Evaluate Clean Only"]
    F4 --> F6["Output Metrics + Confusion Matrix"]
    F5 --> F6
    
    G --> G1["python scripts/train_rl.py --classifier models/classifier.pt"]
    G1 --> G2["Load Frozen Classifier"]
    G2 --> G3["Generate Adversarial Training Data"]
    G3 --> G4["Train PPO Policy Network"]
    G4 --> G5["Save Policy → models/policy.pt"]
    G5 --> G6["Evaluate RL-Enhanced Model"]
    
    H --> H1["python scripts/inference.py --text 'Your text here'"]
    H1 --> H2["Tokenize Input"]
    H2 --> H3["Classifier Prediction"]
    H3 --> H4["Policy Adjustment"]
    H4 --> H5["Output: Label + Confidence"]
    
    I --> E
    E4 --> F
    F6 --> G
    G6 --> J["Generate Report"]
    J --> K["📊 Results in results/ folder"]
```

### 2.2 Training Flow (Detailed)

```mermaid
sequenceDiagram
    participant User
    participant CLI as train.py
    participant Data as Dataset
    participant Model as RoBERTa Classifier
    participant Disk as File System
    participant TB as TensorBoard
    
    User->>CLI: python scripts/train.py --config default.yaml
    CLI->>Data: load_toxicity_dataset(directory)
    Data->>Data: Download/Load Jigsaw dataset
    Data->>Data: Binarize labels (toxic >= 0.5)
    Data->>Data: Class balance (50/50)
    Data-->>CLI: train_dataset, test_dataset
    
    CLI->>Model: Initialize RoBERTa-base + Linear(768→2)
    CLI->>CLI: Setup optimizer (AdamW, lr=2e-5)
    CLI->>CLI: Setup loss (CrossEntropyLoss)
    
    loop Each Epoch (1 to N)
        loop Each Batch
            CLI->>Model: Forward pass (input_ids, attention_mask)
            Model-->>CLI: logits
            CLI->>CLI: Compute loss
            CLI->>Model: Backward pass + optimizer step
            CLI->>TB: Log batch loss, accuracy
        end
        CLI->>TB: Log epoch metrics
        CLI->>Disk: Save if best checkpoint
    end
    
    CLI->>Disk: Save final model → models/classifier.pt
    CLI-->>User: Training complete! Accuracy: XX%
```

### 2.3 RL Training Flow (Detailed)

```mermaid
sequenceDiagram
    participant User
    participant CLI as train_rl.py
    participant Data as Adversarial Dataset
    participant Classifier as Frozen Classifier
    participant Policy as Policy Network
    participant PPO as PPO Agent
    
    User->>CLI: python scripts/train_rl.py --classifier models/classifier.pt
    CLI->>Classifier: Load pretrained weights (frozen)
    CLI->>Policy: Initialize PolicyNetwork(2→128→2)
    CLI->>Data: Generate mixed clean + adversarial dataset
    
    loop Each Epoch
        loop Each Batch
            CLI->>Classifier: Get logits (frozen, no grad)
            Classifier-->>PPO: states = logits
            PPO->>Policy: Select action (softmax → sample)
            Policy-->>PPO: action, old_probs
            PPO->>PPO: Compute reward (1.5 if adv correct, 1.0 if clean correct, -1 if wrong)
            PPO->>Policy: Get new_probs
            PPO->>PPO: Compute PPO clipped loss
            PPO->>Policy: Update weights
        end
    end
    
    CLI->>CLI: Save policy → models/policy.pt
    CLI-->>User: RL training complete! Adversarial accuracy: XX%
```

---

## 3. Web Dashboard Flows (Streamlit)

### 3.1 Application Navigation

```mermaid
flowchart LR
    A["🏠 Home / Overview"] --> B["📝 Live Demo"]
    A --> C["📊 Results Dashboard"]
    A --> D["🔬 Adversarial Explorer"]
    A --> E["📈 Training Curves"]
    A --> F["ℹ️ About / How It Works"]
```

### 3.2 Page: Home / Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│  🛡️ Safety Alignment Classifier                                     │
│  Adversarially Robust Toxicity Detection with RL                     │
│                                                                      │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐   │
│  │  92.3%     │  │  58.7%     │  │  81.2%     │  │  22.5%     │   │
│  │  Clean Acc │  │  Adv. Acc  │  │  RL Acc    │  │  Recovery  │   │
│  └────────────┘  └────────────┘  └────────────┘  └────────────┘   │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │              Architecture Diagram                             │   │
│  │  [RoBERTa Classifier] → [Adversarial Attacks] → [RL Policy]  │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  Quick Links: [Try Demo] [View Results] [GitHub]                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.3 Page: Live Demo — User Flow

```mermaid
flowchart TD
    A["User enters text in input box"] --> B["Click 'Analyze' button"]
    B --> C["Backend tokenizes with RoBERTa"]
    C --> D["Run through Classifier"]
    D --> E["Display: Clean Prediction + Confidence"]
    
    E --> F{"User toggles 'Apply Adversarial Attack'?"}
    F -->|"Yes"| G["User selects attack type from dropdown"]
    G --> H["Apply selected perturbation"]
    H --> I["Show perturbed text (highlighted changes)"]
    I --> J["Run perturbed text through Classifier"]
    J --> K["Display: Adversarial Prediction + Confidence"]
    
    K --> L{"User toggles 'Apply RL Correction'?"}
    L -->|"Yes"| M["Run through Policy Network"]
    M --> N["Display: RL-Corrected Prediction + Confidence"]
    N --> O["Show side-by-side comparison"]
    
    F -->|"No"| P["Done"]
    L -->|"No"| P
    O --> P
```

### 3.4 Page: Live Demo — Wireframe

```
┌─────────────────────────────────────────────────────────────────────┐
│  📝 Live Toxicity Detection Demo                                     │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  Enter text to analyze:                                       │   │
│  │  ┌──────────────────────────────────────────────────────┐     │   │
│  │  │  This is a terrible and hateful comment about...      │     │   │
│  │  └──────────────────────────────────────────────────────┘     │   │
│  │  [🔍 Analyze]                                                 │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌──────────────────────┐  ┌──────────────────────┐                 │
│  │ ⚔️ Attack Type:       │  │ 🎚️ Perturbation:     │                 │
│  │ [Keyboard Typo    ▾] │  │ [████░░░░░░] 40%     │                 │
│  │ ☑ Apply Attack       │  │ ☑ Apply RL Fix       │                 │
│  └──────────────────────┘  └──────────────────────┘                 │
│                                                                      │
│  ═══════════════════════════════════════════════════════════════     │
│                                                                      │
│  ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐    │
│  │ 🟢 Clean          │ │ 🔴 Adversarial    │ │ 🟢 RL-Corrected  │    │
│  │                  │ │                  │ │                  │    │
│  │ TOXIC            │ │ NOT TOXIC        │ │ TOXIC            │    │
│  │ Confidence: 94%  │ │ Confidence: 62%  │ │ Confidence: 87%  │    │
│  │                  │ │                  │ │                  │    │
│  │ "This is a       │ │ "Thsi si a       │ │ "Thsi si a       │    │
│  │  terrible and    │ │  terrlble adn    │ │  terrlble adn    │    │
│  │  hateful..."     │ │  haetful..."     │ │  haetful..."     │    │
│  └──────────────────┘ └──────────────────┘ └──────────────────┘    │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ 📊 Confidence Comparison Bar Chart                            │   │
│  │  Clean:       ████████████████████████████████████░░ 94%      │   │
│  │  Adversarial: ████████████████████████░░░░░░░░░░░░░░ 62%      │   │
│  │  RL-Fixed:    █████████████████████████████████░░░░░ 87%      │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.5 Page: Results Dashboard — Wireframe

```
┌─────────────────────────────────────────────────────────────────────┐
│  📊 Experiment Results Dashboard                                     │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  Performance Comparison Table                                 │   │
│  │  ┌──────────────┬──────────┬──────────┬──────────┐           │   │
│  │  │ Metric       │ Clean    │ Adv.     │ RL-Fixed │           │   │
│  │  ├──────────────┼──────────┼──────────┼──────────┤           │   │
│  │  │ Accuracy     │ 92.3%    │ 58.7%    │ 81.2%    │           │   │
│  │  │ Precision    │ 91.5%    │ 55.2%    │ 79.8%    │           │   │
│  │  │ Recall       │ 93.1%    │ 62.3%    │ 82.6%    │           │   │
│  │  │ F1-Score     │ 92.3%    │ 58.5%    │ 81.2%    │           │   │
│  │  └──────────────┴──────────┴──────────┴──────────┘           │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌───────────────────────────┐  ┌───────────────────────────┐      │
│  │  Confusion Matrix (Clean) │  │ Confusion Matrix (Adv.)   │      │
│  │   ┌─────┬─────┐          │  │   ┌─────┬─────┐           │      │
│  │   │ 462 │  38 │          │  │   │ 312 │ 188 │           │      │
│  │   ├─────┼─────┤          │  │   ├─────┼─────┤           │      │
│  │   │  35 │ 465 │          │  │   │ 225 │ 275 │           │      │
│  │   └─────┴─────┘          │  │   └─────┴─────┘           │      │
│  └───────────────────────────┘  └───────────────────────────┘      │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  Per-Attack Robustness Breakdown                              │   │
│  │  Keyboard Typo:    ████████████████████░░░░ 78%               │   │
│  │  Char Insertion:   █████████████████████░░░ 82%               │   │
│  │  Synonym Replace:  ██████████████░░░░░░░░░░ 55%               │   │
│  │  Homophone Swap:   ███████████████████░░░░░ 74%               │   │
│  │  Word Shuffle:     ████████████████░░░░░░░░ 65%               │   │
│  │  Back Translation: ██████████████████░░░░░░ 70%               │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.6 Page: Adversarial Explorer

```
┌─────────────────────────────────────────────────────────────────────┐
│  🔬 Adversarial Text Explorer                                       │
│                                                                      │
│  Original Text:                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  "You are such a terrible person and I hate everything       │   │
│  │   you stand for in this world."                               │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  [Generate All Attacks]                                              │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ Attack: Keyboard Typo                                         │   │
│  │ "You are suxh a terribke person and I hatr everything..."     │   │
│  │ Changed: 3 words | Classifier: NOT TOXIC (❌ fooled!)         │   │
│  ├──────────────────────────────────────────────────────────────┤   │
│  │ Attack: Synonym Replacement                                   │   │
│  │ "You are such a awful individual and I detest everything..."  │   │
│  │ Changed: 3 words | Classifier: TOXIC (✅ detected!)           │   │
│  ├──────────────────────────────────────────────────────────────┤   │
│  │ Attack: Homophone Substitution                                │   │
│  │ "You are such a terrible person and I hate everything..."     │   │
│  │ Changed: 0 words | Classifier: TOXIC (✅ detected!)           │   │
│  ├──────────────────────────────────────────────────────────────┤   │
│  │ Attack: Word Order Shuffle                                    │   │
│  │ "Are you such a terrible person and hate I everything..."     │   │
│  │ Changed: reordered | Classifier: NOT TOXIC (❌ fooled!)       │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. State Management

### 4.1 Application State (Streamlit)

```python
# Session state structure
st.session_state = {
    # Models (loaded once)
    "classifier": ToxicityClassifier,      # Loaded model
    "policy": PolicyNetwork,               # Loaded policy
    "tokenizer": RobertaTokenizer,         # Shared tokenizer
    
    # User inputs
    "input_text": str,                     # Current input text
    "attack_type": str,                    # Selected attack
    "perturbation_prob": float,            # Attack strength
    "apply_attack": bool,                  # Toggle
    "apply_rl": bool,                      # Toggle
    
    # Results
    "clean_prediction": dict,              # {label, confidence, logits}
    "adversarial_prediction": dict,        # {label, confidence, logits, perturbed_text}
    "rl_prediction": dict,                 # {label, confidence, adjusted_logits}
    
    # Experiment results
    "results_df": pd.DataFrame,            # Pre-computed results table
    "confusion_matrices": dict,            # Pre-computed confusion matrices
}
```

### 4.2 Error Handling Flow

```mermaid
flowchart TD
    A["User Action"] --> B{"Valid Input?"}
    B -->|"Empty text"| C["Show warning: 'Please enter text'"]
    B -->|"Too long"| D["Truncate to 256 tokens, show info"]
    B -->|"Valid"| E{"Model Loaded?"}
    E -->|"No"| F["Show error: 'Model not found. Run training first.'"]
    E -->|"Yes"| G["Process normally"]
    G --> H{"Inference Error?"}
    H -->|"Yes"| I["Show error with traceback option"]
    H -->|"No"| J["Display results"]
```

---

## 5. API Endpoints (Future — FastAPI)

### 5.1 Endpoint Design

| Method | Endpoint | Description | Request Body | Response |
|--------|----------|-------------|-------------|----------|
| POST | `/predict` | Single text classification | `{"text": str}` | `{"label": str, "confidence": float, "logits": [float]}` |
| POST | `/predict/batch` | Batch classification | `{"texts": [str]}` | `[{"label": str, "confidence": float}]` |
| POST | `/predict/adversarial` | Predict with adversarial analysis | `{"text": str, "attack": str}` | `{"clean": {...}, "adversarial": {...}, "rl_corrected": {...}}` |
| GET | `/health` | Health check | — | `{"status": "ok", "model_loaded": bool}` |
| GET | `/attacks` | List available attacks | — | `{"attacks": [str]}` |

### 5.2 API Flow

```mermaid
sequenceDiagram
    participant Client
    participant API as FastAPI
    participant Model as Classifier + Policy
    
    Client->>API: POST /predict {"text": "..."}
    API->>API: Validate input
    API->>Model: Tokenize + Forward pass
    Model-->>API: logits
    API->>API: Compute label + confidence
    API-->>Client: {"label": "toxic", "confidence": 0.94}
```

---

## 6. Navigation & Routing

### 6.1 Streamlit Sidebar Navigation

```
┌───────────────────┐
│ 🛡️ SAC Dashboard   │
│                   │
│ ─────────────────│
│                   │
│ 🏠 Overview       │
│ 📝 Live Demo      │
│ 📊 Results        │
│ 🔬 Adv. Explorer  │
│ 📈 Training Logs  │
│ ℹ️ About          │
│                   │
│ ─────────────────│
│                   │
│ Model Status:     │
│ ✅ Classifier     │
│ ✅ Policy Net     │
│                   │
│ Device: GPU       │
│ Latency: ~50ms    │
└───────────────────┘
```

---

## 7. Deployment Options

### 7.1 Local Development
```bash
streamlit run app/streamlit_app.py --server.port 8501
```

### 7.2 Cloud Deployment Options

| Platform | Free Tier | GPU Support | Notes |
|----------|-----------|-------------|-------|
| **Streamlit Cloud** | ✅ (1 app) | ❌ | Best for portfolio demo; CPU only |
| **HuggingFace Spaces** | ✅ | ✅ (paid) | Gradio integration native |
| **Render** | ✅ | ❌ | Docker support |
| **Google Colab** | ✅ | ✅ | Great for notebooks |
| **AWS EC2** | Free tier | ✅ (paid) | Full control |

### 7.3 Recommended: Streamlit Cloud + HuggingFace Spaces
- Deploy **Streamlit app** on Streamlit Cloud for the interactive demo
- Upload **model weights** to HuggingFace Hub for easy loading
- Add a "Live Demo" badge to README and resume
