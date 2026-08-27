# Design / UI-UX Document
# Safety Alignment Classifier — Visual Design Specification

**Version:** 1.0  
**Date:** August 27, 2026  
**Author:** Lavish Bansal  

---

## 1. Design Philosophy

### 1.1 Core Principles
| Principle | Application |
|-----------|-------------|
| **Clarity** | Complex ML concepts presented with simple, intuitive visuals |
| **Trust** | Security/safety theme reinforced through shield iconography and green/red status indicators |
| **Comparison** | Side-by-side layouts make clean vs adversarial vs RL-fixed immediately obvious |
| **Interactivity** | Every core concept is explorable — users can input their own text and see real-time results |
| **Professional** | Portfolio-grade polish that impresses hiring managers and technical reviewers |

### 1.2 Design Inspirations
- **HuggingFace Model Cards** — Clean model performance display
- **OpenAI Playground** — Real-time text interaction pattern
- **Weights & Biases Dashboards** — Experiment tracking aesthetics
- **TensorFlow Playground** — Interactive ML exploration

---

## 2. Brand Identity

### 2.1 Color Palette

#### Primary Theme: Dark Mode (Default)

| Color | Hex | Usage |
|-------|-----|-------|
| **Background (Primary)** | `#0E1117` | Main app background |
| **Background (Card)** | `#1A1D23` | Card/container backgrounds |
| **Background (Elevated)** | `#262730` | Hover states, modal backgrounds |
| **Text (Primary)** | `#FAFAFA` | Headings, primary content |
| **Text (Secondary)** | `#B0B8C1` | Descriptions, labels |
| **Accent (Safe/Green)** | `#00D26A` | Correct predictions, safe status |
| **Accent (Danger/Red)** | `#FF4B4B` | Toxic labels, incorrect, adversarial |
| **Accent (Warning/Amber)** | `#FFA726` | Warnings, adversarial status |
| **Accent (Info/Blue)** | `#1F77B4` | RL-corrected, informational |
| **Accent (Purple)** | `#9D65C9` | RL/Policy elements |
| **Border** | `#333640` | Subtle borders |
| **Gradient Start** | `#667EEA` | Gradient hero sections |
| **Gradient End** | `#764BA2` | Gradient hero sections |

#### Status Colors

```
✅ Correct / Non-Toxic:  #00D26A (green)
❌ Incorrect / Toxic:     #FF4B4B (red)
⚠️ Adversarial / Warning: #FFA726 (amber)
🔵 RL-Corrected:          #1F77B4 (blue)
🟣 Policy Network:        #9D65C9 (purple)
```

### 2.2 Typography

| Element | Font | Weight | Size |
|---------|------|--------|------|
| **App Title** | Inter | 800 (ExtraBold) | 32px |
| **Page Heading** | Inter | 700 (Bold) | 28px |
| **Section Heading** | Inter | 600 (SemiBold) | 22px |
| **Card Title** | Inter | 600 (SemiBold) | 18px |
| **Body Text** | Inter | 400 (Regular) | 16px |
| **Labels / Captions** | Inter | 500 (Medium) | 14px |
| **Code / Monospace** | JetBrains Mono | 400 | 14px |
| **Metric Numbers** | Inter | 700 (Bold) | 36px |

### 2.3 Iconography

| Icon | Context | Source |
|------|---------|--------|
| 🛡️ | App logo / safety theme | Emoji or custom SVG |
| 📝 | Text input / demo | Emoji |
| 📊 | Results / metrics | Emoji |
| 🔬 | Adversarial explorer | Emoji |
| ⚡ | Performance / speed | Emoji |
| 🎯 | Accuracy target | Emoji |
| 🧠 | ML model / intelligence | Emoji |
| 🔄 | RL correction / recovery | Emoji |

---

## 3. Component Library

### 3.1 Metric Card

```
┌──────────────────────┐
│  📊                   │
│  92.3%               │  ← Large number, bold, color-coded
│  Clean Accuracy      │  ← Label, muted text
│  ▲ +2.1% from last  │  ← Trend indicator (optional)
└──────────────────────┘

Variants:
- Success (green border-left): Metric is at or above target
- Warning (amber border-left): Metric is below target but acceptable
- Danger (red border-left): Metric is critically low
```

**Streamlit Implementation:**
```python
def metric_card(label, value, delta=None, status="success"):
    color_map = {
        "success": "#00D26A",
        "warning": "#FFA726", 
        "danger": "#FF4B4B",
        "info": "#1F77B4"
    }
    color = color_map[status]
    st.markdown(f"""
    <div style="
        background: #1A1D23;
        border-left: 4px solid {color};
        border-radius: 8px;
        padding: 20px;
        margin: 8px 0;
    ">
        <div style="color: #B0B8C1; font-size: 14px;">{label}</div>
        <div style="color: {color}; font-size: 36px; font-weight: 700;">{value}</div>
    </div>
    """, unsafe_allow_html=True)
```

### 3.2 Prediction Result Card

```
┌──────────────────────────────────┐
│  🟢 Clean Prediction              │
│  ──────────────────────────────  │
│                                  │
│  Label:  ██ TOXIC               │  ← Badge with background color
│  Confidence: ████████████░░ 87%  │  ← Progress bar
│                                  │
│  "This is a terrible and         │  ← Text preview
│   hateful comment..."            │
│                                  │
│  Logits: [0.13, 0.87]           │  ← Technical detail (collapsible)
└──────────────────────────────────┘
```

### 3.3 Text Diff Display

```
┌──────────────────────────────────────────────────┐
│  Original → Adversarial (Keyboard Typo)           │
│  ──────────────────────────────────────────────── │
│                                                    │
│  You are such a [terrible → terrlble] person and  │  
│  I [hate → haet] everything you stand for         │
│                                                    │
│  Words changed: 2 / 15 (13.3%)                    │
└──────────────────────────────────────────────────┘

Highlighting:
- Deleted/original word: red text with strikethrough
- Inserted/new word: green text with underline  
- Unchanged words: normal text
```

### 3.4 Comparison Panel (3-column)

```
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│   🟢 CLEAN    │ │  🔴 ATTACK    │ │  🔵 RL-FIX    │
│              │ │              │ │              │
│   TOXIC      │ │  NOT TOXIC   │ │   TOXIC      │
│   94% ████░  │ │  62% ████░░  │ │  87% ████░   │
│              │ │              │ │              │
│  "terrible   │ │ "terrlble    │ │ "terrlble    │
│   and        │ │  adn         │ │  adn         │
│   hateful"   │ │  haetful"    │ │  haetful"    │
└──────────────┘ └──────────────┘ └──────────────┘
     ✅              ❌ Fooled!         ✅ Fixed!
```

### 3.5 Attack Selector

```
┌──────────────────────────────────────────────┐
│  ⚔️ Adversarial Attack Configuration          │
│  ────────────────────────────────────────────│
│                                              │
│  Attack Type:  [Keyboard Typo          ▾]   │
│                                              │
│  Strength:     [████████░░░░░░░░] 40%       │
│                Low          Medium    High    │
│                                              │
│  ☑ Apply to input    ☐ Show all attacks      │
└──────────────────────────────────────────────┘
```

---

## 4. Page Layouts

### 4.1 Home Page

```
┌═══════════════════════════════════════════════════════════════════════┐
│ SIDEBAR          │  MAIN CONTENT                                      │
│                  │                                                     │
│ 🛡️ SAC           │  ┌─────────────────────────────────────────────┐   │
│                  │  │            HERO SECTION                      │   │
│ ─────────        │  │  ┌──────────────────────────────────────┐   │   │
│ 🏠 Overview ←    │  │  │  🛡️ Safety Alignment Classifier       │   │   │
│ 📝 Live Demo     │  │  │                                      │   │   │
│ 📊 Results       │  │  │  Adversarially Robust Toxicity       │   │   │
│ 🔬 Explorer      │  │  │  Detection with Reinforcement        │   │   │
│ 📈 Training      │  │  │  Learning                            │   │   │
│ ℹ️ About         │  │  │                                      │   │   │
│                  │  │  │  Background: gradient                 │   │   │
│ ─────────        │  │  │  #667EEA → #764BA2                   │   │   │
│                  │  │  └──────────────────────────────────────┘   │   │
│ Model Status     │  └─────────────────────────────────────────────┘   │
│ ✅ Classifier    │                                                     │
│ ✅ Policy        │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ │
│                  │  │  92.3%  │ │  58.7%  │ │  81.2%  │ │  38.3%  │ │
│ Device: GPU      │  │ Clean   │ │ Adv.    │ │ RL-Fix  │ │Recovery │ │
│                  │  │ Acc.    │ │ Acc.    │ │ Acc.    │ │ Rate    │ │
│                  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘ │
│                  │                                                     │
│                  │  ┌─────────────────────────────────────────────┐   │
│                  │  │  HOW IT WORKS (3-step visual)               │   │
│                  │  │                                             │   │
│                  │  │  ① Train Classifier                        │   │
│                  │  │  RoBERTa fine-tuned on Jigsaw dataset      │   │
│                  │  │         ↓                                   │   │
│                  │  │  ② Attack with Adversarial Perturbations   │   │
│                  │  │  6 attack types simulate evasion tactics    │   │
│                  │  │         ↓                                   │   │
│                  │  │  ③ Defend with RL Policy Network           │   │
│                  │  │  PPO-trained policy corrects predictions    │   │
│                  │  └─────────────────────────────────────────────┘   │
│                  │                                                     │
│                  │  ┌──────────────────┐  ┌──────────────────┐       │
│                  │  │ [📝 Try Demo →]   │  │ [📊 View Results]│       │
│                  │  └──────────────────┘  └──────────────────┘       │
└═══════════════════════════════════════════════════════════════════════┘
```

### 4.2 Live Demo Page

```
┌═══════════════════════════════════════════════════════════════════════┐
│ SIDEBAR          │  MAIN CONTENT                                      │
│                  │                                                     │
│ (same sidebar)   │  📝 Live Toxicity Detection                        │
│                  │                                                     │
│                  │  ┌─────────────────────────────────────────────┐   │
│                  │  │  Enter text to analyze:                     │   │
│                  │  │  ┌─────────────────────────────────────┐   │   │
│                  │  │  │                                     │   │   │
│                  │  │  │  (text area, 4 lines)               │   │   │
│                  │  │  │                                     │   │   │
│                  │  │  └─────────────────────────────────────┘   │   │
│                  │  │  Or try a sample: [Toxic] [Clean] [Tricky] │   │
│                  │  │                                             │   │
│                  │  │  [🔍 Analyze]                               │   │
│                  │  └─────────────────────────────────────────────┘   │
│                  │                                                     │
│                  │  ┌────────────────┐  ┌────────────────────────┐   │
│                  │  │ Attack Type:   │  │ Strength: [████░] 40%  │   │
│                  │  │ [Keyboard ▾]   │  │ ☑ Attack  ☑ RL-Fix    │   │
│                  │  └────────────────┘  └────────────────────────┘   │
│                  │                                                     │
│                  │  ─────── Results ────────                          │
│                  │                                                     │
│                  │  col1          col2          col3                   │
│                  │  ┌──────────┐ ┌──────────┐ ┌──────────┐           │
│                  │  │ 🟢 Clean  │ │ 🔴 Attack │ │ 🔵 Fixed  │           │
│                  │  │ TOXIC    │ │ NOT TOXIC│ │ TOXIC    │           │
│                  │  │ 94%      │ │ 62%      │ │ 87%      │           │
│                  │  └──────────┘ └──────────┘ └──────────┘           │
│                  │                                                     │
│                  │  ┌─────────────────────────────────────────────┐   │
│                  │  │  Confidence Comparison (grouped bar chart)  │   │
│                  │  └─────────────────────────────────────────────┘   │
│                  │                                                     │
│                  │  ▸ Show technical details (expandable)             │
│                  │    logits, attention weights, token analysis       │
└═══════════════════════════════════════════════════════════════════════┘
```

---

## 5. Visualization Design

### 5.1 Charts & Plots

#### Confusion Matrix Style
- **Library**: matplotlib + seaborn
- **Colormap**: `Blues` for clean, `Reds` for adversarial, `Purples` for RL
- **Annotations**: Show counts + percentages
- **Title**: Bold, 16pt, Inter font
- **Grid**: Off (clean look)

#### Bar Charts (Metrics Comparison)
- **Style**: Grouped horizontal bars
- **Colors**: Green (clean), Red (adversarial), Blue (RL-fixed)
- **Labels**: Right-aligned percentage values
- **Background**: Dark (#1A1D23)
- **Grid**: Subtle horizontal gridlines

#### Training Curves
- **Style**: Line plot with shaded confidence interval
- **Colors**: Gradient from light to dark as epochs progress
- **Axes**: Clear labels, grid enabled
- **Legend**: Top-right, semi-transparent background

#### Per-Attack Robustness Radar Chart
```
          Keyboard Typo
              100%
              │
   Back       │        Char
   Trans.  ───┼────  Insertion
              │
              │
   Word    ───┼────  Synonym
   Shuffle    │      Replace
              │
          Homophone
           Swap

  ── Clean (green)
  ── Adversarial (red)  
  ── RL-Fixed (blue)
```

### 5.2 Animation & Transitions

| Element | Animation | Duration | Trigger |
|---------|-----------|----------|---------|
| Metric cards | Fade-in + slide up | 300ms | Page load |
| Prediction results | Fade-in sequential | 200ms stagger | After analysis |
| Confidence bars | Width animation (0→value) | 500ms ease-out | After prediction |
| Attack toggle | Card flip | 300ms | Toggle click |
| Page transitions | Cross-fade | 200ms | Navigation |

---

## 6. Responsive Design

### 6.1 Breakpoints (Streamlit-specific)

| Layout | Sidebar | Columns | Card Layout |
|--------|---------|---------|-------------|
| Desktop (>1200px) | Visible, 300px | 3-4 columns | Grid |
| Tablet (768-1200px) | Collapsible | 2 columns | Grid |
| Mobile (<768px) | Hidden (toggle) | 1 column | Stack |

### 6.2 Mobile Considerations
- Metric cards stack vertically
- Comparison panel becomes a tabbed interface (Clean | Attack | RL-Fix)
- Text input remains full-width
- Charts resize to container width

---

## 7. Accessibility

| Feature | Implementation |
|---------|---------------|
| **Color Contrast** | All text passes WCAG AA (4.5:1 ratio minimum) |
| **Color Blindness** | Don't rely on color alone — use icons (✅❌⚠️) alongside colors |
| **Screen Reader** | All charts have alt-text descriptions |
| **Keyboard Navigation** | All interactive elements focusable |
| **Font Size** | Minimum 14px body text; 36px for metric numbers |
| **Labels** | All form inputs have visible labels |

---

## 8. Example Sentences (For Demo)

### 8.1 Pre-loaded Samples

| Category | Text | Expected Label |
|----------|------|---------------|
| **Clean** | "Thank you for your thoughtful contribution to this discussion." | Non-toxic |
| **Toxic** | "You are an absolute idiot and everything you say is garbage." | Toxic |
| **Subtle Toxic** | "I'm sure you tried your best, but honestly it's embarrassing." | Toxic |
| **Adversarial (manual)** | "U r such a terribl3 perzon and I h8 u" | Toxic (hard to detect) |
| **Edge Case** | "This movie was so bad it killed my brain cells." | Non-toxic (sarcasm) |

### 8.2 Sample Adversarial Transformations

**Original**: "You are such a terrible person"

| Attack | Result |
|--------|--------|
| Keyboard Typo | "You are suxh a terribke person" |
| Char Insertion | "You are sucwh a terribzle person" |
| Synonym Replace | "You are such a awful individual" |
| Homophone | "You are such a terrible person" (no common homophones) |
| Word Shuffle | "Are you such a terrible person" |
| Back Translation | "You are such a terrible person" (preserved) |

---

## 9. Design Assets Needed

| Asset | Format | Purpose |
|-------|--------|---------|
| App Logo (Shield + AI) | SVG / PNG | Sidebar header, favicon |
| Architecture Diagram | SVG / PNG | Home page "How it Works" |
| Attack Type Icons (6) | SVG | Attack selector cards |
| Results Placeholder | PNG | README hero image |
| Demo GIF | GIF / WebP | README demo animation |
| Social Preview Image | 1200×630 PNG | GitHub repo / LinkedIn sharing |
