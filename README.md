# 🏆 World Cup 2026 – Match Predictor & Tournament Simulator

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)

A **Deep Learning CLI application** that predicts international football match outcomes and simulates the **entire FIFA World Cup 2026** tournament using Monte Carlo methods. Trained on **49,477 historical international matches** (1872–2026).

---

## 📋 Table of Contents

- [Architecture](#-architecture)
- [How It Works](#-how-it-works)
- [Features](#-features)
- [Quick Start](#-quick-start)
- [Usage Guide](#-usage-guide)
- [Model Details](#-model-details)
- [Dataset](#-dataset)
- [Performance](#-performance)
- [Project Structure](#-project-structure)
- [Technical Stack](#-technical-stack)
- [License](#-license)

---

## 🧠 Architecture

### Model: Hybrid Embedding + MLP (Not a GNN)

This model uses a **standard feedforward architecture with learned team embeddings**, not a Graph Neural Network. Match prediction is fundamentally a *pairwise classification problem* — two teams with features → 3 outcomes. GNNs are designed for graph-structured data (e.g., player passing networks), which isn't needed here.

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT LAYER                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Team A ────────────→ nn.Embedding(323, 32) ────────┐          │
│                                                      │          │
│  Team B ────────────→ nn.Embedding(323, 32) ────────┤          │
│                                                      │          │
│  Numerical Feats ───→ MLP(5 → 128 → 64) ────────────┤          │
│  • home_elo (0–2500)                                │          │
│  • away_elo (0–2500)                                │          │
│  • elo_diff                                         │          │
│  • tournament_weight (0.5–4.0)                      │          │
│  • neutral_int (0 or 1)                             │          │
│                                                      │          │
│                          ┌───────────────────────────┘          │
│                          ▼                                      │
│              torch.cat([h, a, n]) → Tensor(128)                 │
│                          │                                      │
│                          ▼                                      │
│              ┌─────────────────────┐                            │
│              │  Combined Network   │                            │
│              │  Linear(128 → 128)  │                            │
│              │  BatchNorm1d + ReLU │                            │
│              │  Dropout(0.3)       │                            │
│              │  Linear(128 → 64)   │                            │
│              │  BatchNorm1d + ReLU │                            │
│              │  Dropout(0.3)       │                            │
│              │  Linear(64 → 3)     │                            │
│              └──────────┬──────────┘                            │
│                         ▼                                       │
│              Softmax → [HomeWin, AwayWin, Draw]                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Total parameters:** 20,131  
**Loss function:** Cross-Entropy Loss (`nn.CrossEntropyLoss`)  
**Optimizer:** Adam (`lr=0.001`)  
**Regularization:** Dropout (0.3), Batch Normalization, Early Stopping

---

## ⚙️ How It Works

### 1️⃣ Training Pipeline

```
Raw Match Data ──→ Elo Ratings ──→ Feature Engineering ──→ Normalization ──→ PyTorch Model ──→ Predictions
  49,477 matches    Chronological    5 numerical feats    StandardScaler    Embedding + MLP     3 classes
  1872–2026         per team         + team embeddings                     20,131 params
```

### 2️⃣ Match Prediction Flow

```
Input: "France" vs "Argentina" (World Cup Final, Neutral Venue)
   │
   ├── France    ──→ Embedding lookup ──→ [0.12, -0.34, ..., 0.87]  (32-dim)
   ├── Argentina ──→ Embedding lookup ──→ [0.45, 0.12, ..., -0.23]  (32-dim)
   └── Features  ──→ [1500, 1500, 0, 4.0, 1]                       (5-dim)
                            │
                            ▼
                   Concatenate → [32 + 32 + 5] → 69-dim
                            │
                            ▼
                   MLP → Softmax → [0.477, 0.207, 0.316]
                            │
                            ▼
              Prediction: France Win (47.7% confidence)
```

### 3️⃣ Tournament Simulation Flow

```
48 Teams
    │
    ▼
Random Group Draw → 16 Groups of 3
    │
    ▼
Group Stage: Each team plays 2 matches
    │
    ▼
Top 2 from each group advance → 32 teams
    │
    ▼
Knockout Stages:
    R32 ──→ R16 ──→ QF ──→ SF ──→ 🏆 FINAL
    │      │       │      │       │
    └── Single-elimination, draws → penalties (50/50)
    │
    ▼
Repeat N times (Monte Carlo) → Aggregate win probabilities
```

---

## ✨ Features

### Core Features
- **Single Match Prediction**: Predict any international match with probabilities
- **Full Tournament Simulation**: Monte Carlo simulation of WC 2026 with 48 teams
- **Advancement Tracking**: See how often each team reaches R16, QF, SF, and Final
- **Elo Rating System**: Built-in Elo computation for accurate team strength
- **Tournament Weighting**: Matches weighted by importance (Friendly → World Cup)

### CLI Commands
| Command | Description |
|---------|-------------|
| `fetch` | Download international football results dataset |
| `train` | Train the Deep Learning model |
| `predict <home> <away>` | Predict a single match outcome |
| `cup` | 🏆 Simulate WC 2026 and predict the winner |
| `simulate` | Alias for `cup` |
| `list-teams` | List all 323 teams in the model |
| `info` | Display model architecture and statistics |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- pip

### Installation

```bash
# Clone or navigate to the project directory
cd WCprediction

# Install dependencies
pip install -r requirements.txt
```

### Run the Full Pipeline

```bash
# Step 1: Download the dataset (49,477 matches)
python Main.py fetch

# Step 2: Train the model (~30 seconds on CPU)
python Main.py train

# Step 3: Predict a match
python Main.py predict "France" "Argentina" --neutral

# Step 4: Simulate WC 2026 (500 simulations, ~2 minutes)
python Main.py cup --sims 500
```

---

## 📖 Usage Guide

### 🎯 Single Match Prediction

Predict the outcome of any international match:

```bash
# Home/Away fixture
python Main.py predict "Brazil" "Germany"

# Neutral venue (e.g., World Cup match)
python Main.py predict "France" "Argentina" --neutral

# Custom tournament
python Main.py predict "Morocco" "Portugal" --tournament "FIFA World Cup"
```

**Output:**
```
==================================================
🏟️  France vs Argentina
==================================================
   Tournament: FIFA World Cup
   Venue: Neutral

   📊 Probabilities:
      France               Win:  47.7%
      Draw                          31.6%
      Argentina            Win:  20.7%

   🎯 Prediction: Home Win (confidence: 47.7%)
==================================================
```

### 🏆 Tournament Simulation

Predict the World Cup 2026 winner:

```bash
# Quick simulation (100 runs)
python Main.py cup --sims 100

# Full simulation (1000 runs for stable results)
python Main.py cup --sims 1000
```

**Output:**
```
======================================================================
🏆 WORLD CUP 2026 — SIMULATION RESULTS
======================================================================
   1000 Monte Carlo simulations | 48 teams
======================================================================

🏆 WINNER LEADERBOARD
──────────────────────────────────────────────────────────────
Rank  Team                       Win %     Wins   R16 %
──────────────────────────────────────────────────────────────
1     Argentina                   8.4%     84/1000  97%
2     Brazil                      7.2%     72/1000  95%
3     France                      6.6%     66/1000  94%
4     Germany                     5.9%     59/1000  91%
5     Spain                       5.4%     54/1000  92%
...

📊 ADVANCEMENT ODDS (Top 10)
──────────────────────────────────────────────────────────────
Team                       R16     QF      SF      🏆 Win
──────────────────────────────────────────────────────────────
Argentina                  97%     65%     35%     8.40%
Brazil                     95%     62%     33%     7.20%
France                     94%     60%     31%     6.60%
...

======================================================================
🔮 PREDICTED WORLD CUP 2026 WINNER: Argentina
   Win probability: 8.40% (1 in ~12)
======================================================================
```

### 🛠️ Other Commands

```bash
# View model information
python Main.py info

# List all teams the model knows
python Main.py list-teams

# Refresh the dataset
python Main.py fetch
```

---

## 📊 Model Details

### Input Features

| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| `home_elo` | Numerical | 0–2500 | Home team's Elo rating before the match |
| `away_elo` | Numerical | 0–2500 | Away team's Elo rating before the match |
| `elo_diff` | Numerical | -2500–2500 | `home_elo - away_elo` |
| `tournament_weight` | Numerical | 0.5–4.0 | Match importance (0.5=Friendly, 4.0=World Cup) |
| `neutral_int` | Binary | 0 or 1 | 0=Home/Away fixture, 1=Neutral venue |
| Team Embeddings | Learned | 32-dim | Each team has a learned 32-dimensional vector |

### Output Specification

| Class | Label | Description |
|-------|-------|-------------|
| 0 | Home Win | Home team wins |
| 1 | Away Win | Away team wins |
| 2 | Draw | Match ends in a draw |

### Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Training samples | 29,365 (1995–2021) |
| Validation samples | ~4,400 |
| Test samples | ~4,400 (2022–2026) |
| Embedding dimension | 32 |
| Hidden dimension | 128 |
| Dropout rate | 0.3 |
| Learning rate | 0.001 |
| Optimizer | Adam |
| Batch size | 128 |
| Max epochs | 30 |
| Early stopping patience | 10 |
| Loss function | Cross-Entropy |
| Weight decay | 0 |

### Elo Rating System

The model uses a custom Elo rating system for international football:
- **Initial rating:** 1500
- **K-factor:** 30
- **Home advantage bonus:** 100 points
- **Goal margin multiplier:** Adjusts K-factor based on win margin (1–5+ goals)

---

## 📁 Dataset

**Source:** [International Football Results](https://www.kaggle.com/datasets/martj42/international-football-results) (1872–present)

**Statistics:**
| Metric | Value |
|--------|-------|
| Total matches | 49,477 |
| Date range | 1872-11-30 to 2026-06-27 |
| Teams | 336 |
| Tournaments | 200 |
| Training period | 1995–2021 (29,365 matches) |
| Test period | 2022–2026 (~4,400 matches) |
| Columns | date, home_team, away_team, home_score, away_score, tournament, city, country, neutral |

---

## 📈 Performance

- **Training accuracy**: ~58-62% on validation set
- **Test accuracy**: ~55-58% on held-out matches (2022–2026)
- **Baseline**: Always predicting "Home Win" achieves ~44-48%
- **Improvement**: ~10-12% over naive baseline

The model's strength comes from:
1. **Team embeddings** that learn latent team qualities from historical data
2. **Elo ratings** providing a dynamic strength signal
3. **Chronological training** ensuring no data leakage
4. **Tournament weighting** that prioritizes important matches

---

## 📂 Project Structure

```
WCprediction/
│
├── Main.py                          # 🏆 Entry point
├── cli.py                           # Command-line interface
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
│
├── data/
│   ├── fetch_data.py                # Dataset downloader
│   ├── raw_results.csv              # 49,477 matches (downloaded)
│   └── processed/                   # Preprocessed data cache
│
├── features/
│   ├── elo.py                       # Elo rating system
│   └── engineer.py                  # Feature engineering pipeline
│
├── models/
│   ├── train_fast.py                # Optimized training pipeline
│   ├── train.py                     # Full training pipeline (slower)
│   ├── predict.py                   # Prediction & simulation engine
│   └── model_artifacts/             # Saved model weights & encoders
│       ├── football_model.pt        # PyTorch model weights
│       ├── model_config.json        # Architecture configuration
│       ├── team_encoder.pkl         # Team name → ID mapping
│       ├── scaler.pkl               # Feature normalizer
│       └── numerical_cols.pkl       # Feature column names
│
├── test_evaluate.py                 # Comprehensive evaluation
├── quick_eval.py                    # Lightweight evaluation
└── test_cup.py                      # Simulation test
```

---

## 🛠️ Technical Stack

| Component | Technology |
|-----------|-----------|
| **Language** | Python 3.10+ |
| **Deep Learning** | PyTorch 2.0+ |
| **Data Processing** | Pandas, NumPy |
| **Machine Learning** | scikit-learn |
| **CLI Framework** | argparse |
| **Data Fetching** | requests |

### Dependencies
```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.2.0
torch>=2.0.0
requests>=2.28.0
tqdm>=4.64.0
```

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 🙏 Acknowledgments

- [International Football Results Dataset](https://www.kaggle.com/datasets/martj42/international-football-results) by Mart Jürisoo
- FIFA World Cup 2026 format information from FIFA.com
- Elo rating system adapted from the World Football Elo Ratings

---

<p align="center">
  <b>🏆 May the best team win 🏆</b><br>
  <i>World Cup 2026 · USA · Canada · Mexico</i>
</p>