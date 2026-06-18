# 🏆 World Cup 2026 Predictor — CLI Commands Reference

---

## 📋 Quick Reference

```
Main.py [command] [options]
```

| Command | Description |
|---------|-------------|
| `fetch` | Download international football results dataset |
| `train` | Train the Deep Learning model |
| `predict` | Predict a single match outcome |
| `cup` | 🏆 Predict WC 2026 winner via tournament simulation |
| `simulate` | Alias for `cup` |
| `list-teams` | List all teams known to the model |
| `info` | Show model architecture and statistics |

---

## 1️⃣ fetch — Download Dataset

Downloads 49,477 international football matches (1872–2026).

```bash
python Main.py fetch
```

**Output:**
```
📥 Fetching dataset...
✅ Dataset saved to data/raw_results.csv
   Shape: (49477, 9)
   Date range: 1872-11-30 to 2026-06-27
   Tournaments: 200
   Teams: 336
```

---

## 2️⃣ train — Train the Model

Trains a Hybrid Embedding + MLP model (20,131 parameters).

```bash
python Main.py train
```

**Output:**
```
🚀 Starting training pipeline...
Using 29365 matches from 1995+
🧠 Model: 20,131 parameters
🚀 Training...
   Epoch  1/30 | Train Loss: 1.0456 | Val Acc: 0.4471
   Epoch  5/30 | Train Loss: 0.9637 | Val Acc: 0.5312
   Epoch 10/30 | Train Loss: 0.8882 | Val Acc: 0.5723
   ...
📊 Test Accuracy: 0.5732 (57.32%)
✅ TRAINING COMPLETE!
```

The model automatically saves the **best weights** (highest validation accuracy) to:
`models/model_artifacts/football_model.pt`

---

## 3️⃣ predict — Match Prediction

Predicts the outcome of a single match with probabilities.

### Usage

```bash
python Main.py predict <home_team> <away_team> [options]
```

### Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `home_team` | ✅ Yes | Name of the home team (use quotes for spaces) |
| `away_team` | ✅ Yes | Name of the away team (use quotes for spaces) |

### Options

| Option | Description |
|--------|-------------|
| `--neutral` | Neutral venue (e.g., World Cup match) |
| `--tournament TEXT` | Tournament name (default: "FIFA World Cup") |

### Examples

```bash
# Home/Away fixture (home advantage applied)
python Main.py predict "Morocco" "France"

# Neutral venue (World Cup / tournament match)
python Main.py predict "France" "Argentina" --neutral

# Different tournament weighting
python Main.py predict "Brazil" "Germany" --tournament "Friendly"

# Custom tournament
python Main.py predict "Senegal" "Netherlands" --tournament "FIFA World Cup" --neutral
```

### Sample Output

```
==================================================
🏟️  Morocco vs France
==================================================
   Tournament: FIFA World Cup
   Venue: Home/Away

   📊 Probabilities:
      Morocco              Win:  19.8%
      Draw                          24.3%
      France               Win:  55.9%

   🎯 Prediction: France Win (confidence: 55.9%)
==================================================
```

### Explanation

| Probability | Meaning |
|-------------|---------|
| Home Win | Home team wins (class 0) |
| Away Win | Away team wins (class 1) |
| Draw | Match ends in a draw (class 2) |

The model outputs **softmax probabilities** that always sum to 100%.

---

## 4️⃣ cup — Tournament Simulation 🏆

Simulates the entire FIFA World Cup 2026 using Monte Carlo methods.

### Format: 48 Teams

```
16 Groups of 3 → Top 2 Advance → R32 → R16 → QF → SF → 🏆 Final
```

### Usage

```bash
python Main.py cup [options]
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--sims NUMBER` | 500 | Number of Monte Carlo simulations to run |

### Examples

```bash
# Quick test (30 seconds)
python Main.py cup --sims 10

# Moderate accuracy (2-3 minutes)
python Main.py cup --sims 200

# High accuracy (5-10 minutes)
python Main.py cup --sims 1000
```

### Sample Output

```
======================================================================
🏆 WORLD CUP 2026 — SIMULATION RESULTS
======================================================================
   500 Monte Carlo simulations | 48 teams
======================================================================

🏆 WINNER LEADERBOARD
──────────────────────────────────────────────────────────────
Rank  Team                       Win %     Wins   R16 %
──────────────────────────────────────────────────────────────
1     Argentina                   8.4%     42/500  97%
2     Brazil                      7.2%     36/500  95%
3     France                      6.6%     33/500  94%
4     Germany                     5.9%     29/500  91%
5     Spain                       5.4%     27/500  92%
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

### How It Works

1. **Group Draw**: 48 teams randomly split into 16 groups of 3
2. **Group Stage**: Each team plays 2 matches (round-robin)
3. **Advancement**: Top 2 from each group (32 teams) → R32
4. **Knockout**: Single-elimination bracket
5. **Penalties**: Draws in KO → penalty shootout (50/50)
6. **Repeat**: N times → aggregate winner probabilities

---

## 5️⃣ simulate — Alias for cup

Identical to the `cup` command:

```bash
python Main.py simulate --sims 500
```

---

## 6️⃣ list-teams — List All Teams

Shows all 323 teams the model knows:

```bash
python Main.py list-teams
```

**Output:**
```
📋 Teams in model (323):
    1. Afghanistan
    2. Albania
    3. Algeria
    4. Andorra
    5. Angola
   ...
```

---

## 7️⃣ info — Model Information

Displays the model architecture, parameters, and features:

```bash
python Main.py info
```

**Output:**
```
📊 Model Information:
   Teams: 323
   Numerical features: 5
   Feature names: ['home_elo', 'away_elo', 'elo_diff', 'tournament_weight', 'neutral_int']
   Device: cpu
   Total parameters: 20,131
```

---

## 🔧 Getting Started — Full Pipeline

```bash
# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Download dataset
python Main.py fetch

# Step 3: Train model
python Main.py train

# Step 4: Predict a match
python Main.py predict "France" "Argentina" --neutral

# Step 5: Simulate World Cup
python Main.py cup --sims 500
```

---

## 📊 Model Details

| Property | Value |
|----------|-------|
| Architecture | Embedding(323, 32) + MLP(5→128→64) + Linear(128→3) |
| Parameters | 20,131 |
| Training data | 29,365 matches (1995–2021) |
| Features | home_elo, away_elo, elo_diff, tournament_weight, neutral_int |
| Output | Home Win / Away Win / Draw (softmax) |
| Loss | Cross-Entropy |
| Optimizer | Adam (lr=0.001) |
| Best weights | Auto-saved to `models/model_artifacts/football_model.pt` |

---

## ⚠️ Notes

- Team names are **case-sensitive** and must match exactly as in the dataset
- Use **quotes** for multi-word team names: `"Saudi Arabia"` not `Saudi Arabia`
- The `--sims` flag for `cup` can be from 1 (fast, inaccurate) to 10,000+ (slow, stable)
- 500 simulations gives a good balance of speed vs accuracy (~2-3 min on CPU)
- The model runs on **CPU** by default (CUDA auto-detected if available)