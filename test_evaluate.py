"""
Comprehensive evaluation of the World Cup Predictor model.
Shows architecture, inputs, outputs, loss, and test metrics.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
import sys
import json
import pickle

sys.path.insert(0, str(Path(__file__).parent))
from models.predict import load_model_artifacts, FootballPredictor
from data.fetch_data import load_local
from features.elo import compute_elo_ratings
from features.engineer import get_tournament_weight
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, log_loss
from torch.utils.data import DataLoader, Dataset

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SimpleDataset(Dataset):
    def __init__(self, X_num, home_ids, away_ids, y):
        self.X_num = torch.FloatTensor(X_num)
        self.home_ids = torch.LongTensor(home_ids)
        self.away_ids = torch.LongTensor(away_ids)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (self.X_num[idx], self.home_ids[idx], self.away_ids[idx], self.y[idx])

print("=" * 70)
print("🏆 WORLD CUP 2026 PREDICTOR - COMPREHENSIVE EVALUATION")
print("=" * 70)

# Step 1: Load model
print("\n📂 Loading model artifacts...")
model, team_encoder, scaler, numerical_cols = load_model_artifacts()
model.eval()

# Step 2: Model architecture
print("\n" + "=" * 70)
print("1️⃣  MODEL ARCHITECTURE")
print("=" * 70)
print(model)
total_params = sum(p.numel() for p in model.parameters())
print(f"\n   Total parameters: {total_params:,}")
print(f"   Device: {DEVICE}")

# Print layer details
print("\n   Architecture breakdown:")
print(f"   ┌─ Team Embedding:    nn.Embedding({len(team_encoder.classes_)}, 32)")
print(f"   ├─ Numerical Network: Linear(5→128) → BatchNorm → ReLU → Dropout")
print(f"   │                     Linear(128→64) → BatchNorm → ReLU → Dropout")
print(f"   └─ Output Layer:      Linear(32+32+64=128 → 128) → BN → ReLU → Dropout(0.3)")
print(f"                         Linear(128→64) → BN → ReLU → Dropout(0.3)")
print(f"                         Linear(64→3)")

# Step 3: Input/Output specs
print("\n" + "=" * 70)
print("2️⃣  INPUTS & OUTPUTS")
print("=" * 70)

print("\n   🔸 INPUT 1: Team identities (categorical)")
print(f"      2 teams (home + away) via Embedding(323 teams, 32 dims each)")
print(f"      Embedding dimension: 32 per team → 64 combined")

print(f"\n   🔸 INPUT 2: Numerical features ({len(numerical_cols)}):")
for i, c in enumerate(numerical_cols):
    desc = {
        'home_elo': 'Elo rating of home team before match (0-2500)',
        'away_elo': 'Elo rating of away team before match (0-2500)',
        'elo_diff': 'Elo difference (home - away)',
        'tournament_weight': 'Importance of match (0.5=Friendly, 4.0=W Cup)',
        'neutral_int': 'Venue type (0=Home/Away, 1=Neutral)',
    }
    print(f"      {i+1}. {c:<20} → {desc.get(c, '')}")

print("\n   🔸 OUTPUT: 3 classes (logits → Softmax probabilities):")
print(f"      Class 0: Home Win  (probability)")
print(f"      Class 1: Away Win  (probability)")
print(f"      Class 2: Draw      (probability)")
print(f"      → Argmax for final prediction")

print("\n   🔸 LOSS FUNCTION: Cross-Entropy Loss")
print(f"      L = -Σ(y_i · log(p_i))")
print(f"      where y = one-hot ground truth, p = predicted probabilities")
print(f"      Optimizer: Adam (lr=0.001, weight_decay=0)")

# Step 4: Load and prepare test data
print("\n" + "=" * 70)
print("3️⃣  TEST SET EVALUATION (on held-out matches from 2020-2026)")
print("=" * 70)

print("\n   Loading match data...")
df = load_local()
df['date'] = pd.to_datetime(df['date'])

# Use recent matches for test (2020 onwards)
df_test = df[df['date'] >= '2020-01-01'].copy()
print(f"   Using {len(df_test)} matches from 2020-2026 as test set")

# Compute Elo for test matches
print("   Computing Elo ratings...")
df_all = df[df['date'] >= '2000-01-01'].sort_values('date').reset_index(drop=True)
df_all = compute_elo_ratings(df_all)

# Get only test-period matches (with Elo computed)
df_test = df_all[df_all['date'] >= '2020-01-01'].copy()
print(f"   Test matches with Elo: {len(df_test)}")

if len(df_test) > 0:
    # Encode teams
    home_ids = []
    away_ids = []
    valid_idx = []
    
    for idx, row in df_test.iterrows():
        try:
            h_id = team_encoder.transform([row['home_team']])[0]
            a_id = team_encoder.transform([row['away_team']])[0]
            home_ids.append(h_id)
            away_ids.append(a_id)
            valid_idx.append(idx)
        except:
            pass
    
    print(f"   Valid test matches (teams in model): {len(valid_idx)}")
    
    if len(valid_idx) > 0:
        # Build features
        X_num = np.zeros((len(valid_idx), len(numerical_cols)))
        for i, c in enumerate(numerical_cols):
            if c in df_test.columns:
                X_num[:, i] = df_test.loc[valid_idx, c].fillna(0).values
            else:
                X_num[:, i] = 0
        
        # Target: 0=HomeWin, 1=AwayWin, 2=Draw
        y = np.zeros(len(valid_idx), dtype=int)
        for i, idx in enumerate(valid_idx):
            hs = df_test.loc[idx, 'home_score']
            aws = df_test.loc[idx, 'away_score']
            if hs > aws:
                y[i] = 0
            elif hs < aws:
                y[i] = 1
            else:
                y[i] = 2
        
        # Scale and predict
        X_scaled = scaler.transform(X_num)
        dataset = SimpleDataset(X_scaled, np.array(home_ids), np.array(away_ids), y)
        loader = DataLoader(dataset, batch_size=256)
        
        all_probs = []
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for Xn, hid, aid, yb in loader:
                Xn, hid, aid = Xn.to(DEVICE), hid.to(DEVICE), aid.to(DEVICE)
                out = model(Xn, hid, aid)
                probs = torch.softmax(out, dim=1)
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(torch.argmax(out, 1).cpu().numpy())
                all_labels.extend(yb.numpy())
        
        acc = accuracy_score(all_labels, all_preds)
        try:
            ll = log_loss(all_labels, all_probs)
        except:
            ll = float('nan')
        
        print(f"\n   📊 TEST METRICS:")
        print(f"   {'─'*50}")
        print(f"   {'Accuracy:':<30} {acc*100:.2f}%")
        print(f"   {'Log Loss:':<30} {ll:.4f}")
        print(f"   {'Test samples:':<30} {len(all_labels)}")
        
        # Per-class metrics
        print(f"\n   📋 PER-CLASS BREAKDOWN:")
        print(f"   {'─'*50}")
        target_names = ['Home Win', 'Away Win', 'Draw']
        report = classification_report(all_labels, all_preds, target_names=target_names, output_dict=True, zero_division=0)
        for cls_name in target_names:
            cls = report[cls_name]
            print(f"   {cls_name:<12} | Precision: {cls['precision']*100:>5.2f}% | Recall: {cls['recall']*100:>5.2f}% | F1: {cls['f1-score']:.3f} | Support: {cls['support']:>5}")
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        print(f"\n   📊 CONFUSION MATRIX (rows=true, cols=predicted):")
        print(f"   {'':>12} {'Home Win':>10} {'Away Win':>10} {'Draw':>10}")
        for i, cls_name in enumerate(target_names):
            print(f"   {cls_name:<12} {cm[i][0]:>10} {cm[i][1]:>10} {cm[i][2]:>10}")
        
        # Baseline comparison
        baseline_home_win = sum(1 for l in all_labels if l == 0) / len(all_labels)
        print(f"\n   📈 BASELINE COMPARISON:")
        print(f"   {'─'*50}")
        print(f"   {'Model Accuracy:':<30} {acc*100:.2f}%")
        print(f"   {'Always predict Home Win:':<30} {baseline_home_win*100:.2f}%")
        print(f"   {'Improvement over baseline:':<30} {(acc - baseline_home_win)*100:+.2f}%")
    else:
        print("❌ No valid test matches found.")
else:
    print("❌ No test matches available.")

print("\n" + "=" * 70)
print("✅ EVALUATION COMPLETE")
print("=" * 70)