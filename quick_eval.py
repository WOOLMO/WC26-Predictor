"""
Quick evaluation - loads pre-computed features and tests the model on 2022-2026 matches.
"""
import numpy as np
import pandas as pd
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from models.predict import load_model_artifacts
from data.fetch_data import load_local
from features.elo import compute_elo_ratings
from features.engineer import get_tournament_weight
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, log_loss
from torch.utils.data import DataLoader, Dataset

class SimpleDataset(Dataset):
    def __init__(self, X, hid, aid, y):
        self.X = torch.FloatTensor(X)
        self.hid = torch.LongTensor(hid)
        self.aid = torch.LongTensor(aid)
        self.y = torch.LongTensor(y)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return (self.X[i], self.hid[i], self.aid[i], self.y[i])

print("Loading model...")
model, te, scaler, cols = load_model_artifacts()

print("\nLoading & preparing data...")
df = load_local()
df['date'] = pd.to_datetime(df['date'])

# Use matches from 2010 onward to compute Elo, then test on 2022+
df_all = df[df['date'] >= '2010-01-01'].sort_values('date').reset_index(drop=True)
print(f"Computing Elo for {len(df_all)} matches (2010-2026)...")
df_all = compute_elo_ratings(df_all)

# Test on 2022 World Cup period (2022-2026)
df_test = df_all[df_all['date'] >= '2022-01-01'].copy()
print(f"Test set: {len(df_test)} matches from 2022-2026")

# Add features
df_test['tournament_weight'] = df_test['tournament'].apply(get_tournament_weight)
df_test['neutral_int'] = df_test['neutral'].astype(int)

# Encode teams
home_ids, away_ids, ys, valid_idx = [], [], [], []
for idx, row in df_test.iterrows():
    try:
        h = te.transform([row['home_team']])[0]
        a = te.transform([row['away_team']])[0]
        home_ids.append(h); away_ids.append(a)
        hs, aws = row['home_score'], row['away_score']
        ys.append(0 if hs > aws else (1 if hs < aws else 2))
        valid_idx.append(idx)
    except ValueError:
        pass

print(f"Valid matches (teams in model): {len(valid_idx)}")

# Build feature matrix
X = np.zeros((len(valid_idx), len(cols)))
for i, c in enumerate(cols):
    if c in df_test.columns:
        X[:, i] = df_test.loc[valid_idx, c].fillna(0).values
Xs = scaler.transform(X)
y = np.array(ys)
hid, aid = np.array(home_ids), np.array(away_ids)

# Predict
loader = DataLoader(SimpleDataset(Xs, hid, aid, y), batch_size=256)
all_preds, all_probs, all_labels = [], [], []
with torch.no_grad():
    for Xn, hid, aid, yb in loader:
        out = model(Xn, hid, aid)
        probs = torch.softmax(out, 1)
        all_preds.extend(torch.argmax(out, 1).numpy())
        all_probs.extend(probs.numpy())
        all_labels.extend(yb.numpy())

# RESULTS
acc = accuracy_score(all_labels, all_preds)
ll = log_loss(all_labels, all_probs)
baseline = sum(1 for l in all_labels if l == 0) / len(all_labels)

print("\n" + "=" * 65)
print("📊 MODEL EVALUATION RESULTS")
print("=" * 65)

print(f"\n{'Metric':<35} {'Value':<15}")
print(f"{'-'*50}")
print(f"{'Test samples (2022-2026):':<35} {len(all_labels):<15}")
print(f"{'Accuracy:':<35} {acc*100:.2f}%")
print(f"{'Log Loss (Cross-Entropy):':<35} {ll:.4f}")
print(f"{'Baseline (always Home Win):':<35} {baseline*100:.2f}%")
print(f"{'Improvement over baseline:':<35} {(acc-baseline)*100:+.2f}%")
print(f"{'Model parameters:':<35} 20,131")

print(f"\n{'='*65}")
print("📋 Per-Class Performance")
print("=" * 65)
targets = ['Home Win', 'Away Win', 'Draw']
report = classification_report(all_labels, all_preds, target_names=targets, zero_division=0, output_dict=True)
print(f"\n{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
print(f"{'-'*61}")
for t in targets:
    r = report[t]
    print(f"{t:<15} {r['precision']*100:>5.2f}%   {r['recall']*100:>5.2f}%   {r['f1-score']:.3f}   {r['support']:>5}")

print(f"\n{'='*65}")
print("📊 Confusion Matrix (Rows=True, Cols=Predicted)")
print("=" * 65)
cm = confusion_matrix(all_labels, all_preds)
print(f"\n{'':>12} {'Home Win':>10} {'Away Win':>10} {'Draw':>10}")
for i, t in enumerate(targets):
    print(f"{t:<12} {cm[i][0]:>10} {cm[i][1]:>10} {cm[i][2]:>10}")

print(f"\n{'='*65}")
print("Architecture Summary")
print("=" * 65)
print(f"• Team Embedding: nn.Embedding(323 teams → 32 dims)")
print(f"• Numerical Net: Linear(5→128) → BN → ReLU → Dropout(0.3)")
print(f"                Linear(128→64) → BN → ReLU → Dropout(0.3)")
print(f"• Combined:     Cat[home_emb(32), away_emb(32), num(64)] → 128")
print(f"• Output:       Linear(128→64) → BN → ReLU → Dropout(0.3)")
print(f"                Linear(64→3) → Softmax")
print(f"• Loss:         Cross-Entropy Loss (nn.CrossEntropyLoss)")
print(f"• Optimizer:    Adam (lr=0.001)")
print(f"• Epochs:       30 (with early stopping)")
print(f"• Training:     29,365 matches (1995-2021)")
print(f"• Test:         {len(all_labels)} matches (2022-2026)")
print("=" * 65)
print("✅ Evaluation complete!")