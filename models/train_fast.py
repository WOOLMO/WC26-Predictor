"""
Optimized training pipeline - uses Elo + numerical features only (no per-row loops).
Trains quickly on the filtered dataset.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, log_loss
import pickle
import json
import sys
from pathlib import Path
import os

sys.path.insert(0, str(Path(__file__).parent.parent))
from features.elo import compute_elo_ratings
from features.engineer import get_tournament_weight
from data.fetch_data import load_local

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')
MODELS_DIR = Path(__file__).parent
DATA_DIR = Path(__file__).parent.parent / "data"


class SimpleMatchDataset(Dataset):
    def __init__(self, X_num, home_ids, away_ids, y):
        self.X_num = torch.FloatTensor(X_num)
        self.home_ids = torch.LongTensor(home_ids)
        self.away_ids = torch.LongTensor(away_ids)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (self.X_num[idx], self.home_ids[idx], self.away_ids[idx], self.y[idx])


class FootballPredictor(nn.Module):
    def __init__(self, n_teams, n_num, embed_dim=32, hidden=128):
        super().__init__()
        self.team_embed = nn.Embedding(n_teams, embed_dim)
        self.num_net = nn.Sequential(
            nn.Linear(n_num, hidden),
            nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2), nn.ReLU(), nn.Dropout(0.3),
        )
        self.out = nn.Linear(2 * embed_dim + hidden // 2, 3)

    def forward(self, X_num, home_id, away_id):
        h = self.team_embed(home_id)
        a = self.team_embed(away_id)
        n = self.num_net(X_num)
        return self.out(torch.cat([h, a, n], dim=1))


def prepare_fast(df):
    """Fast feature preparation without per-row loops."""
    print("Preparing features...")
    
    # Compute Elo ratings first
    df = compute_elo_ratings(df)
    
    # Get team IDs
    all_teams = pd.unique(df[['home_team', 'away_team']].values.ravel())
    team_encoder = LabelEncoder()
    team_encoder.fit(all_teams)
    
    df['home_id'] = team_encoder.transform(df['home_team'])
    df['away_id'] = team_encoder.transform(df['away_team'])
    
    # Tournament weight
    if 'tournament' in df.columns:
        df['tournament_weight'] = df['tournament'].apply(get_tournament_weight)
    else:
        df['tournament_weight'] = 1.0
    
    # Neutral indicator
    df['neutral_int'] = df.get('neutral', pd.Series([False]*len(df))).astype(int)
    
    # Build numerical features (no per-row loops)
    num_cols = ['home_elo', 'away_elo', 'elo_diff', 'tournament_weight', 'neutral_int']
    
    # Add recent avg goals (simple: use last 5 matches per team, pre-computed with shift)
    # For simplicity in fast mode, just use Elo + tournament weight + neutral
    # This is still quite powerful
    X_num = df[num_cols].values
    
    # Target
    y = ((df['home_score'] < df['away_score']).astype(int) * 1 + 
         (df['home_score'] == df['away_score']).astype(int) * 2).values
    
    # Scale
    scaler = StandardScaler()
    X_num_scaled = scaler.fit_transform(X_num)
    
    # Chronological split
    n = len(df)
    train_n = int(n * 0.7)
    val_n = int(n * 0.15)
    
    datasets = {}
    for name, s, e in [('train', 0, train_n), ('val', train_n, train_n+val_n), ('test', train_n+val_n, n)]:
        datasets[name] = SimpleMatchDataset(
            X_num_scaled[s:e], df['home_id'].values[s:e], 
            df['away_id'].values[s:e], y[s:e])
    
    print(f"Train: {train_n}, Val: {val_n}, Test: {n - train_n - val_n}")
    print(f"Numerical features: {num_cols}")
    
    return datasets, team_encoder, scaler, num_cols, df


def train_fast(continue_training: bool = True):
    """Train the model. If continue_training=True and saved model exists, load it and continue."""
    print("=" * 60)
    print("🏆 WORLD CUP PREDICTOR - FAST TRAINING")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    
    # Load data
    print("\n📥 Loading data...")
    df = load_local()
    if df is None:
        print("❌ No data. Run fetch first.")
        return
    
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'] >= '1995-01-01'].sort_values('date').reset_index(drop=True)
    print(f"Using {len(df)} matches from 1995+")
    
    # Prepare features
    datasets, team_encoder, scaler, num_cols, df_processed = prepare_fast(df)
    
    # Check for existing model to continue training from
    artifacts_path = MODELS_DIR / "model_artifacts"
    config_path = artifacts_path / "model_config.json"
    weights_path = artifacts_path / "football_model.pt"
    
    model = None
    loaded_epochs = 0
    
    if continue_training and config_path.exists() and weights_path.exists():
        try:
            with open(config_path, 'r') as f:
                saved_config = json.load(f)
            # Verify compatibility
            if (saved_config['n_teams'] == len(team_encoder.classes_) and 
                saved_config.get('n_num', saved_config.get('n_numerical_features')) == len(num_cols)):
                model = FootballPredictor(
                    n_teams=saved_config['n_teams'],
                    n_num=saved_config.get('n_num', saved_config.get('n_numerical_features')),
                    embed_dim=saved_config.get('embed_dim', 32),
                    hidden=saved_config.get('hidden_dim', saved_config.get('hidden', 128))
                ).to(DEVICE)
                model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
                print(f"\n🔄 Loaded previous best model — continuing training from saved weights")
                # Check for training state (epoch count)
                state_path = artifacts_path / "training_state.json"
                if state_path.exists():
                    with open(state_path, 'r') as f:
                        state = json.load(f)
                        loaded_epochs = state.get('epochs_completed', 0)
                        print(f"   Previously trained for {loaded_epochs} epochs")
            else:
                print(f"\n⚠️  Saved model incompatible (different teams/features). Creating fresh model.")
        except Exception as e:
            print(f"\n⚠️  Could not load previous model: {e}. Creating fresh model.")
    
    if model is None:
        model = FootballPredictor(
            n_teams=len(team_encoder.classes_),
            n_num=len(num_cols),
            embed_dim=32,
            hidden=128
        ).to(DEVICE)
        print(f"\n🧠 Created new model from scratch")
    
    print(f"   Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Training setup
    train_loader = DataLoader(datasets['train'], batch_size=128, shuffle=True)
    val_loader = DataLoader(datasets['val'], batch_size=128)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train — continue from where we left off
    best_val_acc = 0
    best_state = None
    start_epoch = loaded_epochs
    total_epochs = 30 + loaded_epochs  # train up to 30 MORE epochs
    
    print(f"\n🚀 Training (epochs {start_epoch+1}-{total_epochs})...")
    for epoch in range(start_epoch, total_epochs):
        model.train()
        train_loss = 0
        for Xn, hid, aid, y in train_loader:
            Xn, hid, aid, y = Xn.to(DEVICE), hid.to(DEVICE), aid.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(Xn, hid, aid), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validate
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for Xn, hid, aid, y in val_loader:
                Xn, hid, aid, y = Xn.to(DEVICE), hid.to(DEVICE), aid.to(DEVICE), y.to(DEVICE)
                out = model(Xn, hid, aid)
                val_loss += criterion(out, y).item()
                _, pred = torch.max(out, 1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        
        val_acc = correct / total
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict().copy()
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"   Epoch {epoch+1:2d}/{total_epochs} | Train Loss: {train_loss/len(train_loader):.4f} | Val Acc: {val_acc:.4f}")
    
    # Load best model
    model.load_state_dict(best_state)
    
    # Test
    test_loader = DataLoader(datasets['test'], batch_size=128)
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for Xn, hid, aid, y in test_loader:
            Xn, hid, aid, y = Xn.to(DEVICE), hid.to(DEVICE), aid.to(DEVICE), y.to(DEVICE)
            out = model(Xn, hid, aid)
            probs = torch.softmax(out, dim=1)
            all_preds.extend(torch.argmax(out, 1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    test_acc = accuracy_score(all_labels, all_preds)
    print(f"\n📊 Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    # Save artifacts
    artifacts_dir = MODELS_DIR / "model_artifacts"
    os.makedirs(artifacts_dir, exist_ok=True)
    
    torch.save(model.state_dict(), artifacts_dir / "football_model.pt")
    config = {
        'n_teams': len(team_encoder.classes_),
        'n_numerical_features': len(num_cols),
        'n_num': len(num_cols),
        'embed_dim': 32, 'hidden_dim': 128, 'n_classes': 3
    }
    with open(artifacts_dir / "model_config.json", 'w') as f:
        json.dump(config, f)
    with open(artifacts_dir / "team_encoder.pkl", 'wb') as f:
        pickle.dump(team_encoder, f)
    with open(artifacts_dir / "scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)
    with open(artifacts_dir / "numerical_cols.pkl", 'wb') as f:
        pickle.dump(num_cols, f)
    
    # Save training state so retraining picks up from here
    training_state = {'epochs_completed': total_epochs}
    with open(artifacts_dir / "training_state.json", 'w') as f:
        json.dump(training_state, f)
    print(f"   Training state saved ({total_epochs} total epochs)")
    
    print(f"💾 Model saved to {artifacts_dir}/")
    print(f"\n✅ TRAINING COMPLETE! Test accuracy: {test_acc:.4f}")
    
    return model, team_encoder, scaler, num_cols


if __name__ == "__main__":
    train_fast()