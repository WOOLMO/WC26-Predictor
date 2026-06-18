"""
Deep Learning model for football match outcome prediction.
Uses PyTorch with embedding layers for teams + numerical features.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
import pickle
import json
from pathlib import Path
import sys
import os

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from features.engineer import engineer_features
from data.fetch_data import fetch_dataset, load_local

MODELS_DIR = Path(__file__).parent
DATA_DIR = Path(__file__).parent.parent / "data"

# Device configuration: prefer CUDA / Apple MPS / CPU
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')
print(f"Using device: {DEVICE}")


class MatchDataset(Dataset):
    """PyTorch Dataset for match data."""

    def __init__(self, numerical_features, team_home_ids, team_away_ids, labels):
        self.numerical = torch.FloatTensor(numerical_features)
        self.home_ids = torch.LongTensor(team_home_ids)
        self.away_ids = torch.LongTensor(team_away_ids)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'numerical': self.numerical[idx],
            'home_id': self.home_ids[idx],
            'away_id': self.away_ids[idx],
            'label': self.labels[idx],
        }


class FootballPredictor(nn.Module):
    """
    Deep Learning model for football match prediction.
    Uses team embeddings + numerical features.
    """
    def __init__(self, n_teams: int, n_numerical_features: int,
                 embed_dim: int = 32, hidden_dim: int = 128,
                 n_classes: int = 3, dropout: float = 0.3):
        super().__init__()

        # Team embeddings
        self.team_embedding = nn.Embedding(n_teams, embed_dim)

        # Numerical feature processing
        self.numerical_net = nn.Sequential(
            nn.Linear(n_numerical_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Combined processing (2 * embed_dim from both teams + hidden_dim // 2 from numerical)
        combined_dim = 2 * embed_dim + hidden_dim // 2
        self.combined_net = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_classes),
        )

    def forward(self, home_id, away_id, numerical):
        # Get team embeddings
        home_embed = self.team_embedding(home_id)  # (batch, embed_dim)
        away_embed = self.team_embedding(away_id)  # (batch, embed_dim)

        # Process numerical features
        numerical_out = self.numerical_net(numerical)  # (batch, hidden_dim // 2)

        # Concatenate all features
        combined = torch.cat([home_embed, away_embed, numerical_out], dim=1)

        # Final classification
        out = self.combined_net(combined)
        return out


def prepare_data(df_features: pd.DataFrame, test_size: float = 0.15, val_size: float = 0.15):
    """
    Prepare data for training: encode teams, scale numerical features, split sets.
    """
    # Encode team names
    all_teams = pd.unique(df_features[['home_team', 'away_team']].values.ravel())
    team_encoder = LabelEncoder()
    team_encoder.fit(all_teams)

    df_features['home_team_id'] = team_encoder.transform(df_features['home_team'])
    df_features['away_team_id'] = team_encoder.transform(df_features['away_team'])

    # Numerical feature columns (exclude non-feature columns)
    exclude_cols = {'home_team', 'away_team', 'date', 'target',
                    'home_goals', 'away_goals', 'home_team_id', 'away_team_id'}
    numerical_cols = [c for c in df_features.columns if c not in exclude_cols]

    print(f"Numerical features ({len(numerical_cols)}): {numerical_cols}")

    # Handle NaN values
    df_features = df_features.fillna(0)

    # Extract features and labels
    X_num = df_features[numerical_cols].values
    y = df_features['target'].values

    # Scale numerical features
    scaler = StandardScaler()
    X_num_scaled = scaler.fit_transform(X_num)

    # Split data chronologically (by sorted dates)
    dates = pd.to_datetime(df_features['date'])
    n = len(df_features)
    idx = np.arange(n)

    # Chronological split: test = last 15%, val = previous 15%, train = first 70%
    test_cut = int(n * (1 - test_size))
    val_cut = int(n * (1 - test_size - val_size))

    train_idx = idx[:val_cut]
    val_idx = idx[val_cut:test_cut]
    test_idx = idx[test_cut:]

    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    # Create datasets
    def make_dataset(indices):
        return MatchDataset(
            numerical_features=X_num_scaled[indices],
            team_home_ids=df_features['home_team_id'].values[indices],
            team_away_ids=df_features['away_team_id'].values[indices],
            labels=y[indices]
        )

    train_dataset = make_dataset(train_idx)
    val_dataset = make_dataset(val_idx)
    test_dataset = make_dataset(test_idx)

    return (train_dataset, val_dataset, test_dataset,
            team_encoder, scaler, numerical_cols)


def train_model(train_dataset, val_dataset, n_teams, n_numerical_features,
                batch_size=64, epochs=50, lr=0.001, weight_decay=1e-5):
    """Train the football prediction model."""

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = FootballPredictor(
        n_teams=n_teams,
        n_numerical_features=n_numerical_features,
        embed_dim=32,
        hidden_dim=128,
        n_classes=3,
        dropout=0.3
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    early_stop_patience = 10

    print(f"\n🚀 Training for up to {epochs} epochs...")
    print(f"   Batch size: {batch_size}, LR: {lr}")
    print(f"   Model params: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            numerical = batch['numerical'].to(DEVICE)
            home_id = batch['home_id'].to(DEVICE)
            away_id = batch['away_id'].to(DEVICE)
            labels = batch['label'].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(home_id, away_id, numerical)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_preds = []
        val_labels_list = []

        with torch.no_grad():
            for batch in val_loader:
                numerical = batch['numerical'].to(DEVICE)
                home_id = batch['home_id'].to(DEVICE)
                away_id = batch['away_id'].to(DEVICE)
                labels = batch['label'].to(DEVICE)

                outputs = model(home_id, away_id, numerical)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                val_preds.extend(outputs.cpu().numpy())
                val_labels_list.extend(labels.cpu().numpy())

        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)

        scheduler.step(avg_val_loss)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"   Epoch {epoch+1:3d}/{epochs} | Train Loss: {train_loss/len(train_loader):.4f} (Acc: {train_acc:.4f}) | Val Loss: {avg_val_loss:.4f} (Acc: {val_acc:.4f})")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"   ⏹️  Early stopping at epoch {epoch+1}")
                break

    # Load best model
    model.load_state_dict(best_model_state)
    return model


def evaluate_model(model, test_dataset, team_encoder, scaler, numerical_cols):
    """Evaluate the model on test data."""
    test_loader = DataLoader(test_dataset, batch_size=64)
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in test_loader:
            numerical = batch['numerical'].to(DEVICE)
            home_id = batch['home_id'].to(DEVICE)
            away_id = batch['away_id'].to(DEVICE)
            labels = batch['label'].to(DEVICE)

            outputs = model(home_id, away_id, numerical)
            probs = torch.softmax(outputs, dim=1)

            all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    try:
        ll = log_loss(all_labels, all_probs)
    except:
        ll = 0.0

    print(f"\n📊 Test Results:")
    print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   Log Loss: {ll:.4f}")

    # Per-class metrics
    from sklearn.metrics import classification_report, confusion_matrix
    print(f"\n{classification_report(all_labels, all_preds, target_names=['Home Win', 'Away Win', 'Draw'])}")

    return accuracy, all_preds, all_probs


def save_model_artifacts(model, team_encoder, scaler, numerical_cols, path=None):
    """Save model and preprocessing artifacts."""
    if path is None:
        path = MODELS_DIR / "model_artifacts"

    os.makedirs(path, exist_ok=True)

    # Save model weights
    model_path = path / "football_model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"💾 Model saved to {model_path}")

    # Save model architecture config
    config = {
        'n_teams': len(team_encoder.classes_),
        'n_numerical_features': len(numerical_cols),
        'embed_dim': 32,
        'hidden_dim': 128,
        'n_classes': 3,
    }
    with open(path / "model_config.json", 'w') as f:
        json.dump(config, f)

    # Save team encoder
    with open(path / "team_encoder.pkl", 'wb') as f:
        pickle.dump(team_encoder, f)

    # Save scaler
    with open(path / "scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)

    # Save numerical columns
    with open(path / "numerical_cols.pkl", 'wb') as f:
        pickle.dump(numerical_cols, f)

    print(f"💾 All artifacts saved to {path}/")


def load_model_artifacts(path=None):
    """Load model and preprocessing artifacts."""
    if path is None:
        path = MODELS_DIR / "model_artifacts"

    # Load config
    with open(path / "model_config.json", 'r') as f:
        config = json.load(f)

    # Re-create model
    model = FootballPredictor(
        n_teams=config['n_teams'],
        n_numerical_features=config['n_numerical_features'],
        embed_dim=config['embed_dim'],
        hidden_dim=config['hidden_dim'],
        n_classes=config['n_classes']
    ).to(DEVICE)

    # Load weights
    model_path = path / "football_model.pt"
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    # Load encoders
    with open(path / "team_encoder.pkl", 'rb') as f:
        team_encoder = pickle.load(f)
    with open(path / "scaler.pkl", 'rb') as f:
        scaler = pickle.load(f)
    with open(path / "numerical_cols.pkl", 'rb') as f:
        numerical_cols = pickle.load(f)

    print(f"📂 Model and artifacts loaded from {path}/")
    return model, team_encoder, scaler, numerical_cols


def run_training_pipeline():
    """Full training pipeline: fetch data, engineer features, train, evaluate."""
    print("=" * 60)
    print("🏆 WORLD CUP PREDICTOR - TRAINING PIPELINE")
    print("=" * 60)

    # Step 1: Get data
    print("\n📥 Step 1: Fetching data...")
    df = load_local()
    if df is None:
        df = fetch_dataset()
    if df is None:
        print("❌ No data available. Exiting.")
        return

    # Save raw data
    os.makedirs(DATA_DIR / "processed", exist_ok=True)
    df.to_csv(DATA_DIR / "processed" / "raw_data.csv", index=False)

    # Step 2: Engineer features
    print("\n🔧 Step 2: Engineering features...")
    df_features = engineer_features(df)

    # Filter to relevant period (e.g., last 30 years for more relevant data)
    df_features = df_features[pd.to_datetime(df_features['date']) >= '1995-01-01']
    print(f"   Using matches from 1995 onwards: {len(df_features)} matches")

    # Step 3: Prepare data
    print("\n📊 Step 3: Preparing data splits...")
    train_dataset, val_dataset, test_dataset, team_encoder, scaler, numerical_cols = \
        prepare_data(df_features)

    # Step 4: Train model
    print("\n🧠 Step 4: Training deep learning model...")
    model = train_model(
        train_dataset, val_dataset,
        n_teams=len(team_encoder.classes_),
        n_numerical_features=len(numerical_cols),
        batch_size=64,
        epochs=50,
        lr=0.001
    )

    # Step 5: Evaluate
    print("\n📈 Step 5: Evaluating model...")
    evaluate_model(model, test_dataset, team_encoder, scaler, numerical_cols)

    # Step 6: Save
    print("\n💾 Step 6: Saving model artifacts...")
    save_model_artifacts(model, team_encoder, scaler, numerical_cols)

    # Save processed features for later use
    df_features.to_csv(DATA_DIR / "processed" / "features.csv", index=False)
    print(f"   Features saved to {DATA_DIR / 'processed' / 'features.csv'}")

    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print("=" * 60)

    return model, team_encoder, scaler, numerical_cols, df_features


if __name__ == "__main__":
    run_training_pipeline()