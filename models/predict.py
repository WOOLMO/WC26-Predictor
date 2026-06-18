"""
Match prediction and WC 2026 simulation using the trained model.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
import sys
import os
import json
import pickle
from typing import List, Tuple, Dict
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')
MODELS_DIR = Path(__file__).parent
DATA_DIR = MODELS_DIR.parent / "data"




class FootballPredictor(nn.Module):
    """
    Hybrid Embedding + MLP architecture.
    
    NOT a GNN: This is a standard feedforward network with learned team embeddings.
    Match prediction doesn't need graph convolutions — teams play pairwise matches,
    and embeddings learn latent team strengths (similar to recommender systems).
    
    Architecture (5-layer network):
        Layer 1: nn.Embedding(323, 32) — learns 32-dim vector per team
        Layer 2: nn.Linear(5, 128) + BatchNorm + ReLU + Dropout  
        Layer 3: nn.Linear(128, 64) + BatchNorm + ReLU + Dropout
        Layer 4: Concat[emb32, emb32, num64] → nn.Linear(128, 64) + BN + ReLU + Dropout
        Layer 5: nn.Linear(64, 3) → Softmax output
    """
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


def load_model_artifacts(path=None):
    """Load model and preprocessing artifacts."""
    if path is None:
        path = MODELS_DIR / "model_artifacts"
    with open(path / "model_config.json", 'r') as f:
        config = json.load(f)
    model = FootballPredictor(
        n_teams=config['n_teams'],
        n_num=config.get('n_num', config.get('n_numerical_features', 5))
    ).to(DEVICE)
    model.load_state_dict(torch.load(path / "football_model.pt", map_location=DEVICE))
    model.eval()
    with open(path / "team_encoder.pkl", 'rb') as f:
        team_encoder = pickle.load(f)
    with open(path / "scaler.pkl", 'rb') as f:
        scaler = pickle.load(f)
    with open(path / "numerical_cols.pkl", 'rb') as f:
        numerical_cols = pickle.load(f)
    print(f"📂 Model loaded ({len(team_encoder.classes_)} teams, {len(numerical_cols)} features)")
    return model, team_encoder, scaler, numerical_cols


def predict_match(model, team_encoder, scaler, numerical_cols,
                  home_team: str, away_team: str,
                  features_dict: dict = None) -> Dict:
    """
    Predict outcome of a single match.
    Model output classes: 0=HomeWin, 1=AwayWin, 2=Draw
    
    features_dict can override defaults. The model expects these exact keys:
    home_elo, away_elo, elo_diff, tournament_weight, neutral_int
    """
    model.eval()
    try:
        home_id = torch.LongTensor([team_encoder.transform([home_team])[0]]).to(DEVICE)
        away_id = torch.LongTensor([team_encoder.transform([away_team])[0]]).to(DEVICE)
    except ValueError as e:
        print(f"⚠️  Unknown team: {e}")
        return None
    
    # Build feature vector: initialize ALL numerical columns to 0
    vec_dict = {col: 0.0 for col in numerical_cols}
    
    # Set sensible defaults
    if 'home_elo' in vec_dict:
        vec_dict['home_elo'] = 1500.0
    if 'away_elo' in vec_dict:
        vec_dict['away_elo'] = 1500.0
    if 'elo_diff' in vec_dict:
        vec_dict['elo_diff'] = 0.0
    if 'tournament_weight' in vec_dict:
        vec_dict['tournament_weight'] = 3.0
    if 'neutral_int' in vec_dict:
        vec_dict['neutral_int'] = 0
    
    # Override with provided features (simulation passes 'neutral_int' and 'tournament_weight')
    if features_dict:
        for k, v in features_dict.items():
            if k in vec_dict:
                vec_dict[k] = v
    
    # Build vector in EXACT column order
    vec = np.array([vec_dict[c] for c in numerical_cols], dtype=np.float32).reshape(1, -1)
    vec_scaled = torch.FloatTensor(scaler.transform(vec)).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(vec_scaled, home_id, away_id)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
    
    return {
        'home_team': home_team,
        'away_team': away_team,
        'home_win_prob': float(probs[0]),  # Class 0
        'away_win_prob': float(probs[1]),  # Class 1
        'draw_prob': float(probs[2]),      # Class 2
        'prediction': ['Home Win', 'Away Win', 'Draw'][np.argmax(probs)],
        'confidence': float(np.max(probs)),
    }


def predict_future_match(home_team: str, away_team: str,
                          neutral: bool = False, tournament: str = "FIFA World Cup"):
    """CLI-friendly single match prediction."""
    try:
        model, team_encoder, scaler, numerical_cols = load_model_artifacts()
    except Exception as e:
        print(f"❌ No trained model found: {e}")
        return None
    
    features = {}
    if 'neutral_int' in numerical_cols:
        features['neutral_int'] = int(neutral)
    if 'tournament_weight' in numerical_cols:
        features['tournament_weight'] = 4.0 if 'World Cup' in tournament else 2.0
    
    return predict_match(model, team_encoder, scaler, numerical_cols,
                         home_team, away_team, features)


# ============================================================
# WC 2026 Tournament Simulation
# ============================================================

WC2026_TEAMS = sorted([
    'United States', 'Canada', 'Mexico',
    'Germany', 'France', 'Spain', 'England', 'Portugal', 'Netherlands',
    'Belgium', 'Italy', 'Croatia', 'Switzerland', 'Denmark',
    'Austria', 'Serbia', 'Turkey', 'Ukraine', 'Poland',
    'Argentina', 'Brazil', 'Uruguay', 'Colombia', 'Ecuador', 'Peru',
    'Japan', 'South Korea', 'Iran', 'Australia', 'Saudi Arabia', 'Qatar',
    'Iraq', 'United Arab Emirates',
    'Morocco', 'Senegal', 'Nigeria', 'Egypt', 'Tunisia', 'Algeria',
    'Cameroon', 'Ghana', 'Ivory Coast',
    'Costa Rica', 'Panama', 'Jamaica',
    'New Zealand',
])


def sample_match_outcome(model, team_encoder, scaler, numerical_cols,
                         home: str, away: str, features: dict) -> Tuple[str, str, dict]:
    """
    Simulate a single match using the model's probability distribution.
    Returns: (winner_or_None_for_draw, loser_or_None_for_draw, detail_dict)
    """
    result = predict_match(model, team_encoder, scaler, numerical_cols, home, away, features)
    if result is None:
        return None, None, None
    
    probs = np.array([result['home_win_prob'], result['away_win_prob'], result['draw_prob']])
    probs = probs / probs.sum()
    outcome = int(np.random.choice([0, 1, 2], p=probs))
    
    if outcome == 0:  # Home Win
        hg, ag = max(1, np.random.poisson(1.8)), max(0, np.random.poisson(0.6))
        if hg <= ag: hg, ag = ag + 1, ag
        winner, loser = home, away
    elif outcome == 1:  # Away Win
        hg, ag = max(0, np.random.poisson(0.6)), max(1, np.random.poisson(1.8))
        if ag <= hg: ag, hg = hg + 1, hg
        winner, loser = away, home
    else:  # Draw
        g = max(0, np.random.poisson(1.2))
        hg, ag = g, g
        # Return winner=None for draw so caller can award 1pt to both
        winner, loser = None, None
    
    detail = {
        'home': home, 'away': away,
        'home_goals': int(hg), 'away_goals': int(ag),
        'home_win_prob': result['home_win_prob'],
        'away_win_prob': result['away_win_prob'],
        'draw_prob': result['draw_prob'],
        'outcome': ['Home Win', 'Away Win', 'Draw'][outcome],
    }
    return winner, loser, detail


def simulate_group_stage(model, team_encoder, scaler, numerical_cols,
                         groups: Dict[str, List[str]]) -> Tuple[Dict, List]:
    """
    Simulate group stage. Groups of 3: each team plays each other once.
    Returns: (standings_dict, all_matches_list)
    - standings: {group_letter: [1st, 2nd, 3rd]} sorted by points then GD then GF
    """
    standings = {}
    all_matches = []
    
    for group_name, teams in groups.items():
        pts = {t: 0 for t in teams}
        gf, ga = {t: 0 for t in teams}, {t: 0 for t in teams}
        
        for i in range(len(teams)):
            for j in range(i + 1, len(teams)):
                home, away = teams[i], teams[j]
                # Build correct feature dict with the right key names
                feat = {'neutral_int': 1, 'tournament_weight': 4.0}
                
                # Alternate home/away for fairness
                if (i + j) % 2 == 0:
                    w, l, detail = sample_match_outcome(
                        model, team_encoder, scaler, numerical_cols, home, away, feat)
                else:
                    w, l, detail = sample_match_outcome(
                        model, team_encoder, scaler, numerical_cols, away, home, feat)
                    # Flip back so detail always shows (home, away) as (teams[i], teams[j])
                    if detail:
                        detail['home'], detail['away'] = home, away
                        detail['home_goals'], detail['away_goals'] = detail['away_goals'], detail['home_goals']
                
                if detail:
                    all_matches.append(detail)
                    hg, ag = detail['home_goals'], detail['away_goals']
                    gf[home] += hg; ga[home] += ag
                    gf[away] += ag; ga[away] += hg
                    
                    # Correct point allocation: winner=None means draw
                    if w == home: pts[home] += 3
                    elif w == away: pts[away] += 3
                    else: pts[home] += 1; pts[away] += 1  # Draw
        
        # Sort: points desc, then goal difference desc, then goals for desc
        standings[group_name] = sorted(teams, key=lambda t: (pts[t], gf[t] - ga[t], gf[t]), reverse=True)
    
    return standings, all_matches


def simulate_knockout_match(model, team_encoder, scaler, numerical_cols,
                             home: str, away: str) -> Tuple[str, dict]:
    """Simulate knockout match. Draws → penalty shootout (50/50)."""
    feat = {'neutral_int': 1, 'tournament_weight': 4.0}
    result = predict_match(model, team_encoder, scaler, numerical_cols, home, away, feat)
    if result is None:
        return np.random.choice([home, away]), None
    
    probs = np.array([result['home_win_prob'], result['away_win_prob'], result['draw_prob']])
    probs = probs / probs.sum()
    outcome = int(np.random.choice([0, 1, 2], p=probs))
    
    if outcome == 0:
        winner, hg, ag = home, 2, 1
    elif outcome == 1:
        winner, hg, ag = away, 1, 2
    else:
        # Draw → penalty shootout
        winner = np.random.choice([home, away])
        hg, ag = 1, 1  # 1-1 after extra time, decided on penalties
    
    return winner, {'home': home, 'away': away, 'home_goals': hg, 'away_goals': ag, 'winner': winner}


def simulate_world_cup(model, team_encoder, scaler, numerical_cols,
                        teams_list: List[str] = None,
                        n_simulations: int = 500) -> Dict[str, float]:
    """
    Full WC 2026 Monte Carlo simulation.
    
    Format: 48 teams, 16 groups of 3, top 2 advance
    Knockout: R32 → R16 → QF → SF → Final
    
    For each simulation:
    1. Random group draw (16 groups of 3)
    2. Simulate all group matches (3 per group = 48 group matches)
    3. Top 2 from each group advance to R32
    4. Simulate knockout matches until 1 winner
    5. Record winner + advancement stats
    """
    # Prepare 48 teams
    known_teams = set(team_encoder.classes_)
    base = [t for t in (teams_list or WC2026_TEAMS) if t in known_teams]
    extras = sorted([t for t in known_teams if t not in base])
    while len(base) < 48 and extras:
        base.append(extras.pop(0))
    teams_list = sorted(base)[:48]
    
    print(f"   Teams: {len(teams_list)} (48-team World Cup)")
    print(f"   Format: 16 groups of 3 → R32 → R16 → QF → SF → Final")
    print(f"   Running {n_simulations} simulations...")
    
    # Stats tracking
    winner_counts = Counter()
    finalist_counts = Counter()
    semi_counts = Counter()
    quarter_counts = Counter()
    r16_counts = Counter()
    r32_counts = Counter()
    
    for sim in range(n_simulations):
        if (sim + 1) % max(1, n_simulations // 10) == 0:
            pct = (sim + 1) / n_simulations * 100
            print(f"   [{pct:3.0f}%] {sim+1}/{n_simulations} simulations")
        
        # 1. Random group draw
        shuffled = teams_list.copy()
        np.random.shuffle(shuffled)
        groups = {chr(ord('A') + i): shuffled[i*3:(i+1)*3] for i in range(16)}
        
        # 2. Group stage
        standings, _ = simulate_group_stage(model, team_encoder, scaler, numerical_cols, groups)
        
        # 3. Top 2 from each group → R32
        all_winners = [standings[g][0] for g in sorted(groups.keys())]
        all_runners = [standings[g][1] for g in sorted(groups.keys())]
        r32_teams = all_winners + all_runners
        for t in r32_teams:
            r32_counts[t] += 1
        
        # 4. Knockout bracket: R32
        # Pair winners vs runners-up cross-group style
        bracket = [(all_winners[i], all_runners[(i + 1) % 16]) for i in range(16)]
        
        round_num = 0
        while len(bracket) > 0:
            n_in_round = len(bracket) * 2
            next_round = []
            round_winners = []
            
            for home, away in bracket:
                w, _ = simulate_knockout_match(model, team_encoder, scaler, numerical_cols, home, away)
                if w:
                    next_round.append(w)
                    round_winners.append(w)
            
            # Track who reached which round
            if n_in_round == 32:  # Just completed R32
                for t in round_winners:
                    r16_counts[t] += 1
            elif n_in_round == 16:  # Just completed R16
                for t in round_winners:
                    quarter_counts[t] += 1
            elif n_in_round == 8:  # Just completed QF
                for t in round_winners:
                    semi_counts[t] += 1
            elif n_in_round == 4:  # Just completed SF → finalists
                for t in round_winners:
                    finalist_counts[t] += 1
            
            if len(next_round) <= 1:
                if len(next_round) == 1:
                    winner_counts[next_round[0]] += 1
                break
            
            # Pair up for next round
            bracket = [(next_round[i], next_round[i+1]) for i in range(0, len(next_round), 2)]
            round_num += 1
    
    total = sum(winner_counts.values())
    if total == 0:
        print("❌ No simulations completed successfully.")
        return {}
    
    results = {team: count / total for team, count in winner_counts.most_common()}
    
    # Sort results by probability descending
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    
    # Print results
    print(f"\n{'='*70}")
    print(f"🏆 WORLD CUP 2026 — SIMULATION RESULTS")
    print(f"{'='*70}")
    print(f"   {n_simulations} Monte Carlo simulations | 48 teams | 16 groups of 3")
    print(f"{'='*70}")
    
    print(f"\n{'🏆 WINNER LEADERBOARD':<70}")
    print(f"{'─'*70}")
    print(f"{'Rank':<6} {'Team':<26} {'Win %':<10} {'Wins':<8} {'Final%':<10}")
    print(f"{'─'*70}")
    for rank, (team, prob) in enumerate(sorted_results[:20], 1):
        w = winner_counts[team]
        fp = finalist_counts.get(team, 0) / max(1, n_simulations) * 100
        print(f"{rank:<6} {team:<26} {prob*100:>6.2f}%   {w:>3}/{total}  {fp:>5.1f}%")
    
    print(f"\n{'📊 ADVANCEMENT ODDS (Top 10)':<70}")
    print(f"{'─'*70}")
    print(f"{'Team':<26} {'R32':<7} {'R16':<7} {'QF':<7} {'SF':<7} {'Final':<7} {'🏆':<7}")
    print(f"{'─'*70}")
    for team, _ in sorted_results[:10]:
        r32 = r32_counts.get(team, 0) / n_simulations * 100
        r16 = r16_counts.get(team, 0) / n_simulations * 100
        qf = quarter_counts.get(team, 0) / n_simulations * 100
        sf = semi_counts.get(team, 0) / n_simulations * 100
        fn = finalist_counts.get(team, 0) / n_simulations * 100
        w = winner_counts.get(team, 0) / n_simulations * 100
        print(f"{team:<26} {r32:>4.0f}%  {r16:>4.0f}%  {qf:>4.0f}%  {sf:>4.0f}%  {fn:>5.1f}%  {w:>4.2f}%")
    
    print(f"\n{'='*70}")
    winner_name = list(results.keys())[0]
    winner_prob = results[winner_name]
    print(f"🔮 PREDICTED WORLD CUP 2026 WINNER: {winner_name}")
    print(f"   Win probability: {winner_prob*100:.2f}% (1 in {max(1, int(1/winner_prob))})")
    print(f"{'='*70}")
    
    return results