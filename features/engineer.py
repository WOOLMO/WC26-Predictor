"""
Feature engineering for football match prediction.
Transforms raw match data into features suitable for ML/DL models.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional

from .elo import compute_elo_ratings

# Tournament weight mapping
TOURNAMENT_WEIGHTS = {
    'Friendly': 0.5,
    'FIFA World Cup qualification': 2.0,
    'UEFA Euro qualification': 2.0,
    'African Cup of Nations qualification': 2.0,
    'Copa América qualification': 2.0,
    'Asian Cup qualification': 2.0,
    'AFC Asian Cup qualification': 2.0,
    'FIFA World Cup': 4.0,
    'UEFA Euro': 3.5,
    'African Cup of Nations': 3.0,
    'Copa América': 3.0,
    'AFC Asian Cup': 3.0,
    'CONCACAF Gold Cup': 2.5,
    'OFC Nations Cup': 2.0,
    'Confederations Cup': 3.0,
    'AFC Challenge Cup': 2.0,
    'AFC Solidarity Cup': 1.5,
    'CFU Caribbean Cup': 1.5,
    'Copa Centroamericana': 1.5,
    'UNCAF Cup': 1.5,
    'CONCACAF Nations League': 2.0,
    'UEFA Nations League': 2.5,
    'AFC Asian Cup qualifiers': 1.5,
    'CAF World Cup qualifiers': 2.0,
    'CAF Confederation Cup': 2.0,
    'COSAFA Cup': 1.5,
    'CECAF Cup': 1.0,
    'WAFU Cup': 1.0,
    'FIFA Confederations Cup': 3.0,
}

# WC 2026 qualified teams (as of June 2026, known qualifiers + likely ones)
WC2026_TEAMS = sorted([
    # Hosts
    'United States', 'Canada', 'Mexico',
    # UEFA (16 spots)
    'Germany', 'France', 'Spain', 'England', 'Portugal', 'Netherlands',
    'Belgium', 'Italy', 'Croatia', 'Switzerland', 'Denmark',
    'Austria', 'Serbia', 'Turkey', 'Ukraine', 'Poland',
    # CONMEBOL (6-7 spots)
    'Argentina', 'Brazil', 'Uruguay', 'Colombia', 'Ecuador', 'Peru',
    # AFC (8 spots)
    'Japan', 'South Korea', 'Iran', 'Australia', 'Saudi Arabia', 'Qatar',
    'Iraq', 'United Arab Emirates',
    # CAF (9-10 spots)
    'Morocco', 'Senegal', 'Nigeria', 'Egypt', 'Tunisia', 'Algeria',
    'Cameroon', 'Ghana', 'Ivory Coast',
    # CONCACAF (4-6 spots including hosts)
    'Costa Rica', 'Panama', 'Jamaica',
    # OFC (1-2 spots)
    'New Zealand',
])


def get_tournament_weight(tournament: str) -> float:
    """Map tournament name to importance weight."""
    for key, weight in TOURNAMENT_WEIGHTS.items():
        if key.lower() in tournament.lower():
            return weight
    return 1.0


def compute_recent_form(df: pd.DataFrame, team: str, date: str, n_matches: int = 10) -> dict:
    """
    Compute recent form features for a team before a given date.
    """
    team_matches = df[
        ((df['home_team'] == team) | (df['away_team'] == team)) &
        (df['date'] < date)
    ].tail(n_matches)

    if len(team_matches) == 0:
        return {
            'avg_goals_scored': 0,
            'avg_goals_conceded': 0,
            'win_rate': 0,
            'recent_points': 0,
            'form_streak': 0,
            'n_recent_matches': 0,
        }

    goals_scored = []
    goals_conceded = []
    points = []
    streak = 0

    for _, match in team_matches.iterrows():
        if match['home_team'] == team:
            scored = match['home_score']
            conceded = match['away_score']
        else:
            scored = match['away_score']
            conceded = match['home_score']

        goals_scored.append(scored)
        goals_conceded.append(conceded)

        if scored > conceded:
            points.append(3)
            streak = max(streak, 0) + 1 if streak >= 0 else 1
        elif scored < conceded:
            points.append(0)
            streak = min(streak, 0) - 1 if streak <= 0 else -1
        else:
            points.append(1)
            streak = 0

    n = len(team_matches)
    return {
        'avg_goals_scored': np.mean(goals_scored),
        'avg_goals_conceded': np.mean(goals_conceded),
        'win_rate': sum(1 for p in points if p == 3) / n,
        'recent_points': sum(points),
        'form_streak': streak,
        'n_recent_matches': n,
    }


def compute_h2h(df: pd.DataFrame, team_a: str, team_b: str, date: str, n_matches: int = 5) -> dict:
    """
    Compute head-to-head features between two teams.
    """
    h2h = df[
        ((df['home_team'] == team_a) & (df['away_team'] == team_b) |
         (df['home_team'] == team_b) & (df['away_team'] == team_a)) &
        (df['date'] < date)
    ].tail(n_matches)

    if len(h2h) == 0:
        return {'h2h_wins_a': 0, 'h2h_wins_b': 0, 'h2h_draws': 0, 'h2h_total': 0}

    wins_a = 0
    wins_b = 0
    draws = 0
    for _, match in h2h.iterrows():
        if match['home_team'] == team_a:
            if match['home_score'] > match['away_score']:
                wins_a += 1
            elif match['home_score'] < match['away_score']:
                wins_b += 1
            else:
                draws += 1
        else:
            if match['home_score'] > match['away_score']:
                wins_b += 1
            elif match['home_score'] < match['away_score']:
                wins_a += 1
            else:
                draws += 1

    return {
        'h2h_wins_a': wins_a,
        'h2h_wins_b': wins_b,
        'h2h_draws': draws,
        'h2h_total': len(h2h),
    }


def engineer_features(df: pd.DataFrame, include_elo: bool = True) -> pd.DataFrame:
    """
    Full feature engineering pipeline.
    Returns a DataFrame with features engineered for each match.
    """
    print("🔧 Engineering features...")
    df = df.sort_values('date').reset_index(drop=True)

    # Ensure date is datetime
    df['date'] = pd.to_datetime(df['date'])

    # Compute Elo ratings
    if include_elo:
        df = compute_elo_ratings(df)

    # Tournament weight
    if 'tournament' in df.columns:
        df['tournament_weight'] = df['tournament'].apply(get_tournament_weight)
    else:
        df['tournament_weight'] = 1.0

    # Neutral venue
    if 'neutral' not in df.columns:
        df['neutral'] = False

    # Compute recent form and H2H for each match
    features = []

    for idx, row in df.iterrows():
        feat = {}
        home = row['home_team']
        away = row['away_team']
        date = row['date']
        date_str = str(date)

        # Team identity features (will be encoded separately)
        feat['home_team'] = home
        feat['away_team'] = away
        feat['date'] = date

        # Elo features
        if include_elo:
            feat['home_elo'] = row['home_elo']
            feat['away_elo'] = row['away_elo']
            feat['elo_diff'] = row['elo_diff']

        # Tournament importance
        feat['tournament_weight'] = row['tournament_weight']

        # Neutral venue
        feat['neutral'] = int(row['neutral'])

        # Recent form for home team
        home_form = compute_recent_form(df, home, date, n_matches=10)
        feat['home_avg_goals_scored'] = home_form['avg_goals_scored']
        feat['home_avg_goals_conceded'] = home_form['avg_goals_conceded']
        feat['home_win_rate'] = home_form['win_rate']
        feat['home_form_streak'] = home_form['form_streak']
        feat['home_recent_points'] = home_form['recent_points']
        feat['home_n_matches'] = home_form['n_recent_matches']

        # Recent form for away team
        away_form = compute_recent_form(df, away, date, n_matches=10)
        feat['away_avg_goals_scored'] = away_form['avg_goals_scored']
        feat['away_avg_goals_conceded'] = away_form['avg_goals_conceded']
        feat['away_win_rate'] = away_form['win_rate']
        feat['away_form_streak'] = away_form['form_streak']
        feat['away_recent_points'] = away_form['recent_points']
        feat['away_n_matches'] = away_form['n_recent_matches']

        # Head-to-head
        h2h = compute_h2h(df, home, away, date, n_matches=5)
        feat['h2h_home_wins'] = h2h['h2h_wins_a']
        feat['h2h_away_wins'] = h2h['h2h_wins_b']
        feat['h2h_draws'] = h2h['h2h_draws']
        feat['h2h_total'] = h2h['h2h_total']

        # Target: result from home team perspective
        if row['home_score'] > row['away_score']:
            feat['target'] = 0  # Home win
        elif row['home_score'] < row['away_score']:
            feat['target'] = 1  # Away win
        else:
            feat['target'] = 2  # Draw

        # Target: goals (for Poisson-based models)
        feat['home_goals'] = row['home_score']
        feat['away_goals'] = row['away_score']

        features.append(feat)

    result = pd.DataFrame(features)
    print(f"✅ Features engineered: {result.shape[0]} matches, {result.shape[1]} columns")
    return result