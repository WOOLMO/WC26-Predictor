"""
Elo rating system for international football teams.
"""
import numpy as np
import pandas as pd
from typing import Dict

# Starting Elo rating for new teams
INITIAL_ELO = 1500
# K-factor determines how much ratings change after a match
K_FACTOR = 30
# Home advantage in Elo terms
HOME_ADVANTAGE = 100
# Expected result margin of victory multiplier
MARGIN_MULTIPLIER = {
    1: 1.0,   # 1 goal diff
    2: 1.5,   # 2 goal diff
    3: 1.75,  # 3 goal diff
    4: 1.9,   # 4 goal diff
    5: 2.0,   # 5+ goal diff
}


def expected_score(rating_a: float, rating_b: float) -> float:
    """Calculate expected score for team A against team B."""
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def goal_margin_multiplier(goal_diff: int) -> float:
    """Get margin multiplier based on goal difference."""
    abs_diff = abs(goal_diff)
    for threshold, mult in sorted(MARGIN_MULTIPLIER.items()):
        if abs_diff <= threshold:
            return mult
    return 2.0


def update_elo(home_elo: float, away_elo: float, home_goals: int, away_goals: int,
               is_neutral: bool = False) -> tuple:
    """
    Update Elo ratings after a match.
    Returns: (new_home_elo, new_away_elo)
    """
    # Apply home advantage
    effective_home = home_elo + (0 if is_neutral else HOME_ADVANTAGE)

    # Expected scores
    exp_home = expected_score(effective_home, away_elo)
    exp_away = 1 - exp_home

    # Actual scores
    if home_goals > away_goals:
        actual_home, actual_away = 1.0, 0.0
    elif home_goals < away_goals:
        actual_home, actual_away = 0.0, 1.0
    else:
        actual_home, actual_away = 0.5, 0.5

    # Goal difference multiplier
    goal_diff = abs(home_goals - away_goals)
    margin_mult = goal_margin_multiplier(goal_diff)

    # Update ratings
    new_home = home_elo + K_FACTOR * margin_mult * (actual_home - exp_home)
    new_away = away_elo + K_FACTOR * margin_mult * (actual_away - exp_away)

    return new_home, new_away


def compute_elo_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Elo ratings for all teams across all matches chronologically.
    Returns the DataFrame with additional columns: home_elo, away_elo, elo_diff.
    """
    df = df.sort_values('date').reset_index(drop=True)
    ratings: Dict[str, float] = {}
    
    home_elos = []
    away_elos = []
    elo_diffs = []

    for _, row in df.iterrows():
        home = row['home_team']
        away = row['away_team']
        home_goals = row['home_score']
        away_goals = row['away_score']
        
        # Determine if neutral venue
        is_neutral = row.get('neutral', False)
        # Many datasets don't have a neutral column; use tournament as proxy
        if 'tournament' in row:
            # Friendly matches are often neutral or at varied venues
            pass

        # Initialize ratings for new teams
        if home not in ratings:
            ratings[home] = INITIAL_ELO
        if away not in ratings:
            ratings[away] = INITIAL_ELO

        current_home_elo = ratings[home]
        current_away_elo = ratings[away]

        home_elos.append(current_home_elo)
        away_elos.append(current_away_elo)
        elo_diffs.append(current_home_elo - current_away_elo)

        # Update ratings
        new_home, new_away = update_elo(
            current_home_elo, current_away_elo,
            home_goals, away_goals,
            is_neutral=is_neutral
        )
        ratings[home] = new_home
        ratings[away] = new_away

    df = df.copy()
    df['home_elo'] = home_elos
    df['away_elo'] = away_elos
    df['elo_diff'] = elo_diffs

    return df