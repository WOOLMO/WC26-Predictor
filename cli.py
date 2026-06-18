#!/usr/bin/env python3
"""
CLI tool for World Cup 2026 Match Predictor.
Train model, predict matches, and simulate the entire tournament.
"""
import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from data.fetch_data import fetch_dataset
from models.predict import MODELS_DIR
from models.predict import (
    load_model_artifacts, predict_future_match, simulate_world_cup,
    WC2026_TEAMS, predict_match
)
from features.engineer import engineer_features
from features.elo import compute_elo_ratings


def cmd_fetch(args):
    """Download the international football results dataset."""
    print("📥 Fetching dataset...")
    df = fetch_dataset()
    if df is not None:
        print(f"✅ Dataset downloaded: {df.shape[0]} matches")
        print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
    else:
        print("❌ Failed to fetch dataset.")


def cmd_train(args):
    """Train the deep learning model."""
    print("🚀 Starting training pipeline...")
    from models.train_fast import train_fast
    train_fast()


def cmd_predict(args):
    """Predict outcome of a single match."""
    result = predict_future_match(args.home, args.away, args.neutral, args.tournament)
    
    if result is None:
        print("❌ Could not make prediction.")
        return
    
    print(f"\n{'='*50}")
    print(f"🏟️  {result['home_team']} vs {result['away_team']}")
    print(f"{'='*50}")
    print(f"   Tournament: {args.tournament}")
    print(f"   Venue: {'Neutral' if args.neutral else 'Home/Away'}")
    print()
    print(f"   📊 Probabilities:")
    print(f"      {result['home_team']:<20} Win: {result['home_win_prob']*100:>5.1f}%")
    print(f"      {'Draw':<20}         {result['draw_prob']*100:>5.1f}%")
    print(f"      {result['away_team']:<20} Win: {result['away_win_prob']*100:>5.1f}%")
    print()
    print(f"   🎯 Prediction: {result['prediction']} (confidence: {result['confidence']*100:.1f}%)")
    print(f"{'='*50}")


def cmd_simulate(args):
    """Simulate WC 2026 tournament (alias for cup)."""
    cmd_cup(args)


def cmd_cup(args):
    """Predict the WC 2026 winner by simulating the full tournament."""
    print("🏆 WORLD CUP 2026 PREDICTOR")
    print("=" * 50)
    print("Loading model...")
    try:
        model, team_encoder, scaler, numerical_cols = load_model_artifacts()
    except Exception as e:
        print(f"❌ No trained model found: {e}")
        print("   Run 'python cli.py train' first.")
        return
    
    n_sim = args.simulations
    print(f"\n{'='*50}")
    print(f"🎯 Simulating WC 2026 to predict the winner")
    print(f"{'='*50}")
    
    results = simulate_world_cup(
        model, team_encoder, scaler, numerical_cols,
        teams_list=WC2026_TEAMS,
        n_simulations=n_sim
    )
    
    if results:
        top_team = list(results.keys())[0]
        top_prob = results[top_team]
        print(f"\n{'='*50}")
        print(f"🔮 PREDICTED WINNER: {top_team}")
        print(f"   Win probability: {top_prob*100:.2f}%")
        print(f"{'='*50}")


def cmd_list_teams(args):
    """List all teams in the dataset / model."""
    try:
        _, team_encoder, _, _ = load_model_artifacts()
        teams = sorted(team_encoder.classes_)
        print(f"\n📋 Teams in model ({len(teams)}):")
        for i, team in enumerate(teams):
            print(f"   {i+1:3d}. {team}")
    except:
        print("❌ No model found. Run 'python cli.py train' first.")


def cmd_info(args):
    """Show model information and statistics."""
    try:
        model, team_encoder, scaler, numerical_cols = load_model_artifacts()
        print(f"\n📊 Model Information:")
        print(f"   Teams: {len(team_encoder.classes_)}")
        print(f"   Numerical features: {len(numerical_cols)}")
        print(f"   Feature names: {numerical_cols}")
        print(f"   Device: {next(model.parameters()).device}")
        print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    except Exception as e:
        print(f"❌ No trained model found: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="🏆 World Cup 2026 Match Predictor CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py fetch                    # Download match data
  python cli.py train                    # Train the model
  python cli.py predict France Argentina # Predict a match
  python cli.py simulate --sims 1000     # Simulate WC 2026 (1000 simulations)
  python cli.py list-teams               # List all known teams
  python cli.py info                     # Show model info
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Fetch
    p_fetch = subparsers.add_parser('fetch', help='Download international football results dataset')
    
    # Train
    p_train = subparsers.add_parser('train', help='Train the deep learning model')
    
    # Predict
    p_predict = subparsers.add_parser('predict', help='Predict match outcome')
    p_predict.add_argument('home', type=str, help='Home team name')
    p_predict.add_argument('away', type=str, help='Away team name')
    p_predict.add_argument('--neutral', action='store_true', help='Neutral venue')
    p_predict.add_argument('--tournament', type=str, default='FIFA World Cup',
                          help='Tournament name (default: FIFA World Cup)')
    
    # Simulate
    p_sim = subparsers.add_parser('simulate', help='Simulate WC 2026 tournament')
    p_sim.add_argument('--sims', '--simulations', type=int, default=500,
                       dest='simulations', help='Number of simulations (default: 500)')
    
    # Cup
    p_cup = subparsers.add_parser('cup', help='🏆 Predict WC 2026 winner (full tournament simulation)')
    p_cup.add_argument('--sims', '--simulations', type=int, default=500,
                       dest='simulations', help='Number of simulations (default: 500)')
    
    # List teams
    subparsers.add_parser('list-teams', help='List all teams known to the model')
    
    # Info
    subparsers.add_parser('info', help='Show model information')
    
    args = parser.parse_args()
    
    if args.command == 'fetch':
        cmd_fetch(args)
    elif args.command == 'train':
        cmd_train(args)
    elif args.command == 'predict':
        cmd_predict(args)
    elif args.command == 'simulate':
        cmd_simulate(args)
    elif args.command == 'cup':
        cmd_cup(args)
    elif args.command == 'list-teams':
        cmd_list_teams(args)
    elif args.command == 'info':
        cmd_info(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()