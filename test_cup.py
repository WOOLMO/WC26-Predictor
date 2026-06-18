"""
Quick test of the cup simulation - 5 simulations to verify it works end-to-end.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from models.predict import load_model_artifacts, simulate_world_cup

print("=" * 60)
print("TESTING WC 2026 CUP SIMULATION")
print("=" * 60)

model, te, scaler, cols = load_model_artifacts()
print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params")
print(f"Teams: {len(te.classes_)}, Features: {len(cols)}")

# Quick prediction test
from models.predict import predict_future_match
result = predict_future_match("France", "Argentina")
if result:
    print(f"\nFrance vs Argentina:")
    print(f"  Home Win: {result['home_win_prob']*100:.1f}%")
    print(f"  Away Win: {result['away_win_prob']*100:.1f}%")
    print(f"  Draw: {result['draw_prob']*100:.1f}%")

print(f"\nRunning 5 cup simulations...")
results = simulate_world_cup(model, te, scaler, cols, n_simulations=5)
if results:
    print(f"\n✅ Simulation works! Top 3:")
    for rank, (team, prob) in enumerate(list(results.items())[:3], 1):
        print(f"  {rank}. {team}: {prob*100:.1f}%")
else:
    print("❌ Simulation failed")