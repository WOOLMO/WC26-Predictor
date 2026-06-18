"""
Comprehensive end-to-end test of all features.
Tests: dataset, model loading, prediction, simulation, CLI commands.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 60)
print("COMPREHENSIVE TEST SUITE")
print("=" * 60)

# [1/5] Dataset
print("\n[1/5] Checking dataset...")
from data.fetch_data import load_local
df = load_local()
assert df is not None, "Dataset missing"
print(f"  ✅ {len(df)} matches loaded ({df['date'].min()} to {df['date'].max()})")

# [2/5] Model artifacts
print("\n[2/5] Checking model artifacts...")
from models.predict import load_model_artifacts
model, te, scaler, cols = load_model_artifacts()
params = sum(p.numel() for p in model.parameters())
assert params > 10000, f"Too few params: {params}"
print(f"  ✅ Model: {params:,} params, {len(te.classes_)} teams, {len(cols)} features")

# [3/5] Single match prediction
print("\n[3/5] Testing match prediction...")
from models.predict import predict_future_match

# Test 1: France vs Argentina
result = predict_future_match("France", "Argentina")
assert result is not None, "Prediction failed"
hw, aw, dr = result['home_win_prob'], result['away_win_prob'], result['draw_prob']
probs_sum = hw + aw + dr
assert 0.99 < probs_sum < 1.01, f"Probabilities don't sum to 1: {probs_sum}"
print(f"  ✅ France vs Argentina: HW={hw*100:.1f}% AW={aw*100:.1f}% Draw={dr*100:.1f}%")

# Test 2: Neutral venue
result2 = predict_future_match("Brazil", "Germany", neutral=True)
assert result2 is not None, "Neutral prediction failed"
print(f"  ✅ Brazil vs Germany (neutral): HW={result2['home_win_prob']*100:.1f}% AW={result2['away_win_prob']*100:.1f}% Draw={result2['draw_prob']*100:.1f}%")

# Test 3: Unknown team
result3 = predict_future_match("UnknownTeamXYZ", "Germany")
assert result3 is None, "Should return None for unknown team"
print(f"  ✅ Unknown team correctly returns None")

# [4/5] Cup simulation
print("\n[4/5] Testing cup simulation (5 sims)...")
from models.predict import simulate_world_cup
results = simulate_world_cup(model, te, scaler, cols, n_simulations=5)
assert results is not None and len(results) > 0, "Simulation returned no results"
assert sum(results.values()) > 0.99, f"Probabilities don't sum to 1: {sum(results.values())}"
top_team = list(results.keys())[0]
top_prob = results[top_team]
print(f"  ✅ Simulation complete! {len(results)} unique winners")
print(f"  ✅ Top winner: {top_team} ({top_prob*100:.2f}%)")

# Test simulation with specific teams
print("\n  Testing simulation with WC teams only...")
from models.predict import WC2026_TEAMS
results2 = simulate_world_cup(model, te, scaler, cols, teams_list=WC2026_TEAMS, n_simulations=3)
assert results2 is not None and len(results2) > 0, "WC teams simulation failed"
print(f"  ✅ WC teams simulation: {len(results2)} unique winners")

# [5/5] CLI commands
print("\n[5/5] Testing CLI commands...")
import subprocess
import os

def test_cli(cmd, expected_exit=0):
    """Run a CLI command and return whether it succeeded."""
    full_cmd = f'cd /d \"{Path.cwd()}\" && python Main.py {cmd} > NUL 2>&1'
    r = os.system(full_cmd)
    status = "✅" if r == expected_exit else "❌"
    print(f"  {status} python Main.py {cmd}")
    return r == expected_exit

all_pass = True
all_pass &= test_cli("list-teams")
all_pass &= test_cli("info")
all_pass &= test_cli('predict "France" "Argentina"')
all_pass &= test_cli('predict "Brazil" "Germany" --neutral')

# Test cup with 2 sims (fast)
all_pass &= test_cli("cup --sims 2")

print(f"\n{'='*60}")
if all_pass:
    print("ALL TESTS PASSED ✅")
else:
    print("SOME TESTS FAILED ⚠️")
print("=" * 60)

# Summary
print(f"\n📊 SUMMARY:")
print(f"  Dataset:     {len(df):,} matches")
print(f"  Model:       {params:,} parameters, {len(te.classes_)} teams, {len(cols)} features")
print(f"  Prediction:  France {hw*100:.1f}% vs Argentina {aw*100:.1f}% (Draw {dr*100:.1f}%)")
print(f"  Simulation:  {len(results)} unique winners in {5} runs")
print(f"  Top winner:  {top_team} ({top_prob*100:.2f}%)")