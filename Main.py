#!/usr/bin/env python3
"""
🏆 World Cup 2026 Match Predictor
==================================
A CLI app using Deep Learning to predict match outcomes and simulate
the entire FIFA World Cup 2026 tournament.

Quick start:
  python Main.py fetch         # Download dataset
  python Main.py train         # Train the model
  python Main.py predict <home> <away>  # Predict a match
  python Main.py simulate      # Simulate WC 2026
"""

import sys
from pathlib import Path

# Make sure imports work
sys.path.insert(0, str(Path(__file__).parent))

if __name__ == '__main__':
    from cli import main
    main()