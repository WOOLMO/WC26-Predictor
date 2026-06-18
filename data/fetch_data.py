"""
Fetch international football results dataset.
Downloads from the public GitHub repository of international football results (1872-present).
"""
import os
import pandas as pd
import requests
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent
RAW_PATH = DATA_DIR / "raw_results.csv"
PROCESSED_DIR = DATA_DIR / "processed"

# Public dataset URL - International football results from 1872 to present
# This is the well-known martj42/international_results dataset mirrored on GitHub
CSV_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"

def fetch_dataset(url=CSV_URL, output_path=RAW_PATH):
    """Download the international football results dataset."""
    print(f"Downloading dataset from: {url}")
    try:
        df = pd.read_csv(url)
        os.makedirs(output_path.parent, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"✅ Dataset saved to {output_path}")
        print(f"   Shape: {df.shape}")
        print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"   Tournaments: {df['tournament'].nunique()}")
        print(f"   Teams: {pd.unique(df[['home_team', 'away_team']].values.ravel()).size}")
        return df
    except Exception as e:
        print(f"❌ Failed to download: {e}")
        print("   Trying alternative source...")
        return fetch_dataset_alternative(output_path)


def fetch_dataset_alternative(output_path=RAW_PATH):
    """Fallback: scrape from a second mirror."""
    # Alternative: kaggle datasets mirrored on GitHub
    alt_urls = [
        "https://raw.githubusercontent.com/jalapic/engsoccerdata/master/data-raw/results.csv",
    ]
    for url in alt_urls:
        try:
            print(f"   Trying: {url}")
            df = pd.read_csv(url)
            os.makedirs(output_path.parent, exist_ok=True)
            df.to_csv(output_path, index=False)
            print(f"✅ Dataset saved to {output_path}")
            return df
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            continue
    print("❌ Could not fetch dataset from any source.")
    print("   Please manually download from: https://www.kaggle.com/datasets/martj42/international-football-results")
    return None


def load_local():
    """Load the dataset from local file if it exists."""
    if RAW_PATH.exists():
        df = pd.read_csv(RAW_PATH)
        print(f"📂 Loaded local dataset: {RAW_PATH}")
        return df
    else:
        print("⚠️  No local dataset found. Run this script to download it.")
        return None


if __name__ == "__main__":
    df = fetch_dataset()
    if df is not None:
        print("\nFirst 5 rows:")
        print(df.head())