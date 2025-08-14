#!/usr/bin/env python3
"""Example usage of the breakout pipeline."""

import sys
import os
sys.path.append('src')

from player_breakout_pipeline import PlayerBreakoutPipeline

def main():
    """Run a simple example of the breakout pipeline."""
    print("Fantasy Football Breakout Player Prediction")
    print("=" * 50)
    
    # Use recent seasons for a quicker example
    seasons = list(range(2020, 2024))
    
    print(f"Using seasons: {seasons}")
    
    # Create pipeline
    pipeline = PlayerBreakoutPipeline(
        seasons=seasons,
        breakout_threshold=0.3,  # 30% increase threshold
        min_training_seasons=2
    )
    
    print("\nPipeline created successfully!")
    print(f"- Breakout threshold: {pipeline.breakout_threshold * 100}%")
    print(f"- Min training seasons: {pipeline.min_training_seasons}")
    
    print("\n🚀 Breakout pipeline is ready to use!")
    print("\nTo run the full pipeline:")
    print("predictions, results = pipeline.run(predict_season=2024, save_csv=True)")
    
    print("\nTo run from command line:")
    print("cd src && python player_breakout_pipeline.py")

if __name__ == "__main__":
    main()