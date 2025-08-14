#!/usr/bin/env python3
"""Example showing both breakout and dropoff pipelines working together."""

import sys
import os
sys.path.append('src')

from player_breakout_pipeline import PlayerBreakoutPipeline
from player_dropoff_pipeline import PlayerDropoffPipeline
import pandas as pd

def main():
    """Demonstrate both pipelines and compare results."""
    print("Fantasy Football ML Pipeline Comparison")
    print("=" * 50)
    
    # Use recent seasons for demonstration
    seasons = [2022, 2023]
    
    print(f"Using seasons: {seasons}")
    print("This will predict for 2024 season based on 2023 data")
    
    print("\n🚀 BREAKOUT PIPELINE")
    print("-" * 30)
    
    # Create and run breakout pipeline
    breakout_pipeline = PlayerBreakoutPipeline(
        seasons=seasons,
        breakout_threshold=0.3,
        min_training_seasons=1
    )
    
    print("Loading data and training breakout model...")
    data = breakout_pipeline.load_and_prepare_data()
    breakout_results = breakout_pipeline.train_model(data)
    breakout_predictions = breakout_pipeline.predict_current_season(data, 2024)
    
    print(f"✅ Breakout model trained (AUC: {breakout_results['test_auc']:.3f})")
    print(f"✅ Generated {len(breakout_predictions)} breakout predictions")
    
    print("\n📉 DROPOFF PIPELINE")
    print("-" * 30)
    
    # Create and run dropoff pipeline  
    dropoff_pipeline = PlayerDropoffPipeline(
        seasons=seasons,
        dropoff_threshold=0.2,
        min_training_seasons=1
    )
    
    print("Loading data and training dropoff model...")
    dropoff_data = dropoff_pipeline.load_and_prepare_data()
    dropoff_results = dropoff_pipeline.train_model(dropoff_data)
    dropoff_predictions = dropoff_pipeline.predict_current_season(dropoff_data, 2024)
    
    print(f"✅ Dropoff model trained (AUC: {dropoff_results['test_auc']:.3f})")
    print(f"✅ Generated {len(dropoff_predictions)} dropoff predictions")
    
    print("\n🔍 ANALYSIS RESULTS")
    print("-" * 30)
    
    # Top breakout candidates
    high_potential = breakout_predictions[breakout_predictions['potential_tier'] == 'High Potential']
    medium_potential = breakout_predictions[breakout_predictions['potential_tier'] == 'Medium Potential']
    
    print(f"\nBreakout Potential Summary:")
    print(f"- High Potential: {len(high_potential)} players")
    print(f"- Medium Potential: {len(medium_potential)} players")
    
    if len(medium_potential) > 0:
        print(f"\nTop 5 Breakout Candidates (Medium+ Potential):")
        top_breakouts = medium_potential.head(5)[['player_name', 'position', 'recent_team', 'breakout_probability']]
        print(top_breakouts.to_string(index=False))
    
    # Top dropoff risks
    high_risk = dropoff_predictions[dropoff_predictions['risk_tier'] == 'High Risk']
    medium_risk = dropoff_predictions[dropoff_predictions['risk_tier'] == 'Medium Risk']
    
    print(f"\nDropoff Risk Summary:")
    print(f"- High Risk: {len(high_risk)} players")
    print(f"- Medium Risk: {len(medium_risk)} players")
    
    if len(medium_risk) > 0:
        print(f"\nTop 5 Dropoff Risks (Medium+ Risk):")
        top_risks = medium_risk.head(5)[['player_name', 'position', 'recent_team', 'dropoff_probability']]
        print(top_risks.to_string(index=False))
    
    print(f"\n💡 FANTASY INSIGHTS")
    print("-" * 30)
    print("- BREAKOUT candidates are players likely to significantly increase their fantasy production")
    print("- DROPOFF risks are players likely to significantly decrease their fantasy production")
    print("- Use breakout predictions to identify undervalued sleepers")
    print("- Use dropoff predictions to avoid overvalued players")
    
    print(f"\n✅ Pipeline comparison completed successfully!")

if __name__ == "__main__":
    main()