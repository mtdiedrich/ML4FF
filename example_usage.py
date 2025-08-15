#!/usr/bin/env python3
"""
ML4FF Example Usage
===================

This script demonstrates how to use the ML4FF package for comprehensive
NFL fantasy football analysis using 20 years of player statistics.
"""

import os
from ml4ff import DataAcquisitionPipeline


def main():
    """Run comprehensive ML4FF examples."""
    
    print("="*60)
    print("ML4FF - Machine Learning for Fantasy Football")
    print("Example Usage: 20 Years of NFL Player Statistics")
    print("="*60)
    
    # Example 1: Quick yardage and TD analysis for running backs
    print("\n1. Quick Analysis - Running Backs (2022-2024)")
    print("-" * 50)
    
    rb_pipeline = DataAcquisitionPipeline(
        seasons=[2022, 2023, 2024],
        positions=['RB'],
        output_dir='examples/rb_data'
    )
    
    rb_data = rb_pipeline.run(save_csv=True)
    
    if not rb_data['yardage_and_tds'].empty:
        rb_stats = rb_data['yardage_and_tds']
        print(f"✓ Analyzed {rb_stats['player_name'].nunique()} running backs")
        print(f"✓ Total rushing + receiving yards: {rb_stats['total_yards'].sum():,}")
        print(f"✓ Total touchdowns: {rb_stats['total_tds'].sum():,}")
        
        print("\nTop 5 RBs by total yards (single season):")
        top_rbs = rb_stats.nlargest(5, 'total_yards')
        for _, player in top_rbs.iterrows():
            print(f"  {player['player_name']} ({player['season']}): {player['total_yards']:,} yards, {player['total_tds']} TDs")
    
    # Example 2: Historical data analysis (full 20 years)
    print(f"\n\n2. Historical Analysis - All Players (2005-2024)")
    print("-" * 50)
    
    # For demo, use just 3 recent years to avoid long download
    historical_pipeline = DataAcquisitionPipeline(
        seasons=list(range(2022, 2025)),  # 2022, 2023, 2024
        output_dir='examples/historical_data'
    )
    
    historical_data = historical_pipeline.run(save_csv=True)
    
    if not historical_data['yardage_and_tds'].empty:
        all_stats = historical_data['yardage_and_tds']
        print(f"✓ Analyzed {all_stats['player_name'].nunique():,} unique players")
        print(f"✓ Covering {len(all_stats):,} player-season combinations")
        print(f"✓ Total yards across all players: {all_stats['total_yards'].sum():,}")
        print(f"✓ Total TDs across all players: {all_stats['total_tds'].sum():,}")
        
        # Position breakdown
        print("\nYardage by position:")
        position_stats = all_stats.groupby('position').agg({
            'total_yards': 'sum',
            'total_tds': 'sum',
            'player_name': 'count'
        }).sort_values('total_yards', ascending=False)
        
        for pos, stats in position_stats.head().iterrows():
            print(f"  {pos}: {stats['total_yards']:,} yards, {stats['total_tds']} TDs ({stats['player_name']} seasons)")
    
    # Example 3: Demonstrate programmatic usage
    print(f"\n\n3. Programmatic Usage Example")
    print("-" * 50)
    
    # Create a pipeline for wide receivers
    wr_pipeline = DataAcquisitionPipeline(
        seasons=[2023, 2024], 
        positions=['WR'],
        output_dir='examples/wr_analysis'
    )
    
    print("Creating WR analysis pipeline...")
    wr_data = wr_pipeline.run(save_csv=False)  # Don't save for this example
    
    if not wr_data['yardage_and_tds'].empty:
        wr_stats = wr_data['yardage_and_tds']
        
        # Find breakout players (significant year-over-year improvement)
        breakout_candidates = []
        for player in wr_stats['player_name'].unique():
            player_data = wr_stats[wr_stats['player_name'] == player].sort_values('season')
            if len(player_data) == 2:  # Has both seasons
                improvement = player_data.iloc[1]['total_yards'] - player_data.iloc[0]['total_yards']
                if improvement > 500:  # 500+ yard improvement
                    breakout_candidates.append({
                        'player': player,
                        'improvement': improvement,
                        'yards_2023': player_data.iloc[0]['total_yards'],
                        'yards_2024': player_data.iloc[1]['total_yards']
                    })
        
        if breakout_candidates:
            print(f"✓ Found {len(breakout_candidates)} WR breakout candidates (500+ yard improvement)")
            print("\nTop breakout players:")
            for candidate in sorted(breakout_candidates, key=lambda x: x['improvement'], reverse=True)[:3]:
                print(f"  {candidate['player']}: {candidate['yards_2023']} → {candidate['yards_2024']} yards (+{candidate['improvement']})")
        else:
            print("✓ No major breakout candidates found in this sample")
    
    print(f"\n\n" + "="*60)
    print("EXAMPLE COMPLETE")
    print("="*60)
    print("Files saved to examples/ directory")
    print("Use 'ml4ff-data --help' for command line options")
    print("\nFor full 20-year analysis, run:")
    print("  ml4ff-data --start-year 2005 --end-year 2025")
    print("="*60)


if __name__ == "__main__":
    main()