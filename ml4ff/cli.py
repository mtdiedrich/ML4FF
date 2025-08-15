"""Command Line Interface for ML4FF pipelines."""

import argparse
import sys
from typing import List

from .data_acquisition import DataAcquisitionPipeline


def data_acquisition_cli():
    """CLI for data acquisition pipeline."""
    parser = argparse.ArgumentParser(description="NFL Data Acquisition Pipeline")
    parser.add_argument("--start-year", type=int, default=2005, 
                       help="Start year for data collection (default: 2005)")
    parser.add_argument("--end-year", type=int, default=2025,
                       help="End year for data collection (default: 2025)")
    parser.add_argument("--output-dir", type=str, default="data",
                       help="Output directory for CSV files (default: data)")
    parser.add_argument("--positions", nargs="+", 
                       help="Specific positions to include (e.g., RB WR TE)")
    parser.add_argument("--no-save", action="store_true",
                       help="Don't save CSV files, just return data")
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = DataAcquisitionPipeline(
        seasons=list(range(args.start_year, args.end_year)),
        output_dir=args.output_dir,
        positions=args.positions
    )
    
    # Run pipeline
    data = pipeline.run(save_csv=not args.no_save)
    
    # Print summary
    print(f"\nData acquisition complete!")
    print(f"Total seasons: {len(pipeline.seasons)}")
    if not data['yardage_and_tds'].empty:
        summary = data['yardage_and_tds']
        print(f"Total player-seasons: {len(summary):,}")
        print(f"Unique players: {summary['player_name'].nunique():,}")


def rookie_projection_cli():
    """CLI for rookie projection pipeline."""
    try:
        from .pipelines.rookie_projection import main
        main()
    except ImportError:
        print("Rookie projection pipeline not available")
        sys.exit(1)


def player_breakout_cli():
    """CLI for player breakout pipeline."""
    try:
        from .pipelines.player_breakout import main
        main()
    except ImportError:
        print("Player breakout pipeline not available")
        sys.exit(1)


def player_dropoff_cli():
    """CLI for player dropoff pipeline."""
    try:
        from .pipelines.player_dropoff import main
        main()
    except ImportError:
        print("Player dropoff pipeline not available")
        sys.exit(1)