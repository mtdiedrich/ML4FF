"""Comprehensive NFL Data Acquisition Pipeline for ML4FF.

This module provides a unified interface for downloading and processing
20 years of NFL player statistics, with special focus on yardage and touchdown data.
"""

import ssl
import warnings
from typing import Dict, List, Optional, Tuple
from urllib.request import urlopen
from dataclasses import dataclass, field
from pathlib import Path
import pandas as pd

warnings.filterwarnings('ignore')

# NFLverse data URLs
PLAYER_STATS_URL = "https://github.com/nflverse/nflverse-data/releases/download/player_stats/player_stats.csv"
ROSTER_URL = "https://github.com/nflverse/nflverse-data/releases/download/rosters/roster_{season}.csv"
DRAFT_URL = "https://github.com/nflverse/nflverse-data/releases/download/draft_picks/draft_picks.csv"
COMBINE_URL = "https://github.com/nflverse/nflverse-data/releases/download/combine/combine.csv"
SNAP_COUNTS_URL = "https://github.com/nflverse/nflverse-data/releases/download/snap_counts/snap_counts_{season}.csv"
INJURY_URL = "https://github.com/nflverse/nflverse-data/releases/download/injuries/injuries_{season}.csv"


@dataclass
class DataAcquisitionPipeline:
    """Comprehensive NFL Data Acquisition Pipeline.
    
    Downloads and processes 20 years of NFL player statistics from nflverse,
    with special focus on yardage and touchdown statistics for fantasy football analysis.
    
    Parameters
    ----------
    seasons : List[int], default range(2005, 2025)
        List of NFL seasons to download data for (20 years by default)
    output_dir : str, default "data"
        Directory to save downloaded data
    season_types : List[str], default ["REG"]
        Types of seasons to include (REG=regular season, POST=playoffs, PRE=preseason)
    positions : Optional[List[str]], default None
        Specific positions to filter for. If None, includes all positions.
    """
    
    seasons: List[int] = field(default_factory=lambda: list(range(2005, 2025)))
    output_dir: str = "data"
    season_types: List[str] = field(default_factory=lambda: ["REG"])
    positions: Optional[List[str]] = None
    
    def __post_init__(self):
        """Initialize data acquisition pipeline."""
        self.ctx = ssl._create_unverified_context()
        Path(self.output_dir).mkdir(exist_ok=True)
        
    def download_player_stats(self) -> pd.DataFrame:
        """Download comprehensive player statistics.
        
        Returns
        -------
        pd.DataFrame
            Player statistics with columns including:
            - player_name, season, week, position
            - rushing_yards, rushing_tds, receiving_yards, receiving_tds
            - passing_yards, passing_tds, fantasy_points, etc.
        """
        print("Downloading player statistics...")
        
        try:
            player_stats = pd.read_csv(urlopen(PLAYER_STATS_URL, context=self.ctx))
            
            # Filter by seasons and season types
            player_stats = player_stats[
                (player_stats['season'].isin(self.seasons)) &
                (player_stats['season_type'].isin(self.season_types))
            ].copy()
            
            # Filter by positions if specified
            if self.positions:
                player_stats = player_stats[
                    player_stats['position'].isin(self.positions)
                ].copy()
            
            print(f"Downloaded {len(player_stats):,} player stat records")
            print(f"Seasons: {min(player_stats['season'])}-{max(player_stats['season'])}")
            print(f"Unique players: {player_stats['player_name'].nunique():,}")
            
            return player_stats
            
        except Exception as e:
            print(f"Error downloading player stats: {e}")
            return pd.DataFrame()
    
    def download_roster_data(self) -> pd.DataFrame:
        """Download roster data for all seasons.
        
        Returns
        -------
        pd.DataFrame
            Roster data with player info, team, age, experience, etc.
        """
        print("Downloading roster data...")
        
        rosters = []
        for season in self.seasons:
            try:
                roster_url = ROSTER_URL.format(season=season)
                roster = pd.read_csv(urlopen(roster_url, context=self.ctx))
                roster['season'] = season
                rosters.append(roster)
                print(f"  Downloaded roster for {season}")
            except Exception as e:
                print(f"  Failed to download roster for {season}: {e}")
                continue
        
        if rosters:
            roster_data = pd.concat(rosters, ignore_index=True)
            print(f"Total roster records: {len(roster_data):,}")
            return roster_data
        else:
            print("No roster data downloaded")
            return pd.DataFrame()
    
    def download_draft_data(self) -> pd.DataFrame:
        """Download NFL draft data.
        
        Returns
        -------
        pd.DataFrame
            Draft data with pick info, college stats, etc.
        """
        print("Downloading draft data...")
        
        try:
            draft_data = pd.read_csv(urlopen(DRAFT_URL, context=self.ctx))
            
            # Filter by seasons
            draft_data = draft_data[
                draft_data['season'].isin(self.seasons)
            ].copy()
            
            print(f"Downloaded {len(draft_data):,} draft records")
            return draft_data
            
        except Exception as e:
            print(f"Error downloading draft data: {e}")
            return pd.DataFrame()
    
    def download_combine_data(self) -> pd.DataFrame:
        """Download NFL combine data.
        
        Returns
        -------
        pd.DataFrame
            Combine data with physical measurements and test results.
        """
        print("Downloading combine data...")
        
        try:
            combine_data = pd.read_csv(urlopen(COMBINE_URL, context=self.ctx))
            print(f"Downloaded {len(combine_data):,} combine records")
            return combine_data
            
        except Exception as e:
            print(f"Error downloading combine data: {e}")
            return pd.DataFrame()
    
    def download_snap_counts(self) -> pd.DataFrame:
        """Download snap count data for all seasons.
        
        Returns
        -------
        pd.DataFrame
            Snap count data showing player usage.
        """
        print("Downloading snap count data...")
        
        snap_data = []
        for season in self.seasons:
            try:
                snap_url = SNAP_COUNTS_URL.format(season=season)
                snaps = pd.read_csv(urlopen(snap_url, context=self.ctx))
                snaps['season'] = season
                snap_data.append(snaps)
                print(f"  Downloaded snap counts for {season}")
            except Exception as e:
                print(f"  Failed to download snap counts for {season}: {e}")
                continue
        
        if snap_data:
            snap_counts = pd.concat(snap_data, ignore_index=True)
            print(f"Total snap count records: {len(snap_counts):,}")
            return snap_counts
        else:
            print("No snap count data downloaded")
            return pd.DataFrame()
    
    def extract_yardage_and_tds(self, player_stats: pd.DataFrame) -> pd.DataFrame:
        """Extract and summarize yardage and touchdown statistics.
        
        Parameters
        ----------
        player_stats : pd.DataFrame
            Raw player statistics data
            
        Returns
        -------
        pd.DataFrame
            Summarized yardage and TD data by player and season
        """
        print("Extracting yardage and touchdown statistics...")
        
        # Handle empty dataframe
        if player_stats.empty:
            print("No data to extract")
            return pd.DataFrame()
        
        # Key columns for yardage and TDs
        key_columns = [
            'player_name', 'season', 'position', 'recent_team',
            'rushing_yards', 'rushing_tds',
            'receiving_yards', 'receiving_tds', 
            'passing_yards', 'passing_tds',
            'fantasy_points', 'fantasy_points_ppr'
        ]
        
        # Filter to available columns
        available_columns = [col for col in key_columns if col in player_stats.columns]
        
        # Check if we have minimum required columns
        required_cols = ['player_name', 'season', 'position']
        if not all(col in available_columns for col in required_cols):
            print(f"Missing required columns: {required_cols}")
            return pd.DataFrame()
        
        # Aggregate by player and season
        stats_summary = player_stats[available_columns].groupby(
            ['player_name', 'season', 'position']
        ).agg({
            col: 'sum' for col in available_columns 
            if col not in ['player_name', 'season', 'position', 'recent_team']
        }).reset_index()
        
        # Add team info (most recent team for each player-season)
        if 'recent_team' in available_columns:
            team_info = player_stats.groupby(['player_name', 'season'])['recent_team'].last().reset_index()
            stats_summary = stats_summary.merge(team_info, on=['player_name', 'season'], how='left')
        
        # Calculate total yards and TDs
        if 'rushing_yards' in stats_summary.columns and 'receiving_yards' in stats_summary.columns:
            stats_summary['total_yards'] = (
                stats_summary['rushing_yards'].fillna(0) + 
                stats_summary['receiving_yards'].fillna(0)
            )
        
        if 'rushing_tds' in stats_summary.columns and 'receiving_tds' in stats_summary.columns:
            stats_summary['total_tds'] = (
                stats_summary['rushing_tds'].fillna(0) + 
                stats_summary['receiving_tds'].fillna(0)
            )
        
        print(f"Extracted stats for {stats_summary['player_name'].nunique():,} unique players")
        print(f"Covering {len(stats_summary):,} player-season combinations")
        
        return stats_summary
    
    def save_data(self, data: Dict[str, pd.DataFrame]) -> None:
        """Save all downloaded data to CSV files.
        
        Parameters
        ----------
        data : Dict[str, pd.DataFrame]
            Dictionary of dataframes to save
        """
        print(f"Saving data to {self.output_dir}/...")
        
        for name, df in data.items():
            if not df.empty:
                filename = Path(self.output_dir) / f"{name}.csv"
                df.to_csv(filename, index=False)
                print(f"  Saved {name}: {len(df):,} records -> {filename}")
            else:
                print(f"  Skipped {name}: No data to save")
    
    def run(self, save_csv: bool = True) -> Dict[str, pd.DataFrame]:
        """Run the complete data acquisition pipeline.
        
        Parameters
        ----------
        save_csv : bool, default True
            Whether to save data to CSV files
            
        Returns
        -------
        Dict[str, pd.DataFrame]
            Dictionary containing all downloaded datasets:
            - player_stats: Raw player statistics
            - roster_data: Player roster information
            - draft_data: NFL draft data
            - combine_data: NFL combine data
            - snap_counts: Snap count data
            - yardage_and_tds: Summarized yardage and TD statistics
        """
        print("="*60)
        print("NFL DATA ACQUISITION PIPELINE")
        print("="*60)
        print(f"Seasons: {min(self.seasons)}-{max(self.seasons)} ({len(self.seasons)} years)")
        print(f"Season types: {', '.join(self.season_types)}")
        if self.positions:
            print(f"Positions: {', '.join(self.positions)}")
        print()
        
        # Download all data
        data = {}
        
        # Core player statistics
        data['player_stats'] = self.download_player_stats()
        
        # Supporting data
        data['roster_data'] = self.download_roster_data()
        data['draft_data'] = self.download_draft_data()
        data['combine_data'] = self.download_combine_data()
        data['snap_counts'] = self.download_snap_counts()
        
        # Extract key yardage and TD statistics
        if not data['player_stats'].empty:
            data['yardage_and_tds'] = self.extract_yardage_and_tds(data['player_stats'])
        else:
            data['yardage_and_tds'] = pd.DataFrame()
        
        # Save data if requested
        if save_csv:
            self.save_data(data)
        
        print("\n" + "="*60)
        print("DATA ACQUISITION COMPLETE")
        print("="*60)
        for name, df in data.items():
            print(f"{name}: {len(df):,} records")
        
        return data


def main():
    """Command line interface for data acquisition."""
    import argparse
    
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
        
        if 'total_yards' in summary.columns:
            total_yards = summary['total_yards'].sum()
            print(f"Total yards across all players: {total_yards:,}")
        
        if 'total_tds' in summary.columns:
            total_tds = summary['total_tds'].sum()
            print(f"Total TDs across all players: {total_tds:,}")


if __name__ == "__main__":
    main()