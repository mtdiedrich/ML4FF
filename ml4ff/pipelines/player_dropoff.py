"""Pipeline for predicting fantasy football player dropoffs using historical nflverse data."""
import ssl
from urllib.request import urlopen
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, average_precision_score
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# URLs for nflverse data
PLAYER_STATS_URL = "https://github.com/nflverse/nflverse-data/releases/download/player_stats/player_stats.csv"
ROSTER_URL = "https://github.com/nflverse/nflverse-data/releases/download/rosters/roster_{season}.csv"
INJURY_URL = "https://github.com/nflverse/nflverse-data/releases/download/injuries/injuries_{season}.csv"

def load_player_data(seasons: List[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load player stats and roster data for multiple seasons."""
    ctx = ssl._create_unverified_context()
    
    # Load player stats
    player_stats = pd.read_csv(urlopen(PLAYER_STATS_URL, context=ctx))
    
    # Filter to regular season only and relevant seasons
    player_stats = player_stats[
        (player_stats['season_type'] == 'REG') & 
        (player_stats['season'].isin(seasons))
    ].copy()
    
    # Load roster data for team changes and experience
    rosters = []
    for season in seasons:
        try:
            roster_url = ROSTER_URL.format(season=season)
            roster = pd.read_csv(urlopen(roster_url, context=ctx))
            roster['season'] = season
            rosters.append(roster)
        except Exception:
            continue
    
    roster_data = pd.concat(rosters, ignore_index=True) if rosters else pd.DataFrame()
    
    return player_stats, roster_data

def aggregate_season_stats(player_stats: pd.DataFrame) -> pd.DataFrame:
    """Aggregate weekly stats to season-level for each player."""
    # Filter out rows with null player names
    player_stats = player_stats[player_stats['player_name'].notna()].copy()
    
    # Group by player and season, sum relevant fantasy stats
    agg_dict = {
        'fantasy_points': 'sum',
        'fantasy_points_ppr': 'sum',
        'carries': 'sum',
        'rushing_yards': 'sum',
        'rushing_tds': 'sum',
        'targets': 'sum',
        'receptions': 'sum',
        'receiving_yards': 'sum',
        'receiving_tds': 'sum',
        'passing_yards': 'sum',
        'passing_tds': 'sum',
        'interceptions': 'sum',
        'week': 'count',  # Count of weeks = games played
        'position': 'first',
        'recent_team': 'first'
    }
    
    season_stats = player_stats.groupby(['player_name', 'season']).agg(agg_dict).reset_index()
    
    # Rename week count to games_played
    season_stats.rename(columns={'week': 'games_played'}, inplace=True)
    
    # Filter to players with meaningful playing time (lowered threshold to include more data)
    season_stats = season_stats[season_stats['games_played'] >= 2].copy()
    
    return season_stats

def calculate_dropoffs(season_stats: pd.DataFrame, dropoff_threshold: float = 0.2) -> pd.DataFrame:
    """Calculate year-over-year dropoffs in fantasy performance."""
    # Sort by player and season
    season_stats = season_stats.sort_values(['player_name', 'season'])
    
    # Calculate previous season stats
    season_stats['prev_fantasy_points'] = season_stats.groupby('player_name')['fantasy_points'].shift(1)
    season_stats['prev_fantasy_points_ppr'] = season_stats.groupby('player_name')['fantasy_points_ppr'].shift(1)
    season_stats['prev_team'] = season_stats.groupby('player_name')['recent_team'].shift(1)
    season_stats['prev_games'] = season_stats.groupby('player_name')['games_played'].shift(1)
    
    # Lowered threshold to include more players in analysis (was 30, now 20)
    substantial_players = season_stats['prev_fantasy_points'] >= 20
    
    # Calculate percentage change in fantasy points
    season_stats['fantasy_change'] = (
        (season_stats['fantasy_points'] - season_stats['prev_fantasy_points']) / 
        season_stats['prev_fantasy_points']
    )
    season_stats['fantasy_ppr_change'] = (
        (season_stats['fantasy_points_ppr'] - season_stats['prev_fantasy_points_ppr']) / 
        season_stats['prev_fantasy_points_ppr']
    )
    
    # Define dropoff (binary target) - only for substantial players
    season_stats['dropoff'] = 0
    season_stats.loc[substantial_players, 'dropoff'] = (
        season_stats.loc[substantial_players, 'fantasy_change'] < -dropoff_threshold
    ).astype(int)
    
    # Team change indicator
    season_stats['team_change'] = (season_stats['recent_team'] != season_stats['prev_team']).astype(int)
    
    # Remove first season for each player and players without substantial previous production
    season_stats = season_stats.dropna(subset=['prev_fantasy_points']).copy()
    season_stats = season_stats[substantial_players].copy()
    
    return season_stats

def engineer_features(season_stats: pd.DataFrame, roster_data: pd.DataFrame) -> pd.DataFrame:
    """Engineer features for dropoff prediction."""
    # Merge with roster data to get age/experience info
    roster_subset = roster_data[['full_name', 'season', 'years_exp', 'birth_date']].copy()
    roster_subset.rename(columns={'full_name': 'player_name'}, inplace=True)
    
    merged = season_stats.merge(roster_subset, on=['player_name', 'season'], how='left')
    
    # Calculate age if birth_date is available
    merged['birth_date'] = pd.to_datetime(merged['birth_date'], errors='coerce')
    merged['age'] = merged.apply(
        lambda row: (pd.Timestamp(f'{row["season"]}-09-01') - row['birth_date']).days / 365.25
        if pd.notna(row['birth_date']) else np.nan, axis=1
    )
    
    # Previous season workload features
    merged['prev_workload'] = merged['prev_fantasy_points'] / merged.groupby('player_name')['prev_fantasy_points'].transform('max')
    
    # Position-specific features
    position_dummies = pd.get_dummies(merged['position'], prefix='pos')
    merged = pd.concat([merged, position_dummies], axis=1)
    
    # Fill missing values
    merged['years_exp'] = merged['years_exp'].fillna(merged['years_exp'].median())
    merged['age'] = merged['age'].fillna(merged['age'].median())
    
    return merged

def select_features(df: pd.DataFrame) -> List[str]:
    """Select features for the model."""
    feature_columns = [
        'prev_fantasy_points', 'prev_fantasy_points_ppr', 'years_exp', 'age',
        'team_change', 'prev_workload', 'games_played'
    ]
    
    # Add position dummies
    pos_columns = [col for col in df.columns if col.startswith('pos_')]
    feature_columns.extend(pos_columns)
    
    # Only return features that exist in the dataframe
    return [col for col in feature_columns if col in df.columns]

@dataclass
class PlayerDropoffPipeline:
    """Pipeline for predicting fantasy football player dropoffs.
    
    Parameters
    ----------
    seasons: List[int]
        List of seasons to include in the analysis.
    dropoff_threshold: float, default 0.2
        Threshold for defining a dropoff (e.g., 0.2 = 20% decline).
    test_size: float, default 0.2
        Proportion of data to use for testing (only used for non-backtest training).
    min_training_seasons: int, default 5
        Minimum number of seasons to use for training before starting predictions.
    """
    
    seasons: List[int]
    dropoff_threshold: float = 0.2
    test_size: float = 0.2
    min_training_seasons: int = 5
    
    def __post_init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100, 
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.backtest_results = []
    
    def load_and_prepare_data(self) -> pd.DataFrame:
        """Load and prepare all data for modeling."""
        print("Loading player data...")
        player_stats, roster_data = load_player_data(self.seasons)
        
        print("Aggregating season stats...")
        season_stats = aggregate_season_stats(player_stats)
        
        print("Calculating dropoffs...")
        dropoff_data = calculate_dropoffs(season_stats, self.dropoff_threshold)
        
        print("Engineering features...")
        feature_data = engineer_features(dropoff_data, roster_data)
        
        return feature_data
    
    def train_model(self, data: pd.DataFrame) -> Dict:
        """Train the dropoff prediction model."""
        # Select features
        self.feature_columns = select_features(data)
        X = data[self.feature_columns].fillna(0)
        y = data['dropoff']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        print("Training model...")
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_pred = self.model.predict(X_train_scaled)
        test_pred = self.model.predict(X_test_scaled)
        
        train_pred_proba = self.model.predict_proba(X_train_scaled)[:, 1]
        test_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        results = {
            'train_auc': roc_auc_score(y_train, train_pred_proba),
            'test_auc': roc_auc_score(y_test, test_pred_proba),
            'train_report': classification_report(y_train, train_pred),
            'test_report': classification_report(y_test, test_pred),
            'feature_importance': dict(zip(self.feature_columns, self.model.feature_importances_))
        }
        
        return results
    
    def predict_current_season(self, data: pd.DataFrame, current_season: int) -> pd.DataFrame:
        """Predict dropoffs for the current season."""
        # Get most recent season data for each player
        current_data = data[data['season'] == current_season - 1].copy()
        
        if current_data.empty:
            print(f"No data available for season {current_season - 1}")
            return pd.DataFrame()
        
        # Prepare features
        X = current_data[self.feature_columns].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        dropoff_proba = self.model.predict_proba(X_scaled)[:, 1]
        dropoff_pred = self.model.predict(X_scaled)
        
        # Create results dataframe
        predictions = current_data[['player_name', 'position', 'recent_team', 'fantasy_points']].copy()
        predictions['dropoff_probability'] = dropoff_proba
        predictions['predicted_dropoff'] = dropoff_pred
        predictions['risk_tier'] = pd.cut(
            dropoff_proba, 
            bins=[0, 0.3, 0.6, 1.0], 
            labels=['Low Risk', 'Medium Risk', 'High Risk']
        )
        
        return predictions.sort_values('dropoff_probability', ascending=False)
    
    def run_backtest(self, data: pd.DataFrame) -> List[Dict]:
        """Run historical backtest using time-series validation.
        
        For each season after min_training_seasons, train on all prior data
        and predict dropoffs for that season.
        """
        print("Running historical backtest...")
        backtest_results = []
        
        # Sort seasons to ensure proper time ordering
        test_seasons = sorted([s for s in self.seasons if s >= min(self.seasons) + self.min_training_seasons])
        
        for test_season in test_seasons:
            print(f"Backtesting season {test_season}...")
            
            # Split data: train on all seasons before test_season
            train_data = data[data['season'] < test_season].copy()
            test_data = data[data['season'] == test_season].copy()
            
            if len(train_data) == 0 or len(test_data) == 0:
                print(f"Insufficient data for season {test_season}")
                continue
            
            # Prepare features
            self.feature_columns = select_features(train_data)
            X_train = train_data[self.feature_columns].fillna(0)
            y_train = train_data['dropoff']
            X_test = test_data[self.feature_columns].fillna(0)
            y_test = test_data['dropoff']
            
            if len(X_train) == 0 or len(X_test) == 0 or y_train.sum() == 0:
                print(f"Insufficient training/test data for season {test_season}")
                continue
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model = RandomForestClassifier(
                n_estimators=100, 
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42
            )
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            test_pred = model.predict(X_test_scaled)
            test_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            # Calculate metrics
            test_auc = roc_auc_score(y_test, test_pred_proba) if len(np.unique(y_test)) > 1 else np.nan
            test_ap = average_precision_score(y_test, test_pred_proba) if len(np.unique(y_test)) > 1 else np.nan
            
            # Feature importance
            feature_importance = dict(zip(self.feature_columns, model.feature_importances_))
            
            # Store detailed results
            season_result = {
                'test_season': test_season,
                'n_train': len(X_train),
                'n_test': len(X_test),
                'n_dropoffs_train': y_train.sum(),
                'n_dropoffs_test': y_test.sum(),
                'test_auc': test_auc,
                'test_ap': test_ap,
                'feature_importance': feature_importance,
                'predictions': test_data[['player_name', 'position', 'recent_team', 'fantasy_points']].copy()
            }
            
            # Add predictions to the dataframe
            season_result['predictions']['actual_dropoff'] = y_test.values
            season_result['predictions']['predicted_dropoff'] = test_pred
            season_result['predictions']['dropoff_probability'] = test_pred_proba
            
            backtest_results.append(season_result)
        
        self.backtest_results = backtest_results
        return backtest_results
    
    def get_backtest_summary(self) -> pd.DataFrame:
        """Summarize backtest results across all seasons."""
        if not self.backtest_results:
            return pd.DataFrame()
        
        summary_data = []
        for result in self.backtest_results:
            summary_data.append({
                'season': result['test_season'],
                'n_train': result['n_train'],
                'n_test': result['n_test'],
                'n_dropoffs_train': result['n_dropoffs_train'],
                'n_dropoffs_test': result['n_dropoffs_test'],
                'dropoff_rate_train': result['n_dropoffs_train'] / result['n_train'] if result['n_train'] > 0 else 0,
                'dropoff_rate_test': result['n_dropoffs_test'] / result['n_test'] if result['n_test'] > 0 else 0,
                'test_auc': result['test_auc'],
                'test_ap': result['test_ap']
            })
        
        return pd.DataFrame(summary_data)
    
    def analyze_backtest_performance(self) -> Dict:
        """Analyze overall backtest performance."""
        if not self.backtest_results:
            return {}
        
        summary = self.get_backtest_summary()
        
        # Calculate aggregate metrics
        all_predictions = []
        all_actuals = []
        
        for result in self.backtest_results:
            if not np.isnan(result['test_auc']):
                all_predictions.extend(result['predictions']['dropoff_probability'].tolist())
                all_actuals.extend(result['predictions']['actual_dropoff'].tolist())
        
        if len(all_predictions) > 0:
            overall_auc = roc_auc_score(all_actuals, all_predictions) if len(np.unique(all_actuals)) > 1 else np.nan
            overall_ap = average_precision_score(all_actuals, all_predictions) if len(np.unique(all_actuals)) > 1 else np.nan
        else:
            overall_auc = np.nan
            overall_ap = np.nan
        
        analysis = {
            'n_seasons_tested': len(self.backtest_results),
            'avg_test_auc': summary['test_auc'].mean(),
            'std_test_auc': summary['test_auc'].std(),
            'overall_auc': overall_auc,
            'overall_ap': overall_ap,
            'avg_dropoff_rate': summary['dropoff_rate_test'].mean(),
            'total_predictions': len(all_predictions),
            'total_actual_dropoffs': sum(all_actuals),
            'summary_by_season': summary
        }
        
        return analysis
    def run(self, predict_season: int = None, save_csv: bool = False, run_backtest: bool = True) -> Tuple[pd.DataFrame, Dict]:
        """Execute the full pipeline.
        
        Parameters
        ----------
        predict_season: int, optional
            Season to make predictions for. If None, uses the latest season + 1.
        save_csv: bool, default False
            Whether to save predictions to CSV.
        run_backtest: bool, default True
            Whether to run historical backtesting.
        
        Returns
        -------
        predictions: pd.DataFrame
            Dropoff predictions for the target season.
        results: Dict
            Model training, evaluation, and backtest results.
        """
        # Load and prepare data
        data = self.load_and_prepare_data()
        
        results = {}
        
        # Run backtest if requested
        if run_backtest:
            backtest_results = self.run_backtest(data)
            backtest_analysis = self.analyze_backtest_performance()
            results['backtest_results'] = backtest_results
            results['backtest_analysis'] = backtest_analysis
        
        # Train model on all available data for final predictions
        model_results = self.train_model(data)
        results.update(model_results)
        
        # Make predictions
        if predict_season is None:
            predict_season = max(self.seasons) + 1
        
        predictions = self.predict_current_season(data, predict_season)
        
        if save_csv and not predictions.empty:
            filename = f"player_dropoff_predictions_{predict_season}.csv"
            predictions.to_csv(filename, index=False)
            print(f"Predictions saved to {filename}")
        
        return predictions, results

def main() -> None:
    # Use extensive historical data - from 2005 to 2024 (20 seasons)
    # Starting from 2005 as fantasy football became more standardized around then
    seasons = list(range(2005, 2025))
    
    print(f"Analyzing {len(seasons)} seasons of data: {min(seasons)}-{max(seasons)}")
    
    pipeline = PlayerDropoffPipeline(
        seasons=seasons,
        dropoff_threshold=0.2,
        min_training_seasons=5
    )
    predictions, results = pipeline.run(predict_season=2025, save_csv=True, run_backtest=True)
    
    # Display backtest results
    if 'backtest_analysis' in results:
        analysis = results['backtest_analysis']
        print("\n" + "="*60)
        print("HISTORICAL BACKTEST RESULTS")
        print("="*60)
        print(f"Seasons tested: {analysis['n_seasons_tested']}")
        print(f"Total predictions made: {analysis['total_predictions']}")
        print(f"Total actual dropoffs: {analysis['total_actual_dropoffs']}")
        print(f"Average dropoff rate: {analysis['avg_dropoff_rate']:.1%}")
        print(f"Overall AUC: {analysis['overall_auc']:.3f}")
        print(f"Overall Average Precision: {analysis['overall_ap']:.3f}")
        print(f"Average Test AUC: {analysis['avg_test_auc']:.3f} ± {analysis['std_test_auc']:.3f}")
        
        print("\nYear-by-Year Backtest Performance:")
        summary = analysis['summary_by_season']
        print(summary[['season', 'n_test', 'dropoff_rate_test', 'test_auc', 'test_ap']].to_string(index=False, float_format='%.3f'))
        
        # Save backtest summary
        summary.to_csv('backtest_summary.csv', index=False)
        print("\nBacktest summary saved to 'backtest_summary.csv'")
    
    # Display current model performance
    print("\n" + "="*60)
    print("CURRENT MODEL PERFORMANCE (All Data)")
    print("="*60)
    print(f"Train AUC: {results['train_auc']:.3f}")
    print(f"Test AUC: {results['test_auc']:.3f}")
    
    print("\nTop Feature Importances:")
    sorted_features = sorted(results['feature_importance'].items(), key=lambda x: x[1], reverse=True)
    for feature, importance in sorted_features[:10]:
        print(f"{feature}: {importance:.3f}")
    
    # Display 2025 predictions
    if not predictions.empty:
        print(f"\n" + "="*60)
        print("TOP PREDICTED DROPOFFS FOR 2025")
        print("="*60)
        print(predictions[['player_name', 'position', 'recent_team', 'fantasy_points', 'dropoff_probability', 'risk_tier']].head(15).to_string(index=False))
        
        print(f"\nRisk Tier Summary:")
        print(predictions['risk_tier'].value_counts())
        
        # Save high-risk players separately
        high_risk = predictions[predictions['risk_tier'] == 'High Risk']
        if not high_risk.empty:
            high_risk.to_csv('high_risk_players_2025.csv', index=False)
            print(f"\n{len(high_risk)} high-risk players saved to 'high_risk_players_2025.csv'")

if __name__ == "__main__":
    main()