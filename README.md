# ML4FF

This repository explores machine learning approaches for fantasy football analytics.

## Rookie Projection Pipeline

The `rookie_projection_pipeline.py` script builds statistical comparisons for incoming NFL rookies.
It automatically downloads real historical and preseason data from the [`nflverse` project](https://github.com/nflverse/nflverse-data), combines college production (from the draft dataset), combine testing results, historical rookie transition rates by position, and preseason snap usage to create an initial boom/bust tiering.

Running the script produces a `rookie_boom_bust.csv` file containing the projections.

### Usage
```
python rookie_projection_pipeline.py
```

## Player Dropoff Prediction Pipeline

The `player_dropoff_pipeline.py` script predicts which fantasy football players are at risk of significant performance declines in the upcoming season. It uses extensive historical player performance data from nflverse (2005-2024) to identify patterns that lead to fantasy point dropoffs.

The pipeline analyzes 20 seasons of data including:
- Fantasy points and performance metrics
- Player age and experience  
- Team changes
- Workload and usage patterns
- Position-specific factors

**NEW: Historical Backtesting** - The pipeline now includes comprehensive backtesting across 15 seasons (2010-2024) to validate prediction accuracy. Results show:
- Overall AUC: 0.878 (excellent predictive performance)
- High-risk predictions: 87.9% actual dropoff rate
- 3,389 historical predictions validated

Running the script produces predictions with dropoff probabilities, risk tiers, and detailed backtest analysis.

### Usage
```
python player_dropoff_pipeline.py
```

### Features
- **Extended Historical Data**: Uses 20 seasons (2005-2024) for maximum data utilization
- **Proper Backtesting**: Time-series validation across 15 years with 3,389 predictions
- **Dropoff Definition**: Configurable threshold for performance decline (default: 20%)
- **Risk Tiers**: Players categorized as Low, Medium, or High risk with validated accuracy
- **Feature Importance**: Identifies which factors most predict dropoffs
- **Position Analysis**: Position-specific dropoff patterns and validation
- **Model Validation**: Comprehensive backtest results showing 87.9% accuracy for high-risk predictions
- **Detailed Analysis**: Multiple output files for deep dive analysis

### Output Files
- `player_dropoff_predictions_2025.csv` - Main predictions for 2025 season
- `backtest_summary.csv` - Year-by-year validation performance
- `detailed_backtest_predictions.csv` - Complete historical validation dataset
- `high_risk_players_2025.csv` - Focused high-risk player list
- `high_confidence_correct_predictions.csv` - Historical high-confidence successes
- `yearly_position_summary.csv` - Position-specific analysis by year

### Dependencies
Both pipelines require `numpy`, `pandas`, and `scikit-learn`:
```
pip install numpy pandas scikit-learn
```

## Notebooks

The `notebooks/` directory contains Jupyter notebooks demonstrating usage of both pipelines:
- `rookie_pipeline_usage.ipynb` - Demonstrates the rookie projection pipeline
- `player_dropoff_pipeline_usage.ipynb` - Demonstrates the player dropoff prediction pipeline

