# ML4FF - Machine Learning for Fantasy Football

A comprehensive machine learning toolkit for fantasy football analysis and predictions.

## Overview

ML4FF provides two main prediction pipelines:

1. **Player Dropoff Pipeline** - Predicts which players are likely to experience significant fantasy point declines
2. **Player Breakout Pipeline** - Predicts which players are likely to have breakout seasons with significant fantasy point increases

Both pipelines use historical NFL player data from nflverse to train machine learning models for fantasy football predictions.

## Features

- Historical backtesting with time-series validation
- Comprehensive feature engineering using player stats, age, experience, and team changes
- Risk/potential tier classifications
- Position-specific analysis
- CSV export for further analysis
- Jupyter notebook examples

## Pipelines

### Player Dropoff Pipeline

Predicts players likely to experience 20%+ decline in fantasy points.

- **Location**: `src/player_dropoff_pipeline.py`
- **Notebook**: `notebooks/player_dropoff_pipeline_usage.ipynb`
- **Output**: Risk tiers (Low/Medium/High Risk)

### Player Breakout Pipeline  

Predicts players likely to experience 30%+ increase in fantasy points.

- **Location**: `src/player_breakout_pipeline.py`
- **Notebook**: `notebooks/player_breakout_pipeline_usage.ipynb`
- **Output**: Potential tiers (Low/Medium/High Potential)

## Quick Start

```python
# Breakout predictions
from src.player_breakout_pipeline import PlayerBreakoutPipeline

pipeline = PlayerBreakoutPipeline(
    seasons=list(range(2018, 2025)),
    breakout_threshold=0.3
)
predictions, results = pipeline.run(predict_season=2025, save_csv=True)

# Dropoff predictions  
from src.player_dropoff_pipeline import PlayerDropoffPipeline

pipeline = PlayerDropoffPipeline(
    seasons=list(range(2018, 2025)),
    dropoff_threshold=0.2
)
predictions, results = pipeline.run(predict_season=2025, save_csv=True)
```

## Dependencies

- pandas
- numpy
- scikit-learn  
- matplotlib

## Data Source

All player data is sourced from [nflverse](https://github.com/nflverse/nflverse-data), which provides comprehensive NFL statistics and roster information.

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

The `notebooks/` directory contains Jupyter notebooks demonstrating usage of the pipelines:
- `rookie_pipeline_usage.ipynb` - Demonstrates the rookie projection pipeline
- `player_dropoff_pipeline_usage.ipynb` - Demonstrates the player dropoff prediction pipeline
- `player_breakout_pipeline_usage.ipynb` - Demonstrates the player breakout prediction pipeline (abstracted)
- `player_breakout_pipeline_procedural.ipynb` - **NEW**: Procedural implementation of the breakout pipeline with discrete steps

### Procedural Breakout Pipeline

The new procedural notebook (`player_breakout_pipeline_procedural.ipynb`) deconstructs the abstraction of the PlayerBreakoutPipeline into discrete functional blocks. Each cell represents a specific step in the process:

1. **Setup & Imports** - Library and function imports
2. **Configuration** - Pipeline parameters
3. **Data Loading** - Raw NFL data from nflverse
4. **Season Aggregation** - Weekly to season-level stats
5. **Breakout Calculation** - Historical breakout identification
6. **Feature Engineering** - Predictive feature creation
7. **Feature Selection** - Relevant variable selection
8. **Data Preparation** - Training/test splits and scaling
9. **Model Training** - Random Forest classifier training
10. **Model Evaluation** - Performance assessment
11. **Feature Analysis** - Understanding important predictors
12. **Predictions Generation** - Breakout forecasts
13. **Analysis & Visualization** - Results interpretation

This approach provides greater transparency and customization compared to the abstracted pipeline, while maintaining identical functionality.

