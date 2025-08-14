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

The `player_dropoff_pipeline.py` script predicts which fantasy football players are at risk of significant performance declines in the upcoming season. It uses historical player performance data from nflverse to identify patterns that lead to fantasy point dropoffs.

The pipeline analyzes multiple seasons of data including:
- Fantasy points and performance metrics
- Player age and experience
- Team changes
- Workload and usage patterns
- Position-specific factors

Running the script produces predictions with dropoff probabilities and risk tiers.

### Usage
```
python player_dropoff_pipeline.py
```

### Features
- **Dropoff Definition**: Configurable threshold for performance decline (default: 20%)
- **Risk Tiers**: Players categorized as Low, Medium, or High risk
- **Feature Importance**: Identifies which factors most predict dropoffs
- **Position Analysis**: Position-specific dropoff patterns
- **Model Validation**: Includes train/test splits and performance metrics

### Dependencies
Both pipelines require `numpy`, `pandas`, and `scikit-learn`:
```
pip install numpy pandas scikit-learn
```

## Notebooks

The `notebooks/` directory contains Jupyter notebooks demonstrating usage of both pipelines:
- `rookie_pipeline_usage.ipynb` - Demonstrates the rookie projection pipeline
- `player_dropoff_pipeline_usage.ipynb` - Demonstrates the player dropoff prediction pipeline

