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

### Dependencies
The pipeline requires `numpy`, `pandas`, and `scikit-learn`:
```
pip install numpy pandas scikit-learn
```

