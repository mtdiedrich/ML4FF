# Installation and Setup

## Requirements

- Python 3.8+
- Internet connection (for downloading nflverse data)

## Dependencies

Install required packages:

```bash
pip install numpy pandas scikit-learn matplotlib
```

## Quick Test

Verify everything works:

```bash
# Test basic functionality
python example_breakout.py

# Test both pipelines
python pipeline_comparison_example.py

# Run full breakout analysis
cd src && python player_breakout_pipeline.py

# Run full dropoff analysis  
cd src && python player_dropoff_pipeline.py
```

## Usage Examples

### Jupyter Notebooks

- `notebooks/player_breakout_pipeline_usage.ipynb` - Comprehensive breakout analysis
- `notebooks/player_dropoff_pipeline_usage.ipynb` - Comprehensive dropoff analysis

### Python Scripts

```python
# Breakout predictions
from src.player_breakout_pipeline import PlayerBreakoutPipeline

pipeline = PlayerBreakoutPipeline(
    seasons=list(range(2018, 2025)),
    breakout_threshold=0.3  # 30% increase
)
predictions, results = pipeline.run(predict_season=2025, save_csv=True)
```

```python  
# Dropoff predictions
from src.player_dropoff_pipeline import PlayerDropoffPipeline

pipeline = PlayerDropoffPipeline(
    seasons=list(range(2018, 2025)), 
    dropoff_threshold=0.2  # 20% decline
)
predictions, results = pipeline.run(predict_season=2025, save_csv=True)
```

## Output Files

- `player_breakout_predictions_2025.csv` - All breakout predictions
- `high_potential_players_2025.csv` - High-potential players only
- `player_dropoff_predictions_2025.csv` - All dropoff predictions  
- `high_risk_players_2025.csv` - High-risk players only
- `breakout_backtest_summary.csv` - Historical performance summary
- `backtest_summary.csv` - Dropoff historical performance summary