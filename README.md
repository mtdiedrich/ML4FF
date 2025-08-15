# ML4FF - Machine Learning for Fantasy Football

A comprehensive machine learning toolkit for fantasy football analysis and predictions, featuring a powerful data acquisition pipeline for 20 years of NFL player statistics.

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/mtdiedrich/ML4FF.git
cd ML4FF

# Install the package
pip install -e .

# Or install requirements manually
pip install -r requirements.txt
```

### Basic Usage

```bash
# Download 20 years of NFL data (2005-2024)
ml4ff-data

# Download specific years and positions
ml4ff-data --start-year 2020 --end-year 2025 --positions RB WR TE

# Get help
ml4ff-data --help
```

### Python API

```python
from ml4ff import DataAcquisitionPipeline

# Download comprehensive NFL data
pipeline = DataAcquisitionPipeline()
data = pipeline.run(save_csv=True)

# Focus on specific positions and years
rb_pipeline = DataAcquisitionPipeline(
    seasons=[2022, 2023, 2024],
    positions=['RB']
)
rb_data = rb_pipeline.run()

# Analyze yardage and touchdown statistics
stats = rb_data['yardage_and_tds']
print(f"Total yards: {stats['total_yards'].sum():,}")
print(f"Total TDs: {stats['total_tds'].sum():,}")
```

## 📊 Data Acquisition Pipeline

The core feature of ML4FF is its comprehensive NFL data acquisition pipeline that downloads and processes:

- **Player Statistics**: 20 years (2005-2024) of weekly player performance data
- **Yardage & Touchdowns**: Detailed rushing, receiving, and passing statistics
- **Roster Data**: Player information, teams, age, experience
- **Draft Data**: NFL draft picks and college statistics
- **Combine Data**: Physical measurements and test results
- **Snap Counts**: Player usage and participation data

### Key Features

- ✅ **20 Years of Data**: Complete NFL statistics from 2005-2024
- ✅ **100,000+ Records**: Comprehensive player statistics database
- ✅ **1,600+ Players**: Historical data for all NFL players
- ✅ **Position Filtering**: Focus on specific positions (RB, WR, TE, QB, etc.)
- ✅ **Flexible Export**: Save to CSV files or use in-memory
- ✅ **Command Line Interface**: Easy-to-use CLI tools
- ✅ **Python API**: Full programmatic access

### Data Summary

When running the full 20-year acquisition:
- **100,763** player statistic records
- **1,693** unique players
- **2.35M** total yards across all players and seasons
- **15,848** total touchdowns
- **7,100** player-season combinations

## 🔧 Advanced Usage

### Command Line Examples

```bash
# Download all data for the last 5 years
ml4ff-data --start-year 2020 --end-year 2025

# Focus on skill positions only
ml4ff-data --positions RB WR TE --output-dir skill_players

# Quick analysis without saving files
ml4ff-data --start-year 2023 --end-year 2025 --no-save
```

### Python Examples

```python
from ml4ff import DataAcquisitionPipeline

# Example 1: Running Back Analysis
rb_pipeline = DataAcquisitionPipeline(
    seasons=list(range(2020, 2025)),
    positions=['RB'],
    output_dir='rb_analysis'
)
rb_data = rb_pipeline.run()

# Find top performers
rb_stats = rb_data['yardage_and_tds']
top_rbs = rb_stats.nlargest(10, 'total_yards')
print(top_rbs[['player_name', 'season', 'total_yards', 'total_tds']])

# Example 2: Wide Receiver Breakout Analysis
wr_pipeline = DataAcquisitionPipeline(
    seasons=[2023, 2024],
    positions=['WR']
)
wr_data = wr_pipeline.run(save_csv=False)

# Find players with significant year-over-year improvement
wr_stats = wr_data['yardage_and_tds']
for player in wr_stats['player_name'].unique():
    player_data = wr_stats[wr_stats['player_name'] == player]
    if len(player_data) == 2:
        improvement = player_data.iloc[1]['total_yards'] - player_data.iloc[0]['total_yards']
        if improvement > 500:
            print(f"{player}: +{improvement} yards improvement")
```

## 🏈 ML Pipelines

ML4FF includes three sophisticated machine learning pipelines for fantasy football analysis:

### 1. Player Dropoff Pipeline
Predicts which players are at risk of significant performance declines.

```python
from ml4ff import PlayerDropoffPipeline

pipeline = PlayerDropoffPipeline(
    seasons=list(range(2018, 2025)),
    dropoff_threshold=0.2  # 20% decline
)
predictions, results = pipeline.run(predict_season=2025, save_csv=True)
```

### 2. Player Breakout Pipeline  
Predicts which players are likely to have breakout seasons.

```python
from ml4ff import PlayerBreakoutPipeline

pipeline = PlayerBreakoutPipeline(
    seasons=list(range(2018, 2025)),
    breakout_threshold=0.3  # 30% increase
)
predictions, results = pipeline.run(predict_season=2025, save_csv=True)
```

### 3. Rookie Projection Pipeline
Creates statistical comparisons and boom/bust tiers for incoming NFL rookies.

```python
from ml4ff import RookieProjectionPipeline

pipeline = RookieProjectionPipeline(season=2024)
projections = pipeline.run(save_csv=True)
```

## 📁 Project Structure

```
ML4FF/
├── ml4ff/                      # Main package
│   ├── __init__.py            # Package initialization
│   ├── data_acquisition.py    # Core data pipeline
│   ├── cli.py                 # Command line interfaces
│   └── pipelines/             # ML prediction pipelines
│       ├── __init__.py
│       ├── rookie_projection.py
│       ├── player_breakout.py
│       └── player_dropoff.py
├── tests/                     # Test suite
├── notebooks/                 # Jupyter examples
├── examples/                  # Usage examples
├── pyproject.toml            # Modern Python packaging
├── requirements.txt          # Dependencies
└── README.md                 # This file
```

## 🧪 Testing

```bash
# Install test dependencies
pip install pytest

# Run tests
pytest

# Run specific test file
pytest tests/test_data_acquisition.py -v
```

## 📈 Example Analysis

Run the comprehensive example to see ML4FF in action:

```bash
python example_usage.py
```

This will demonstrate:
- Running back analysis across multiple seasons
- Historical data trends
- Breakout player identification
- Position-specific insights

## 🛠️ Technical Requirements

- **Python**: 3.8+
- **Core Dependencies**: pandas, numpy, scikit-learn, matplotlib
- **Data Source**: [NFLverse](https://github.com/nflverse/nflverse-data)
- **Internet Connection**: Required for data downloads

## 📊 Data Output

The pipeline generates comprehensive CSV files:

- `player_stats.csv` - Raw weekly player statistics
- `yardage_and_tds.csv` - Summarized yardage and touchdown data
- `roster_data.csv` - Player roster information
- `draft_data.csv` - NFL draft data
- `combine_data.csv` - NFL combine results
- `snap_counts.csv` - Player snap count data

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📜 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- Data provided by [NFLverse](https://github.com/nflverse/nflverse-data)
- Built with Python scientific computing ecosystem
- Inspired by the fantasy football analytics community

