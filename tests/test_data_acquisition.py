"""Tests for data acquisition pipeline."""

import pytest
import pandas as pd
from ml4ff.data_acquisition import DataAcquisitionPipeline


class TestDataAcquisitionPipeline:
    """Test suite for DataAcquisitionPipeline."""
    
    def test_pipeline_initialization(self):
        """Test pipeline can be initialized with default parameters."""
        pipeline = DataAcquisitionPipeline()
        assert len(pipeline.seasons) == 20  # 2005-2024
        assert pipeline.output_dir == "data"
        assert pipeline.season_types == ["REG"]
        assert pipeline.positions is None
    
    def test_pipeline_custom_seasons(self):
        """Test pipeline with custom season range."""
        pipeline = DataAcquisitionPipeline(seasons=[2022, 2023, 2024])
        assert pipeline.seasons == [2022, 2023, 2024]
    
    def test_pipeline_custom_positions(self):
        """Test pipeline with specific positions."""
        pipeline = DataAcquisitionPipeline(positions=["RB", "WR", "TE"])
        assert pipeline.positions == ["RB", "WR", "TE"]
    
    def test_extract_yardage_and_tds(self):
        """Test yardage and TD extraction functionality."""
        pipeline = DataAcquisitionPipeline()
        
        # Create sample data
        sample_data = pd.DataFrame({
            'player_name': ['Player A', 'Player A', 'Player B'],
            'season': [2023, 2023, 2023],
            'position': ['RB', 'RB', 'WR'],
            'rushing_yards': [50, 30, 0],
            'rushing_tds': [1, 0, 0],
            'receiving_yards': [20, 15, 80],
            'receiving_tds': [0, 1, 1],
            'recent_team': ['BUF', 'BUF', 'MIA']
        })
        
        result = pipeline.extract_yardage_and_tds(sample_data)
        
        # Check aggregation worked
        assert len(result) == 2  # 2 unique players
        assert 'total_yards' in result.columns
        assert 'total_tds' in result.columns
        
        # Check Player A aggregation (80 rush + 35 rec = 115 total yards)
        player_a = result[result['player_name'] == 'Player A'].iloc[0]
        assert player_a['rushing_yards'] == 80
        assert player_a['receiving_yards'] == 35
        assert player_a['total_yards'] == 115
        assert player_a['total_tds'] == 2


def test_pipeline_basic_functionality():
    """Test basic pipeline functionality with minimal data."""
    # This is a simple smoke test that doesn't require internet
    pipeline = DataAcquisitionPipeline(seasons=[])  # Empty seasons to avoid downloads
    
    # Test with empty dataframe
    empty_df = pd.DataFrame()
    result = pipeline.extract_yardage_and_tds(empty_df)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0
    
    # Test with properly structured but empty dataframe  
    empty_structured_df = pd.DataFrame(columns=['player_name', 'season', 'position'])
    result = pipeline.extract_yardage_and_tds(empty_structured_df)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0