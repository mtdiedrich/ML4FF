"""Detailed analysis and visualization of backtest results."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from player_dropoff_pipeline import PlayerDropoffPipeline


def analyze_backtest_trends(pipeline_results):
    """Analyze trends in backtest performance over time."""
    if not pipeline_results.get('backtest_results'):
        return
    
    backtest_results = pipeline_results['backtest_results']
    
    # Create comprehensive analysis
    print("="*80)
    print("DETAILED BACKTEST ANALYSIS")
    print("="*80)
    
    # 1. Performance stability analysis
    aucs = [r['test_auc'] for r in backtest_results if not np.isnan(r['test_auc'])]
    print(f"\nModel Stability Analysis:")
    print(f"AUC Range: {min(aucs):.3f} - {max(aucs):.3f}")
    print(f"AUC Std Dev: {np.std(aucs):.3f}")
    print(f"Coefficient of Variation: {np.std(aucs)/np.mean(aucs):.3f}")
    
    # 2. Feature importance evolution
    print(f"\nFeature Importance Evolution:")
    feature_importance_by_year = {}
    for result in backtest_results:
        year = result['test_season']
        for feature, importance in result['feature_importance'].items():
            if feature not in feature_importance_by_year:
                feature_importance_by_year[feature] = []
            feature_importance_by_year[feature].append((year, importance))
    
    # Show top 5 features and their stability
    avg_importance = {feature: np.mean([imp for year, imp in years]) 
                     for feature, years in feature_importance_by_year.items()}
    top_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)[:5]
    
    for feature, avg_imp in top_features:
        importances = [imp for year, imp in feature_importance_by_year[feature]]
        print(f"{feature}: avg={avg_imp:.3f}, std={np.std(importances):.3f}")
    
    # 3. Prediction accuracy by risk tier
    print(f"\nPrediction Accuracy by Risk Tier:")
    all_predictions = []
    for result in backtest_results:
        pred_data = result['predictions'].copy()
        pred_data['test_season'] = result['test_season']
        all_predictions.append(pred_data)
    
    all_predictions = pd.concat(all_predictions, ignore_index=True)
    
    # Add risk tiers to historical predictions
    all_predictions['risk_tier'] = pd.cut(
        all_predictions['dropoff_probability'], 
        bins=[0, 0.3, 0.6, 1.0], 
        labels=['Low Risk', 'Medium Risk', 'High Risk']
    )
    
    for tier in ['Low Risk', 'Medium Risk', 'High Risk']:
        tier_data = all_predictions[all_predictions['risk_tier'] == tier]
        if len(tier_data) > 0:
            accuracy = tier_data['actual_dropoff'].mean()
            print(f"{tier}: {len(tier_data)} predictions, {accuracy:.1%} actual dropoff rate")
    
    # 4. Position-specific analysis
    print(f"\nPosition-Specific Dropoff Rates:")
    for position in ['QB', 'RB', 'WR', 'TE']:
        pos_data = all_predictions[all_predictions['position'] == position]
        if len(pos_data) > 0:
            dropoff_rate = pos_data['actual_dropoff'].mean()
            avg_prob = pos_data['dropoff_probability'].mean()
            print(f"{position}: {len(pos_data)} predictions, {dropoff_rate:.1%} actual dropoff, {avg_prob:.3f} avg predicted prob")
    
    return all_predictions


def save_detailed_results(pipeline_results, all_predictions):
    """Save detailed backtest results to CSV files."""
    
    # Save detailed predictions with actual outcomes
    all_predictions.to_csv('detailed_backtest_predictions.csv', index=False)
    print(f"\nDetailed backtest predictions saved to 'detailed_backtest_predictions.csv'")
    
    # Create summary statistics by position and year
    yearly_position_summary = all_predictions.groupby(['test_season', 'position']).agg({
        'actual_dropoff': ['count', 'mean'],
        'dropoff_probability': 'mean'
    }).round(3)
    yearly_position_summary.columns = ['n_predictions', 'actual_dropoff_rate', 'avg_predicted_prob']
    yearly_position_summary.to_csv('yearly_position_summary.csv')
    print(f"Yearly position summary saved to 'yearly_position_summary.csv'")
    
    # Save high-accuracy predictions for each year
    high_confidence_correct = all_predictions[
        (all_predictions['dropoff_probability'] > 0.7) & 
        (all_predictions['actual_dropoff'] == 1)
    ][['test_season', 'player_name', 'position', 'recent_team', 'fantasy_points', 'dropoff_probability']]
    
    high_confidence_correct.to_csv('high_confidence_correct_predictions.csv', index=False)
    print(f"High-confidence correct predictions saved to 'high_confidence_correct_predictions.csv'")


def main():
    """Run detailed backtest analysis."""
    # Use the same parameters as the main pipeline
    seasons = list(range(2005, 2025))
    
    pipeline = PlayerDropoffPipeline(
        seasons=seasons,
        dropoff_threshold=0.2,
        min_training_seasons=5
    )
    
    # Just run backtest (skip final training and predictions)
    data = pipeline.load_and_prepare_data()
    backtest_results = pipeline.run_backtest(data)
    backtest_analysis = pipeline.analyze_backtest_performance()
    
    pipeline_results = {
        'backtest_results': backtest_results,
        'backtest_analysis': backtest_analysis
    }
    
    # Run detailed analysis
    all_predictions = analyze_backtest_trends(pipeline_results)
    
    # Save detailed results
    if all_predictions is not None:
        save_detailed_results(pipeline_results, all_predictions)


if __name__ == "__main__":
    main()