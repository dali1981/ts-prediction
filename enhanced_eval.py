import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any

class FinancialModelEvaluator:
    """Comprehensive evaluation framework for financial time series models."""
    
    def __init__(self, predictions: np.ndarray, actuals: np.ndarray, prices: np.ndarray):
        self.predictions = predictions
        self.actuals = actuals  
        self.prices = prices
        
    def directional_accuracy(self) -> Dict[str, float]:
        """Calculate directional prediction accuracy."""
        pred_direction = np.sign(np.diff(self.predictions))
        actual_direction = np.sign(np.diff(self.actuals))
        
        accuracy = accuracy_score(actual_direction > 0, pred_direction > 0)
        
        return {
            'directional_accuracy': accuracy,
            'up_predictions': np.sum(pred_direction > 0),
            'down_predictions': np.sum(pred_direction < 0),
            'actual_ups': np.sum(actual_direction > 0),
            'actual_downs': np.sum(actual_direction < 0)
        }
    
    def trading_metrics(self, transaction_cost: float = 0.001) -> Dict[str, float]:
        """Calculate trading-specific metrics."""
        # Simple strategy: buy when prediction > current, sell otherwise
        positions = np.where(self.predictions > self.prices[:-len(self.predictions)], 1, -1)
        returns = np.diff(self.prices[-len(positions):]) / self.prices[-len(positions)-1:-1]
        
        # Apply transaction costs
        position_changes = np.abs(np.diff(np.concatenate([[0], positions])))
        costs = position_changes * transaction_cost
        
        strategy_returns = positions[:-1] * returns - costs[1:]
        
        # Calculate metrics
        total_return = np.sum(strategy_returns)
        volatility = np.std(strategy_returns) * np.sqrt(252)  # Annualized
        sharpe_ratio = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252) if np.std(strategy_returns) > 0 else 0
        
        # Maximum drawdown
        cumulative = np.cumprod(1 + strategy_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        return {
            'total_return': total_return,
            'annualized_volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'hit_rate': np.mean(strategy_returns > 0)
        }
    
    def prediction_analysis(self) -> Dict[str, Any]:
        """Analyze prediction quality and characteristics."""
        return {
            'prediction_mean': np.mean(self.predictions),
            'prediction_std': np.std(self.predictions),
            'prediction_range': np.max(self.predictions) - np.min(self.predictions),
            'correlation_with_actual': np.corrcoef(self.predictions, self.actuals)[0, 1],
            'mean_absolute_error': np.mean(np.abs(self.predictions - self.actuals)),
            'root_mean_square_error': np.sqrt(np.mean((self.predictions - self.actuals) ** 2))
        }
    
    def generate_report(self) -> None:
        """Generate comprehensive evaluation report."""
        print("=== FINANCIAL MODEL EVALUATION REPORT ===\n")
        
        # Basic prediction metrics
        pred_metrics = self.prediction_analysis()
        print("📊 PREDICTION QUALITY:")
        print(f"  Correlation with actual: {pred_metrics['correlation_with_actual']:.4f}")
        print(f"  MAE: {pred_metrics['mean_absolute_error']:.4f}")
        print(f"  RMSE: {pred_metrics['root_mean_square_error']:.4f}")
        
        # Directional accuracy
        dir_metrics = self.directional_accuracy()
        print(f"\n🎯 DIRECTIONAL ACCURACY:")
        print(f"  Accuracy: {dir_metrics['directional_accuracy']:.2%}")
        print(f"  Predicted ups: {dir_metrics['up_predictions']}, downs: {dir_metrics['down_predictions']}")
        print(f"  Actual ups: {dir_metrics['actual_ups']}, downs: {dir_metrics['actual_downs']}")
        
        # Trading metrics
        trading = self.trading_metrics()
        print(f"\n💰 TRADING PERFORMANCE:")
        print(f"  Total return: {trading['total_return']:.2%}")
        print(f"  Sharpe ratio: {trading['sharpe_ratio']:.3f}")
        print(f"  Max drawdown: {trading['max_drawdown']:.2%}")
        print(f"  Hit rate: {trading['hit_rate']:.2%}")
        
        # Model utility assessment
        print(f"\n🔍 MODEL UTILITY ASSESSMENT:")
        if dir_metrics['directional_accuracy'] > 0.55:
            print("  ✅ Directional accuracy suggests potential trading value")
        else:
            print("  ❌ Poor directional accuracy - not suitable for trading")
            
        if abs(pred_metrics['correlation_with_actual']) > 0.3:
            print("  ✅ Reasonable correlation with actual prices")
        else:
            print("  ❌ Weak correlation - predictions may be noisy")
            
        if trading['sharpe_ratio'] > 1.0:
            print("  ✅ Strong risk-adjusted returns")
        elif trading['sharpe_ratio'] > 0.5:
            print("  ⚠️  Moderate risk-adjusted returns")
        else:
            print("  ❌ Poor risk-adjusted returns")

# Usage example for the notebook results
def evaluate_notebook_model():
    """Specific evaluation for the notebook's fusion model results."""
    print("🔍 NOTEBOOK MODEL ANALYSIS:")
    print("\n❌ CRITICAL ISSUES IDENTIFIED:")
    print("  • Severe overfitting (val_loss >> train_loss)")
    print("  • No convergence over 200 epochs") 
    print("  • Test loss (6974) indicates complete failure")
    print("  • Missing essential financial metrics")
    
    print("\n📋 RECOMMENDED IMMEDIATE FIXES:")
    print("  1. Add regularization (dropout, weight decay)")
    print("  2. Reduce model complexity or increase data")
    print("  3. Implement proper cross-validation")
    print("  4. Add directional accuracy tracking")
    print("  5. Include trading simulation metrics")
    
    print("\n🎯 DATA QUALITY IMPROVEMENTS:")
    print("  • Increase dataset size (>10k samples minimum)")
    print("  • Add more diverse market conditions")
    print("  • Implement proper train/val/test splits")
    print("  • Add feature engineering (technical indicators)")
    
    return "Model requires significant improvements before practical use"

# Diagnostic tools for Brooks tokenization
def diagnose_brooks_tokens(token_data: pd.Series) -> Dict[str, Any]:
    """Analyze Brooks token effectiveness for market prediction."""
    token_counts = token_data.value_counts()
    entropy = -np.sum((token_counts / len(token_data)) * np.log2(token_counts / len(token_data)))
    
    return {
        'entropy': entropy,
        'unique_tokens': len(token_counts),
        'most_common': token_counts.head().to_dict(),
        'token_distribution': 'balanced' if entropy > 4.0 else 'skewed'
    }

if __name__ == "__main__":
    evaluate_notebook_model()