# modules/statistical_tests.py
"""
Statistical significance testing module for research paper.
Provides rigorous statistical analysis of model performance.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


# =========================================
# Paired Statistical Tests
# =========================================

def paired_ttest(results_model1: np.ndarray, results_model2: np.ndarray, 
                 model1_name: str = "Model A", model2_name: str = "Model B") -> Dict:
    """
    Perform paired t-test between two models' performance metrics.
    
    Args:
        results_model1: Array of performance scores for model 1 (e.g., accuracies across folds)
        results_model2: Array of performance scores for model 2
        model1_name: Name of first model
        model2_name: Name of second model
    
    Returns:
        Dictionary with test statistics, p-value, and interpretation
    """
    results_model1 = np.array(results_model1)
    results_model2 = np.array(results_model2)
    
    if len(results_model1) != len(results_model2):
        raise ValueError("Arrays must have same length for paired test")
    
    if len(results_model1) < 2:
        return {
            "test": "Paired t-test",
            "model1": model1_name,
            "model2": model2_name,
            "mean_diff": 0.0,
            "t_statistic": 0.0,
            "p_value": 1.0,
            "significant": False,
            "interpretation": "Insufficient samples for testing"
        }
    
    # Perform paired t-test
    t_stat, p_value = stats.ttest_rel(results_model1, results_model2)
    
    mean_diff = np.mean(results_model1) - np.mean(results_model2)
    
    # Effect size (Cohen's d for paired samples)
    diff = results_model1 - results_model2
    cohen_d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff) > 0 else 0
    
    return {
        "test": "Paired t-test",
        "model1": model1_name,
        "model2": model2_name,
        "model1_mean": float(np.mean(results_model1)),
        "model2_mean": float(np.mean(results_model2)),
        "mean_diff": float(mean_diff),
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "cohen_d": float(cohen_d),
        "significant": p_value < 0.05,
        "interpretation": _interpret_ttest(mean_diff, p_value, model1_name, model2_name)
    }


def _interpret_ttest(mean_diff: float, p_value: float, model1: str, model2: str) -> str:
    """Generate human-readable interpretation of t-test results."""
    if p_value >= 0.05:
        return f"No significant difference between {model1} and {model2} (p={p_value:.4f})"
    
    if mean_diff > 0:
        return f"{model1} significantly outperforms {model2} (p={p_value:.4f}, diff={mean_diff:.4f})"
    else:
        return f"{model2} significantly outperforms {model1} (p={p_value:.4f}, diff={abs(mean_diff):.4f})"


# =========================================
# McNemar's Test (Classification)
# =========================================

def mcnemar_test(y_true: np.ndarray, pred_model1: np.ndarray, pred_model2: np.ndarray,
                 model1_name: str = "Model A", model2_name: str = "Model B") -> Dict:
    """
    McNemar's test for comparing two classifiers on same dataset.
    Tests whether two models make significantly different errors.
    
    Args:
        y_true: True labels
        pred_model1: Predictions from model 1
        pred_model2: Predictions from model 2
    
    Returns:
        Dictionary with test statistics and interpretation
    """
    y_true = np.array(y_true)
    pred_model1 = np.array(pred_model1)
    pred_model2 = np.array(pred_model2)
    
    # Create contingency table
    # n01: model1 wrong, model2 correct
    # n10: model1 correct, model2 wrong
    correct1 = (pred_model1 == y_true)
    correct2 = (pred_model2 == y_true)
    
    n01 = np.sum(~correct1 & correct2)  # model1 wrong, model2 right
    n10 = np.sum(correct1 & ~correct2)  # model1 right, model2 wrong
    
    # McNemar's test statistic with continuity correction
    if (n01 + n10) == 0:
        p_value = 1.0
        chi2_stat = 0.0
    else:
        chi2_stat = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
        p_value = 1 - stats.chi2.cdf(chi2_stat, df=1)
    
    acc1 = np.mean(correct1)
    acc2 = np.mean(correct2)
    
    return {
        "test": "McNemar's test",
        "model1": model1_name,
        "model2": model2_name,
        "model1_accuracy": float(acc1),
        "model2_accuracy": float(acc2),
        "n01": int(n01),  # model1 wrong, model2 right
        "n10": int(n10),  # model1 right, model2 wrong
        "chi2_statistic": float(chi2_stat),
        "p_value": float(p_value),
        "significant": p_value < 0.05,
        "interpretation": _interpret_mcnemar(acc1, acc2, p_value, model1_name, model2_name)
    }


def _interpret_mcnemar(acc1: float, acc2: float, p_value: float, model1: str, model2: str) -> str:
    """Generate interpretation of McNemar's test."""
    if p_value >= 0.05:
        return f"No significant difference in error patterns between {model1} and {model2} (p={p_value:.4f})"
    
    if acc1 > acc2:
        return f"{model1} makes significantly different (better) predictions than {model2} (p={p_value:.4f})"
    else:
        return f"{model2} makes significantly different (better) predictions than {model1} (p={p_value:.4f})"


# =========================================
# Bootstrap Confidence Intervals
# =========================================

def bootstrap_ci(data: np.ndarray, n_bootstrap: int = 1000, 
                 confidence_level: float = 0.95, metric_func=None) -> Tuple[float, float, float]:
    """
    Calculate bootstrap confidence intervals for a metric.
    
    Args:
        data: Original data
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (default 0.95 for 95% CI)
        metric_func: Function to compute metric (default: mean)
    
    Returns:
        (mean, lower_bound, upper_bound)
    """
    if metric_func is None:
        metric_func = np.mean
    
    data = np.array(data)
    n = len(data)
    
    if n < 2:
        return float(metric_func(data)), float(metric_func(data)), float(metric_func(data))
    
    bootstrap_metrics = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_metrics.append(metric_func(sample))
    
    bootstrap_metrics = np.array(bootstrap_metrics)
    
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    mean_val = float(metric_func(data))
    lower_bound = float(np.percentile(bootstrap_metrics, lower_percentile))
    upper_bound = float(np.percentile(bootstrap_metrics, upper_percentile))
    
    return mean_val, lower_bound, upper_bound


# =========================================
# Diebold-Mariano Test (Forecasting)
# =========================================

def diebold_mariano_test(errors_model1: np.ndarray, errors_model2: np.ndarray,
                        model1_name: str = "Model A", model2_name: str = "Model B",
                        loss_function: str = "MSE") -> Dict:
    """
    Diebold-Mariano test for comparing forecast accuracy.
    Tests if two forecasts have significantly different accuracy.
    
    Args:
        errors_model1: Forecast errors from model 1 (y_true - y_pred)
        errors_model2: Forecast errors from model 2
        model1_name: Name of first model
        model2_name: Name of second model  
        loss_function: "MSE" or "MAE"
    
    Returns:
        Dictionary with test statistics and interpretation
    """
    errors_model1 = np.array(errors_model1)
    errors_model2 = np.array(errors_model2)
    
    if len(errors_model1) != len(errors_model2):
        raise ValueError("Error arrays must have same length")
    
    # Compute loss differential
    if loss_function.upper() == "MSE":
        d = errors_model1**2 - errors_model2**2
    elif loss_function.upper() == "MAE":
        d = np.abs(errors_model1) - np.abs(errors_model2)
    else:
        raise ValueError(f"Unknown loss function: {loss_function}")
    
    # Mean and variance of loss differential
    d_mean = np.mean(d)
    d_var = np.var(d, ddof=1)
    
    n = len(d)
    
    if d_var == 0 or n < 2:
        return {
            "test": "Diebold-Mariano",
            "model1": model1_name,
            "model2": model2_name,
            "dm_statistic": 0.0,
            "p_value": 1.0,
            "significant": False,
            "interpretation": "Insufficient variance or samples for testing"
        }
    
    # DM test statistic
    dm_stat = d_mean / np.sqrt(d_var / n)
    
    # Two-tailed test
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    
    return {
        "test": "Diebold-Mariano",
        "model1": model1_name,
        "model2": model2_name,
        "loss_function": loss_function,
        "dm_statistic": float(dm_stat),
        "p_value": float(p_value),
        "significant": p_value < 0.05,
        "mean_loss_diff": float(d_mean),
        "interpretation": _interpret_dm(d_mean, p_value, model1_name, model2_name)
    }


def _interpret_dm(mean_diff: float, p_value: float, model1: str, model2: str) -> str:
    """Generate interpretation of Diebold-Mariano test."""
    if p_value >= 0.05:
        return f"No significant difference in forecast accuracy between {model1} and {model2} (p={p_value:.4f})"
    
    if mean_diff < 0:
        return f"{model1} has significantly better forecast accuracy than {model2} (p={p_value:.4f})"
    else:
        return f"{model2} has significantly better forecast accuracy than {model1} (p={p_value:.4f})"


# =========================================
# Comprehensive Model Comparison
# =========================================

def compare_models_statistical(results_df: pd.DataFrame, metric_column: str = "Accuracy",
                               model_column: str = "Model") -> pd.DataFrame:
    """
    Perform pairwise statistical comparisons between all models.
    
    Args:
        results_df: DataFrame with columns [Model, Fold, Accuracy/F1/etc.]
        metric_column: Name of metric column to compare
        model_column: Name of model column
    
    Returns:
        DataFrame with pairwise comparison results
    """
    models = results_df[model_column].unique()
    comparisons = []
    
    for i, model1 in enumerate(models):
        for model2 in models[i+1:]:
            scores1 = results_df[results_df[model_column] == model1][metric_column].values
            scores2 = results_df[results_df[model_column] == model2][metric_column].values
            
            # Ensure same number of folds
            min_len = min(len(scores1), len(scores2))
            scores1 = scores1[:min_len]
            scores2 = scores2[:min_len]
            
            if len(scores1) >= 2:
                test_result = paired_ttest(scores1, scores2, model1, model2)
                comparisons.append({
                    "Model_1": model1,
                    "Model_2": model2,
                    "Mean_1": test_result["model1_mean"],
                    "Mean_2": test_result["model2_mean"],
                    "Difference": test_result["mean_diff"],
                    "p_value": test_result["p_value"],
                    "Significant": "✓" if test_result["significant"] else "✗",
                    "Winner": model1 if test_result["model1_mean"] > test_result["model2_mean"] else model2
                })
    
    return pd.DataFrame(comparisons)


# =========================================
# Results with Confidence Intervals
# =========================================

def add_confidence_intervals(results_df: pd.DataFrame, metric_columns: List[str],
                            group_by: str = "Model", n_bootstrap: int = 1000) -> pd.DataFrame:
    """
    Add bootstrap confidence intervals to aggregated results.
    
    Args:
        results_df: DataFrame with multiple folds/runs per model
        metric_columns: List of metric column names
        group_by: Column to group by (usually "Model")
    
    Returns:
        DataFrame with mean ± CI for each metric
    """
    ci_results = []
    
    for model_name, group in results_df.groupby(group_by):
        row = {"Model": model_name}
        
        for metric in metric_columns:
            if metric in group.columns:
                data = group[metric].values
                mean, lower, upper = bootstrap_ci(data, n_bootstrap=n_bootstrap)
                
                row[f"{metric}_mean"] = mean
                row[f"{metric}_CI_lower"] = lower
                row[f"{metric}_CI_upper"] = upper
                row[f"{metric}_formatted"] = f"{mean:.4f} [{lower:.4f}, {upper:.4f}]"
        
        ci_results.append(row)
    
    return pd.DataFrame(ci_results)
