# modules/backtester.py
"""
Backtesting and evaluation module for research paper.
Provides:
  - Classification metrics (accuracy, precision, recall, F1)
  - Regression metrics (RMSE, MAE, MAPE)
  - Directional accuracy
  - Confusion matrices
  - Ablation study runner
  - Simulated P&L (equity curve)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, mean_absolute_error, mean_squared_error,
    classification_report
)
import os
import json
from datetime import datetime


# =========================================
# Metric Computation
# =========================================

def compute_classification_metrics(y_true, y_pred, model_name="Model"):
    """Compute full classification metrics."""
    y_true = np.array(y_true).astype(int)
    y_pred = np.array(y_pred).astype(int)

    return {
        "Model": model_name,
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "Recall": round(recall_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "F1-Score": round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
    }


def compute_regression_metrics(y_true, y_pred, model_name="Model"):
    """Compute regression metrics."""
    y_true = np.array(y_true).astype(float)
    y_pred = np.array(y_pred).astype(float)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # MAPE (handle zeros)
    mask = y_true != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = 0.0

    # Directional accuracy
    dir_acc = np.mean((y_pred > 0) == (y_true > 0))

    return {
        "Model": model_name,
        "MAE": round(mae, 6),
        "RMSE": round(rmse, 6),
        "MAPE (%)": round(mape, 2),
        "Directional Acc": round(dir_acc, 4),
    }


def get_confusion_matrix(y_true, y_pred):
    """Return confusion matrix as numpy array."""
    return confusion_matrix(y_true, y_pred)


def get_classification_report_text(y_true, y_pred, target_names=None):
    """Return sklearn classification report as string."""
    if target_names is None:
        target_names = ["Bearish", "Bullish"]
    return classification_report(y_true, y_pred, target_names=target_names, zero_division=0)


# =========================================
# Feature Importance
# =========================================

def get_feature_importance(model, feature_names):
    """Extract feature importance from tree-based models."""
    try:
        importances = model.feature_importances_
        fi_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values("Importance", ascending=False)
        return fi_df
    except Exception:
        return pd.DataFrame({"Feature": [], "Importance": []})


# =========================================
# Simulated Trading P&L (Equity Curve)
# =========================================

def simulate_trading(y_true_returns, y_pred_direction, initial_capital=100000):
    """
    Simulate trading based on model predictions.
    Buy when model predicts Bullish (1), stay cash when Bearish (0).

    Parameters
    ----------
    y_true_returns : array
        Actual next-period returns (fractions, e.g., 0.02 = 2%)
    y_pred_direction : array
        Predicted direction (0=Bearish, 1=Bullish)
    initial_capital : float
        Starting capital

    Returns
    -------
    dict with equity_curve, total_return, buy_hold_return, sharpe_ratio
    """
    y_true_returns = np.array(y_true_returns).astype(float)
    y_pred_direction = np.array(y_pred_direction).astype(int)
    min_len = min(len(y_true_returns), len(y_pred_direction))
    y_true_returns = y_true_returns[:min_len]
    y_pred_direction = y_pred_direction[:min_len]

    # Model strategy: invest when bullish, hold cash when bearish
    strategy_returns = y_true_returns * y_pred_direction

    # Equity curves
    strategy_equity = initial_capital * np.cumprod(1 + strategy_returns)
    buyhold_equity = initial_capital * np.cumprod(1 + y_true_returns)

    # Metrics
    total_return = (strategy_equity[-1] / initial_capital - 1) * 100 if len(strategy_equity) > 0 else 0
    buyhold_return = (buyhold_equity[-1] / initial_capital - 1) * 100 if len(buyhold_equity) > 0 else 0

    # Sharpe ratio (annualized, assume 252 trading days)
    if len(strategy_returns) > 1 and np.std(strategy_returns) > 0:
        sharpe = (np.mean(strategy_returns) / np.std(strategy_returns)) * np.sqrt(252)
    else:
        sharpe = 0.0

    # Max drawdown
    peak = np.maximum.accumulate(strategy_equity)
    drawdown = (strategy_equity - peak) / peak
    max_drawdown = np.min(drawdown) * 100 if len(drawdown) > 0 else 0

    return {
        "strategy_equity": strategy_equity.tolist(),
        "buyhold_equity": buyhold_equity.tolist(),
        "total_return_pct": round(total_return, 2),
        "buyhold_return_pct": round(buyhold_return, 2),
        "sharpe_ratio": round(sharpe, 4),
        "max_drawdown_pct": round(max_drawdown, 2),
        "n_trades": int(np.sum(y_pred_direction)),
        "n_periods": int(min_len),
    }


# =========================================
# Ablation Study
# =========================================

def run_ablation_study(data: pd.DataFrame, sentiment_scores: dict):
    """
    Run ablation study comparing:
      1. ML only (sentiment_score=0)
      2. ML + FinBERT only
      3. ML + TextBlob only
      4. ML + Hybrid Sentiment (full system)
      5. Baseline (always majority)

    Parameters
    ----------
    data : pd.DataFrame (OHLCV)
    sentiment_scores : dict with keys 'hybrid', 'finbert', 'textblob'
        Each value is a float in [-1, 1]

    Returns
    -------
    pd.DataFrame with ablation results
    """
    from modules.predictive_ml import train_all_models

    variants = {
        "ML Only (No Sentiment)": 0.0,
        "ML + FinBERT": sentiment_scores.get("finbert", 0.0),
        "ML + TextBlob": sentiment_scores.get("textblob", 0.0),
        "ML + Hybrid (Full System)": sentiment_scores.get("hybrid", 0.0),
    }

    results = []
    for variant_name, sent_score in variants.items():
        try:
            res = train_all_models(data, sentiment_score=sent_score)
            if res and "models" in res:
                # Use the best non-baseline model for this variant
                best_model = None
                best_acc = -1
                for mname, mdata in res["models"].items():
                    if "Baseline" in mname:
                        continue
                    if mdata.get("accuracy", 0) > best_acc:
                        best_acc = mdata["accuracy"]
                        best_model = mname

                if best_model:
                    m = res["models"][best_model]
                    results.append({
                        "Variant": variant_name,
                        "Best Model": best_model,
                        "Accuracy": round(m["accuracy"], 4),
                        "F1-Score": round(m["f1"], 4),
                        "RMSE": round(m["rmse"], 6),
                        "MAE": round(m["mae"], 6),
                        "Dir. Accuracy": round(m["directional_accuracy"], 4),
                    })

                # Also include baseline for reference (only once)
                if variant_name == "ML Only (No Sentiment)" and "Baseline (Always Majority)" in res["models"]:
                    bl = res["models"]["Baseline (Always Majority)"]
                    results.append({
                        "Variant": "Baseline (Always Majority)",
                        "Best Model": "Naive",
                        "Accuracy": round(bl["accuracy"], 4),
                        "F1-Score": round(bl["f1"], 4),
                        "RMSE": round(bl["rmse"], 6),
                        "MAE": round(bl["mae"], 6),
                        "Dir. Accuracy": round(bl["directional_accuracy"], 4),
                    })
        except Exception as e:
            print(f"Ablation error for {variant_name}: {e}")
            results.append({
                "Variant": variant_name, "Best Model": "Error",
                "Accuracy": 0, "F1-Score": 0, "RMSE": 0, "MAE": 0, "Dir. Accuracy": 0,
            })

    return pd.DataFrame(results)


# =========================================
# Results Saver
# =========================================

def save_results(results_dict: dict, output_dir: str = "results"):
    """
    Save experiment results to CSV and JSON files.

    Parameters
    ----------
    results_dict : dict
        Keys like 'metrics_comparison', 'ablation_study', 'trading_simulation', etc.
        Values are DataFrames or dicts.
    output_dir : str
        Directory to save results.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "confusion_matrices"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for key, value in results_dict.items():
        filepath = os.path.join(output_dir, f"{key}_{timestamp}")
        if isinstance(value, pd.DataFrame):
            value.to_csv(f"{filepath}.csv", index=False)
        elif isinstance(value, dict):
            # Convert numpy types for JSON serialization
            clean = {}
            for k, v in value.items():
                if isinstance(v, (np.integer,)):
                    clean[k] = int(v)
                elif isinstance(v, (np.floating,)):
                    clean[k] = float(v)
                elif isinstance(v, np.ndarray):
                    clean[k] = v.tolist()
                else:
                    clean[k] = v
            with open(f"{filepath}.json", "w") as f:
                json.dump(clean, f, indent=2)

    print(f"Results saved to {output_dir}/")
