# run_experiments.py
"""
Standalone experiment runner for research paper.
Runs all models × ablation variants on representative NSE stocks.
Saves results to results/ folder.

Usage:
    python run_experiments.py
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.utils import fetch_price_data
from modules.predictive_ml import train_all_models
from modules.backtester import (
    compute_classification_metrics,
    compute_regression_metrics,
    get_confusion_matrix,
    get_feature_importance,
    simulate_trading,
    run_ablation_study,
    save_results,
    get_classification_report_text,
)
from modules.feature_engineering import get_feature_columns

# Try to import sentiment modules
try:
    from modules.sentiment_engine import analyze_finbert, analyze_general_sentiment, analyze_hybrid_sentiment, get_news_for_stock
    _SENTIMENT_AVAILABLE = True
except Exception:
    _SENTIMENT_AVAILABLE = False


# =========================================
# Configuration
# =========================================

# Representative NSE stocks for experiments (mix of sectors)
EXPERIMENT_STOCKS = [
    "RELIANCE.NS",   # Energy / Conglomerate
    "TCS.NS",        # IT
    "HDFCBANK.NS",   # Banking
    "INFY.NS",       # IT
    "ITC.NS",        # FMCG
    "SBIN.NS",       # Public Bank
    "BHARTIARTL.NS", # Telecom
    "TATAMOTORS.NS", # Auto
    "SUNPHARMA.NS",  # Pharma
    "WIPRO.NS",      # IT
]

HORIZONS = {
    "intraday": {"period": "5d", "interval": "1h"},
    "long_term": {"period": "6mo", "interval": "1d"},
}

RESULTS_DIR = "results"


# =========================================
# Sentiment Score Fetcher
# =========================================

def get_sentiment_scores(ticker):
    """Fetch sentiment scores for a stock (FinBERT, TextBlob, Hybrid)."""
    if not _SENTIMENT_AVAILABLE:
        return {"finbert": 0.0, "textblob": 0.0, "hybrid": 0.0}

    try:
        headlines = get_news_for_stock(ticker)
        if not headlines:
            return {"finbert": 0.0, "textblob": 0.0, "hybrid": 0.0}

        finbert_scores = []
        textblob_scores = []
        hybrid_scores = []

        for h in headlines[:10]:  # Limit to 10 for speed
            title = h.get("title", "")
            if not title:
                continue

            fin = analyze_finbert(title)
            gen = analyze_general_sentiment(title)
            hyb = analyze_hybrid_sentiment(title)

            finbert_scores.append(fin.get("positive", 0) - fin.get("negative", 0))
            textblob_scores.append(gen.get("positive", 0) - gen.get("negative", 0))
            hybrid_scores.append(hyb.get("positive", 0) - hyb.get("negative", 0))

        return {
            "finbert": float(np.mean(finbert_scores)) if finbert_scores else 0.0,
            "textblob": float(np.mean(textblob_scores)) if textblob_scores else 0.0,
            "hybrid": float(np.mean(hybrid_scores)) if hybrid_scores else 0.0,
        }
    except Exception as e:
        print(f"  Sentiment fetch error for {ticker}: {e}")
        return {"finbert": 0.0, "textblob": 0.0, "hybrid": 0.0}


# =========================================
# Main Experiment Runner
# =========================================

def run_all_experiments():
    print("=" * 60)
    print("RESEARCH EXPERIMENT RUNNER")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Stocks: {len(EXPERIMENT_STOCKS)}")
    print("=" * 60)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "confusion_matrices"), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "plots"), exist_ok=True)

    all_cls_metrics = []
    all_reg_metrics = []
    all_trading_results = []
    all_feature_importance = []
    all_ablation_results = []
    all_confusion_matrices = {}

    for horizon_name, horizon_params in HORIZONS.items():
        print(f"\n{'='*50}")
        print(f"HORIZON: {horizon_name.upper()}")
        print(f"{'='*50}")

        for stock in EXPERIMENT_STOCKS:
            print(f"\n--- Processing {stock} ({horizon_name}) ---")

            # 1. Fetch data
            try:
                data = fetch_price_data(stock, **horizon_params)
                if data is None or data.empty:
                    print(f"  No data for {stock}, skipping.")
                    continue
                print(f"  Data: {len(data)} rows, {data.index[0]} to {data.index[-1]}")
            except Exception as e:
                print(f"  Data fetch error: {e}")
                continue

            # 2. Get sentiment scores
            print(f"  Fetching sentiment...")
            sent_scores = get_sentiment_scores(stock)
            print(f"  Sentiment — FinBERT: {sent_scores['finbert']:.3f}, "
                  f"TextBlob: {sent_scores['textblob']:.3f}, "
                  f"Hybrid: {sent_scores['hybrid']:.3f}")

            # 3. Train all models (with hybrid sentiment)
            print(f"  Training models...")
            try:
                results = train_all_models(data, sentiment_score=sent_scores["hybrid"])
            except Exception as e:
                print(f"  Training error: {e}")
                continue

            if results is None:
                print(f"  Not enough data to train, skipping.")
                continue

            y_cls_test = results["y_cls_test"]
            y_reg_test = results["y_reg_test"]

            for model_name, model_data in results["models"].items():
                print(f"    {model_name}: Acc={model_data.get('accuracy', 0):.4f}, "
                      f"F1={model_data.get('f1', 0):.4f}, "
                      f"RMSE={model_data.get('rmse', 0):.6f}")

                # Classification metrics
                cls_metrics = compute_classification_metrics(
                    y_cls_test[:len(model_data["cls_pred"])],
                    model_data["cls_pred"],
                    model_name
                )
                cls_metrics["Stock"] = stock
                cls_metrics["Horizon"] = horizon_name
                all_cls_metrics.append(cls_metrics)

                # Regression metrics
                reg_metrics = compute_regression_metrics(
                    y_reg_test[:len(model_data["reg_pred"])],
                    model_data["reg_pred"],
                    model_name
                )
                reg_metrics["Stock"] = stock
                reg_metrics["Horizon"] = horizon_name
                all_reg_metrics.append(reg_metrics)

                # Confusion matrix
                cm = get_confusion_matrix(
                    y_cls_test[:len(model_data["cls_pred"])],
                    model_data["cls_pred"]
                )
                all_confusion_matrices[f"{stock}_{horizon_name}_{model_name}"] = cm.tolist()

                # Feature importance (for tree models only)
                if model_name in ["RandomForest", "XGBoost"] and "clf" in model_data:
                    feature_cols = [c for c in get_feature_columns()]
                    fi = get_feature_importance(model_data["clf"], feature_cols)
                    if not fi.empty:
                        fi["Stock"] = stock
                        fi["Horizon"] = horizon_name
                        fi["Model"] = model_name
                        all_feature_importance.append(fi)

                # Trading simulation
                if "Baseline" not in model_name:
                    try:
                        sim = simulate_trading(
                            y_reg_test[:len(model_data["cls_pred"])],
                            model_data["cls_pred"]
                        )
                        sim["Stock"] = stock
                        sim["Horizon"] = horizon_name
                        sim["Model"] = model_name
                        all_trading_results.append(sim)
                    except Exception as e:
                        print(f"    Trading sim error: {e}")

            # 4. Run ablation study
            print(f"  Running ablation study...")
            try:
                ablation_df = run_ablation_study(data, sent_scores)
                if not ablation_df.empty:
                    ablation_df["Stock"] = stock
                    ablation_df["Horizon"] = horizon_name
                    all_ablation_results.append(ablation_df)
            except Exception as e:
                print(f"  Ablation error: {e}")

    # =========================================
    # Aggregate and Save Results
    # =========================================
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)

    results_to_save = {}

    # Classification metrics
    if all_cls_metrics:
        cls_df = pd.DataFrame(all_cls_metrics)
        results_to_save["classification_metrics"] = cls_df
        print(f"\nClassification Metrics Summary:")
        print(cls_df.groupby("Model")[["Accuracy", "Precision", "Recall", "F1-Score"]].mean().round(4).to_string())

    # Regression metrics
    if all_reg_metrics:
        reg_df = pd.DataFrame(all_reg_metrics)
        results_to_save["regression_metrics"] = reg_df
        print(f"\nRegression Metrics Summary:")
        print(reg_df.groupby("Model")[["MAE", "RMSE", "MAPE (%)", "Directional Acc"]].mean().round(4).to_string())

    # Trading simulations
    if all_trading_results:
        # Extract non-list fields for CSV
        trading_summary = []
        for sim in all_trading_results:
            trading_summary.append({
                "Stock": sim["Stock"],
                "Horizon": sim["Horizon"],
                "Model": sim["Model"],
                "Strategy Return (%)": sim["total_return_pct"],
                "Buy&Hold Return (%)": sim["buyhold_return_pct"],
                "Sharpe Ratio": sim["sharpe_ratio"],
                "Max Drawdown (%)": sim["max_drawdown_pct"],
                "# Trades": sim["n_trades"],
                "# Periods": sim["n_periods"],
            })
        trading_df = pd.DataFrame(trading_summary)
        results_to_save["trading_simulation"] = trading_df
        print(f"\nTrading Simulation Summary:")
        print(trading_df.groupby("Model")[["Strategy Return (%)", "Buy&Hold Return (%)", "Sharpe Ratio"]].mean().round(4).to_string())

    # Feature importance (aggregate top features)
    if all_feature_importance:
        fi_df = pd.concat(all_feature_importance, ignore_index=True)
        results_to_save["feature_importance"] = fi_df
        print(f"\nTop 10 Features (avg importance):")
        top_fi = fi_df.groupby("Feature")["Importance"].mean().sort_values(ascending=False).head(10)
        print(top_fi.round(4).to_string())

    # Ablation study
    if all_ablation_results:
        ablation_df = pd.concat(all_ablation_results, ignore_index=True)
        results_to_save["ablation_study"] = ablation_df
        print(f"\nAblation Study Summary:")
        print(ablation_df.groupby("Variant")[["Accuracy", "F1-Score", "RMSE", "Dir. Accuracy"]].mean().round(4).to_string())

    # Confusion matrices
    if all_confusion_matrices:
        results_to_save["confusion_matrices"] = all_confusion_matrices

    # Save everything
    save_results(results_to_save, RESULTS_DIR)

    print("\n" + "=" * 60)
    print(f"DONE! Results saved to {RESULTS_DIR}/")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)


if __name__ == "__main__":
    run_all_experiments()
