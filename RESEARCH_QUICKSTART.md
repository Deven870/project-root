# Quick Start: Research Experiments

## Running Experiments with Statistical Testing

### 1. Basic Experiment with Baselines

```python
import pandas as pd
from modules.utils import fetch_price_data
from modules.baseline_models import run_all_baselines
from modules.predictive_ml import train_all_models
from modules.backtester import simulate_trading

# Fetch data
ticker = "RELIANCE.NS"
data = fetch_price_data(ticker, period="6mo", interval="1d")
prices = data['Close']

# Run baseline strategies
baseline_results = run_all_baselines(prices)
print("\n=== Baseline Strategies ===")
print(baseline_results.to_string(index=False))

# Compare with your ML models
# (Full implementation in run_experiments.py)
```

### 2. Statistical Significance Testing

```python
from modules.statistical_tests import (
    paired_ttest,
    mcnemar_test,
    bootstrap_ci,
    compare_models_statistical
)
import numpy as np

# Example: Compare two models across multiple folds
model1_accuracies = [0.65, 0.68, 0.67, 0.70, 0.66]  # 5-fold CV
model2_accuracies = [0.62, 0.64, 0.63, 0.65, 0.61]

result = paired_ttest(
    model1_accuracies, 
    model2_accuracies,
    "Random Forest",
    "XGBoost"
)

print(f"\nStatistical Test Result:")
print(f"  {result['interpretation']}")
print(f"  p-value: {result['p_value']:.4f}")
print(f"  Cohen's d: {result['cohen_d']:.4f}")

# Add confidence intervals
mean, lower, upper = bootstrap_ci(model1_accuracies)
print(f"\nRandom Forest Accuracy: {mean:.4f} [{lower:.4f}, {upper:.4f}]")
```

### 3. Complete Comparison Pipeline

```python
# Assuming you have results from multiple models and folds
results_df = pd.DataFrame({
    'Model': ['RF', 'RF', 'RF', 'XGB', 'XGB', 'XGB', 'LSTM', 'LSTM', 'LSTM'],
    'Fold': [1, 2, 3, 1, 2, 3, 1, 2, 3],
    'Accuracy': [0.65, 0.68, 0.67, 0.62, 0.64, 0.63, 0.70, 0.72, 0.71],
    'F1-Score': [0.63, 0.66, 0.65, 0.60, 0.62, 0.61, 0.68, 0.70, 0.69]
})

# Pairwise statistical comparison
comparison = compare_models_statistical(results_df, metric_column='Accuracy')
print("\n=== Pairwise Model Comparison ===")
print(comparison.to_string(index=False))

# Add confidence intervals
from modules.statistical_tests import add_confidence_intervals
ci_results = add_confidence_intervals(
    results_df, 
    metric_columns=['Accuracy', 'F1-Score'],
    group_by='Model'
)
print("\n=== Results with Confidence Intervals ===")
print(ci_results[['Model', 'Accuracy_formatted', 'F1-Score_formatted']].to_string(index=False))
```

### 4. Running Full Research Pipeline

```bash
# Run comprehensive experiments with all stocks
python run_experiments.py

# This will:
# 1. Train all models (RF, XGBoost, LSTM)
# 2. Run ablation studies (with/without sentiment)
# 3. Perform backtesting
# 4. Generate trading simulations
# 5. Save all results to results/ folder
```

### 5. Analyzing Results for Paper

```python
import glob
import os

# Load saved results
results_dir = "results"
metric_files = glob.glob(os.path.join(results_dir, "metrics_*.csv"))

all_metrics = []
for f in metric_files:
    df = pd.read_csv(f)
    all_metrics.append(df)

combined = pd.concat(all_metrics, ignore_index=True)

# Statistical comparison
from modules.statistical_tests import compare_models_statistical

comparison = compare_models_statistical(
    combined,
    metric_column='Accuracy',
    model_column='Model'
)

# Identify best model
best_model = comparison.loc[comparison['p_value'].idxmin(), 'Winner']
print(f"\nBest performing model: {best_model}")

# Generate LaTeX table for paper
print("\n=== LaTeX Table ===")
print(comparison.to_latex(index=False, float_format="%.4f"))
```

### 6. Baseline vs ML Comparison

```python
# Compare your best ML model with baselines
ticker = "TCS.NS"
data = fetch_price_data(ticker, period="6mo", interval="1d")

# Run baselines
baseline_results = run_all_baselines(data['Close'])

# Your ML model results (from backtesting)
ml_results = {
    "Strategy": "ML Hybrid",
    "Total Return (%)": 15.5,
    "Sharpe Ratio": 1.25,
    "Max Drawdown (%)": 8.3,
    "# Trades": 45
}

# Combine
all_strategies = pd.concat([
    baseline_results,
    pd.DataFrame([ml_results])
], ignore_index=True)

print("\n=== Strategy Comparison ===")
print(all_strategies.to_string(index=False))

# Statistical test vs best baseline
best_baseline_idx = baseline_results['Sharpe Ratio'].idxmax()
best_baseline = baseline_results.iloc[best_baseline_idx]['Strategy']

print(f"\nYour ML model vs {best_baseline}:")
print(f"  Return difference: {ml_results['Total Return (%)'] - baseline_results.iloc[best_baseline_idx]['Total Return (%)']:.2f}%")
print(f"  Sharpe improvement: {ml_results['Sharpe Ratio'] - baseline_results.iloc[best_baseline_idx]['Sharpe Ratio']:.2f}")
```

## Expected Workflow for Paper

1. **Data Collection** (1 day)
   - Run experiments on all 10 stocks
   - Save results to results/ folder
   - Document data statistics

2. **Model Training** (2-3 days)
   - 5-fold time-series cross-validation
   - Train all models with different configurations
   - Run ablation studies

3. **Statistical Analysis** (1 day)
   - Run pairwise t-tests
   - Calculate confidence intervals
   - Perform McNemar's tests for classification

4. **Baseline Comparison** (1 day)
   - Run all baseline strategies
   - Compare ML models vs baselines
   - Statistical significance testing

5. **Results Compilation** (1 day)
   - Generate tables with CI
   - Create comparison charts
   - Write results section

6. **Paper Writing** (3-5 days)
   - Introduction & Related Work
   - Methodology description
   - Results & Discussion
   - Limitations & Future Work

## Sample Research Questions

Your paper could address:

1. **RQ1:** Do ML models outperform traditional technical analysis strategies for NSE stocks?
   - **Hypothesis:** ML models achieve significantly higher Sharpe ratios than MA crossover and RSI strategies.

2. **RQ2:** Does sentiment analysis improve prediction accuracy?
   - **Hypothesis:** Models with sentiment features outperform ablated versions without sentiment.

3. **RQ3:** Which model architecture performs best for multi-horizon prediction?
   - **Hypothesis:** LSTM outperforms tree-based models for long-term, but RF is better for intraday.

4. **RQ4:** How does performance vary across market sectors?
   - **Hypothesis:** IT sector stocks are more predictable than FMCG due to higher volatility.

## Tips for Publication

✅ **Do:**
- Report all metrics with confidence intervals
- Always compare against baselines
- Use proper statistical tests (p-values)
- Document all hyperparameters
- Make code publicly available
- Discuss limitations honestly

❌ **Don't:**
- Cherry-pick best results
- Report only on successful stocks
- Ignore multiple testing correction (Bonferroni)
- Overstate practical applicability
- Forget transaction costs in simulations

## Resources

- **Statistical Testing:** modules/statistical_tests.py
- **Baselines:** modules/baseline_models.py  
- **Full Experiments:** run_experiments.py
- **Research Checklist:** RESEARCH_READINESS.md
