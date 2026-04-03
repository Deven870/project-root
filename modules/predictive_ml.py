# modules/predictive_ml.py
"""
ML-based stock prediction module.
Implements Random Forest, XGBoost, and LSTM models for:
  - Trend classification (Bullish / Bearish)
  - Next-period return regression

Uses walk-forward train/test split to avoid data leakage.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error
import joblib
import warnings
warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier, XGBRegressor
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except (ImportError, OSError, Exception):
    _TORCH_AVAILABLE = False

from modules.feature_engineering import build_features, get_feature_columns


# =========================================
# LSTM Model Definition (PyTorch)
# =========================================
if _TORCH_AVAILABLE:
    class LSTMClassifier(nn.Module):
        def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                                batch_first=True, dropout=dropout)
            self.fc = nn.Sequential(
                nn.Linear(hidden_size, 32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, 2),  # binary: Bearish=0, Bullish=1
            )

        def forward(self, x):
            # x shape: (batch, seq_len, features)
            lstm_out, _ = self.lstm(x)
            last_hidden = lstm_out[:, -1, :]
            return self.fc(last_hidden)

    class LSTMRegressor(nn.Module):
        def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                                batch_first=True, dropout=dropout)
            self.fc = nn.Sequential(
                nn.Linear(hidden_size, 32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, 1),
            )

        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            last_hidden = lstm_out[:, -1, :]
            return self.fc(last_hidden)


# =========================================
# Walk-forward Train/Test Split
# =========================================
def walk_forward_split(features_df: pd.DataFrame, train_ratio: float = 0.8):
    """
    Time-series aware split. No shuffling.
    Returns (X_train, y_cls_train, y_reg_train, X_test, y_cls_test, y_reg_test, scaler)
    """
    feature_cols = [c for c in get_feature_columns() if c in features_df.columns]
    if not feature_cols or features_df.empty:
        return None, None, None, None, None, None, None

    X = features_df[feature_cols].values
    y_cls = features_df["target_direction"].values
    y_reg = features_df["target_return"].values

    split_idx = int(len(X) * train_ratio)
    if split_idx < 10 or (len(X) - split_idx) < 5:
        return None, None, None, None, None, None, None

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_cls_train, y_cls_test = y_cls[:split_idx], y_cls[split_idx:]
    y_reg_train, y_reg_test = y_reg[:split_idx], y_reg[split_idx:]

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Replace any remaining NaN/inf
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    return X_train, y_cls_train, y_reg_train, X_test, y_cls_test, y_reg_test, scaler


# =========================================
# Model Training Functions
# =========================================

def train_random_forest(X_train, y_cls_train, y_reg_train):
    """Train Random Forest classifier and regressor (TUNED FOR PRODUCTION)."""
    # Optimized hyperparameters for better accuracy
    clf = RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_split=4,
        min_samples_leaf=2, max_features='sqrt', bootstrap=True,
        class_weight='balanced', random_state=42, n_jobs=-1, warm_start=False
    )
    clf.fit(X_train, y_cls_train)

    reg = RandomForestRegressor(
        n_estimators=300, max_depth=12, min_samples_split=4,
        min_samples_leaf=2, max_features='sqrt', bootstrap=True,
        random_state=42, n_jobs=-1, warm_start=False
    )
    reg.fit(X_train, y_reg_train)

    return clf, reg


def train_xgboost(X_train, y_cls_train, y_reg_train):
    """Train XGBoost classifier and regressor (TUNED FOR PRODUCTION)."""
    if not _XGB_AVAILABLE:
        return None, None

    # Optimized hyperparameters for better accuracy
    clf = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.03,
        min_child_weight=2, subsample=0.8, colsample_bytree=0.8,
        gamma=1, reg_alpha=0.5, reg_lambda=1.0,
        random_state=42, use_label_encoder=False, eval_metric="logloss",
        verbosity=0, scale_pos_weight=1
    )
    clf.fit(X_train, y_cls_train, verbose=False)

    reg = XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.03,
        min_child_weight=2, subsample=0.8, colsample_bytree=0.8,
        gamma=1, reg_alpha=0.5, reg_lambda=1.0,
        random_state=42, verbosity=0
    )
    reg.fit(X_train, y_reg_train, verbose=False)

    return clf, reg


def train_lstm(X_train, y_cls_train, y_reg_train, seq_len=10, epochs=50, lr=0.001):
    """Train LSTM classifier and regressor."""
    if not _TORCH_AVAILABLE:
        return None, None, None

    device = torch.device("cpu")

    # Create sequences
    def make_sequences(X, y, seq_len):
        Xs, ys = [], []
        for i in range(len(X) - seq_len):
            Xs.append(X[i : i + seq_len])
            ys.append(y[i + seq_len])
        return np.array(Xs), np.array(ys)

    X_seq_cls, y_seq_cls = make_sequences(X_train, y_cls_train, seq_len)
    X_seq_reg, y_seq_reg = make_sequences(X_train, y_reg_train, seq_len)

    if len(X_seq_cls) < 10:
        return None, None, seq_len

    n_features = X_train.shape[1]

    # --- Classifier ---
    clf_model = LSTMClassifier(n_features).to(device)
    clf_optimizer = torch.optim.Adam(clf_model.parameters(), lr=lr)
    clf_criterion = nn.CrossEntropyLoss()

    X_t = torch.FloatTensor(X_seq_cls).to(device)
    y_t = torch.LongTensor(y_seq_cls.astype(int)).to(device)

    clf_model.train()
    for epoch in range(epochs):
        clf_optimizer.zero_grad()
        outputs = clf_model(X_t)
        loss = clf_criterion(outputs, y_t)
        loss.backward()
        clf_optimizer.step()

    # --- Regressor ---
    reg_model = LSTMRegressor(n_features).to(device)
    reg_optimizer = torch.optim.Adam(reg_model.parameters(), lr=lr)
    reg_criterion = nn.MSELoss()

    X_t_r = torch.FloatTensor(X_seq_reg).to(device)
    y_t_r = torch.FloatTensor(y_seq_reg.astype(float)).unsqueeze(1).to(device)

    reg_model.train()
    for epoch in range(epochs):
        reg_optimizer.zero_grad()
        outputs = reg_model(X_t_r)
        loss = reg_criterion(outputs, y_t_r)
        loss.backward()
        reg_optimizer.step()

    return clf_model, reg_model, seq_len


# =========================================
# Prediction Functions
# =========================================

def predict_with_sklearn(clf, reg, X_test):
    """Get predictions from sklearn-style models."""
    y_cls_pred = clf.predict(X_test)
    y_cls_proba = clf.predict_proba(X_test)
    y_reg_pred = reg.predict(X_test)
    # Confidence = max class probability
    confidences = np.max(y_cls_proba, axis=1)
    return y_cls_pred, y_reg_pred, confidences


def predict_with_lstm(clf_model, reg_model, X_test, seq_len=10):
    """Get predictions from LSTM models."""
    if not _TORCH_AVAILABLE or clf_model is None:
        return None, None, None

    device = torch.device("cpu")
    clf_model.eval()
    reg_model.eval()

    # Create sequences from test data
    if len(X_test) < seq_len:
        return None, None, None

    X_seqs = []
    for i in range(len(X_test) - seq_len):
        X_seqs.append(X_test[i : i + seq_len])
    X_seqs = np.array(X_seqs)

    with torch.no_grad():
        X_t = torch.FloatTensor(X_seqs).to(device)
        cls_out = clf_model(X_t)
        proba = torch.softmax(cls_out, dim=1).numpy()
        y_cls_pred = np.argmax(proba, axis=1)
        confidences = np.max(proba, axis=1)

        reg_out = reg_model(X_t)
        y_reg_pred = reg_out.squeeze().numpy()

    return y_cls_pred, y_reg_pred, confidences


# =========================================
# Full Training Pipeline
# =========================================

def train_all_models(data: pd.DataFrame, sentiment_score: float = 0.0):
    """
    Train all models on a stock's OHLCV data.

    Returns a dict with model objects and evaluation metrics.
    """
    features_df = build_features(data, sentiment_score)
    if features_df.empty or len(features_df) < 30:
        return None

    split = walk_forward_split(features_df)
    X_train, y_cls_train, y_reg_train, X_test, y_cls_test, y_reg_test, scaler = split

    if X_train is None:
        return None

    results = {
        "features_df": features_df,
        "scaler": scaler,
        "X_test": X_test,
        "y_cls_test": y_cls_test,
        "y_reg_test": y_reg_test,
        "models": {},
    }

    # --- Random Forest ---
    try:
        rf_clf, rf_reg = train_random_forest(X_train, y_cls_train, y_reg_train)
        rf_cls_pred, rf_reg_pred, rf_conf = predict_with_sklearn(rf_clf, rf_reg, X_test)
        results["models"]["RandomForest"] = {
            "clf": rf_clf, "reg": rf_reg,
            "cls_pred": rf_cls_pred, "reg_pred": rf_reg_pred,
            "confidence": rf_conf,
            "accuracy": accuracy_score(y_cls_test, rf_cls_pred),
            "f1": f1_score(y_cls_test, rf_cls_pred, average="weighted", zero_division=0),
            "mae": mean_absolute_error(y_reg_test, rf_reg_pred),
            "rmse": np.sqrt(mean_squared_error(y_reg_test, rf_reg_pred)),
            "directional_accuracy": np.mean(
                (rf_reg_pred > 0) == (y_reg_test > 0)
            ),
        }
    except Exception as e:
        print(f"RF training error: {e}")

    # --- XGBoost ---
    if _XGB_AVAILABLE:
        try:
            xgb_clf, xgb_reg = train_xgboost(X_train, y_cls_train, y_reg_train)
            if xgb_clf is not None:
                xgb_cls_pred, xgb_reg_pred, xgb_conf = predict_with_sklearn(xgb_clf, xgb_reg, X_test)
                results["models"]["XGBoost"] = {
                    "clf": xgb_clf, "reg": xgb_reg,
                    "cls_pred": xgb_cls_pred, "reg_pred": xgb_reg_pred,
                    "confidence": xgb_conf,
                    "accuracy": accuracy_score(y_cls_test, xgb_cls_pred),
                    "f1": f1_score(y_cls_test, xgb_cls_pred, average="weighted", zero_division=0),
                    "mae": mean_absolute_error(y_reg_test, xgb_reg_pred),
                    "rmse": np.sqrt(mean_squared_error(y_reg_test, xgb_reg_pred)),
                    "directional_accuracy": np.mean(
                        (xgb_reg_pred > 0) == (y_reg_test > 0)
                    ),
                }
        except Exception as e:
            print(f"XGBoost training error: {e}")

    # --- LSTM ---
    if _TORCH_AVAILABLE:
        try:
            seq_len = min(10, len(X_train) // 3)
            if seq_len >= 3:
                lstm_clf, lstm_reg, seq_len = train_lstm(
                    X_train, y_cls_train, y_reg_train, seq_len=seq_len, epochs=50
                )
                if lstm_clf is not None:
                    lstm_cls_pred, lstm_reg_pred, lstm_conf = predict_with_lstm(
                        lstm_clf, lstm_reg, X_test, seq_len
                    )
                    if lstm_cls_pred is not None and len(lstm_cls_pred) > 0:
                        # LSTM predictions are shorter due to sequence windowing
                        y_cls_test_lstm = y_cls_test[seq_len:]
                        y_reg_test_lstm = y_reg_test[seq_len:]
                        min_len = min(len(lstm_cls_pred), len(y_cls_test_lstm))
                        results["models"]["LSTM"] = {
                            "clf": lstm_clf, "reg": lstm_reg,
                            "seq_len": seq_len,
                            "cls_pred": lstm_cls_pred[:min_len],
                            "reg_pred": lstm_reg_pred[:min_len],
                            "confidence": lstm_conf[:min_len],
                            "accuracy": accuracy_score(y_cls_test_lstm[:min_len], lstm_cls_pred[:min_len]),
                            "f1": f1_score(y_cls_test_lstm[:min_len], lstm_cls_pred[:min_len], average="weighted", zero_division=0),
                            "mae": mean_absolute_error(y_reg_test_lstm[:min_len], lstm_reg_pred[:min_len]),
                            "rmse": np.sqrt(mean_squared_error(y_reg_test_lstm[:min_len], lstm_reg_pred[:min_len])),
                            "directional_accuracy": np.mean(
                                (lstm_reg_pred[:min_len] > 0) == (y_reg_test_lstm[:min_len] > 0)
                            ),
                        }
        except Exception as e:
            print(f"LSTM training error: {e}")

    # --- Naive Baseline: always predict majority class ---
    majority = int(np.mean(y_cls_train) >= 0.5)
    baseline_pred = np.full_like(y_cls_test, majority)
    results["models"]["Baseline (Always Majority)"] = {
        "cls_pred": baseline_pred,
        "reg_pred": np.full_like(y_reg_test, np.mean(y_reg_train)),
        "confidence": np.full(len(y_cls_test), 0.5),
        "accuracy": accuracy_score(y_cls_test, baseline_pred),
        "f1": f1_score(y_cls_test, baseline_pred, average="weighted", zero_division=0),
        "mae": mean_absolute_error(y_reg_test, np.full_like(y_reg_test, np.mean(y_reg_train))),
        "rmse": np.sqrt(mean_squared_error(y_reg_test, np.full_like(y_reg_test, np.mean(y_reg_train)))),
        "directional_accuracy": np.mean(baseline_pred == (y_reg_test > 0).astype(int)),
    }

    return results


# =========================================
# Quick Prediction API (used by utils.py)
# =========================================

def predict_intraday(data: pd.DataFrame, sentiment_score: float = 0.0):
    """
    Quick prediction for a single stock (intraday horizon).
    Returns (trend_str, confidence_float)
    """
    return _quick_predict(data, sentiment_score)


def predict_long_term(data: pd.DataFrame, sentiment_score: float = 0.0):
    """
    Quick prediction for a single stock (long-term horizon).
    Returns (trend_str, confidence_float)
    """
    return _quick_predict(data, sentiment_score)


    # Import ensemble
    from modules.ensemble_model import EnsemblePredictor
    from modules.feature_engineering import build_features, get_feature_columns
    
    """
    Internal: trains best available model and returns latest prediction.
    For the app's real-time use. Uses ENSEMBLE for best accuracy.
    """
    if data is None or data.empty:
        return "N/A", 0.0

    try:
        features_df = build_features(data, sentiment_score)
        if features_df.empty or len(features_df) < 20:
            # Fallback to simple heuristic
            return _simple_fallback(data)

        split = walk_forward_split(features_df, train_ratio=0.85)
        X_train, y_cls_train, y_reg_train, X_test, y_cls_test, y_reg_test, scaler = split

        if X_train is None or len(X_train) < 10:
            return _simple_fallback(data)

        # Use ENSEMBLE (RF + XGBoost + LSTM)
        print("  [INFO] Training ensemble model (RF + XGBoost + LSTM)...")
        ensemble = EnsemblePredictor()
        ensemble.train(X_train, y_cls_train, y_reg_train)

        # Predict on the last row of test set (most recent)
        last_row = X_test[-1:] if len(X_test) > 0 else X_train[-1:]
        trend, confidence, predicted_return = ensemble.predict(last_row)
        
        return trend, confidence

    except Exception as e:
        print(f"Quick predict error: {e}")
        return _simple_fallback(data)


def _quick_predict_old(data: pd.DataFrame, sentiment_score: float = 0.0):


def _simple_fallback(data: pd.DataFrame):
    """Fallback heuristic when ML can't run (too little data)."""
    try:
        close_last = float(data["Close"].iloc[-1])
        open_last = float(data["Open"].iloc[-1])
        if close_last > open_last:
            return "Bullish", 0.55
        elif close_last < open_last:
            return "Bearish", 0.55
        else:
            return "Neutral", 0.5
    except Exception:
        return "N/A", 0.0
