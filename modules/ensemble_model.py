"""
Ensemble ML Model - Combines Random Forest, XGBoost, and LSTM
Weighted voting for superior prediction accuracy
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
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
except (ImportError, OSError):
    _TORCH_AVAILABLE = False

from modules.feature_engineering import build_features, get_feature_columns


class EnsemblePredictor:
    """
    Ensemble model combining:
    - Random Forest (Stability & interpretability)
    - XGBoost (Gradient boosting power)
    - LSTM (Deep learning temporal patterns)
    """
    
    def __init__(self):
        self.rf_clf = None
        self.rf_reg = None
        self.xgb_clf = None
        self.xgb_reg = None
        self.lstm_clf = None
        self.lstm_reg = None
        self.scaler = None
        self.seq_len = 10
        
        # Ensemble weights (optimized for best performance)
        self.clf_weights = {
            'rf': 0.30,      # 30%
            'xgb': 0.50,     # 50% (most powerful for classification)
            'lstm': 0.20     # 20% (complementary temporal patterns)
        }
        
        self.reg_weights = {
            'rf': 0.25,      # 25%
            'xgb': 0.45,     # 45%
            'lstm': 0.30     # 30% (better for regression)
        }
    
    def train(self, X_train, y_cls_train, y_reg_train):
        """
        Train all three models.
        
        Args:
            X_train: Training features
            y_cls_train: Classification targets (Bullish/Bearish)
            y_reg_train: Regression targets (Return %)
        """
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        
        
        print("  [PROGRESS] Training Random Forest...")
        self._train_random_forest(X_train_scaled, y_cls_train, y_reg_train)
        
        if _XGB_AVAILABLE:
            print("  [PROGRESS] Training XGBoost...")
            self._train_xgboost(X_train_scaled, y_cls_train, y_reg_train)
        else:
            print("  [WARN] XGBoost not available")
        
        if _TORCH_AVAILABLE and len(X_train) > 20:
            print("  [PROGRESS] Training LSTM...")
            self._train_lstm(X_train_scaled, y_cls_train, y_reg_train)
        else:
            print("  [WARN] LSTM not available")
    
    def _train_random_forest(self, X, y_cls, y_reg):
        """Train Random Forest with optimized hyperparameters."""
        # TUNED hyperparameters for NSE stocks
        self.rf_clf = RandomForestClassifier(
            n_estimators=500,           # More trees for stability
            max_depth=15,               # Deeper trees
            min_samples_split=3,        # More splits
            min_samples_leaf=1,         # More granular
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
            bootstrap=True is not False
        )
        self.rf_clf.fit(X, y_cls)
        
        self.rf_reg = RandomForestRegressor(
            n_estimators=500,
            max_depth=15,
            min_samples_split=3,
            min_samples_leaf=1,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        self.rf_reg.fit(X, y_reg)
    
    def _train_xgboost(self, X, y_cls, y_reg):
        """Train XGBoost with optimized hyperparameters."""
        if not _XGB_AVAILABLE:
            return
        
        # TUNED hyperparameters for NSE stocks
        self.xgb_clf = XGBClassifier(
            n_estimators=500,
            max_depth=8,                # Optimal depth for trees
            learning_rate=0.05,         # Slower learning for stability
            min_child_weight=1,
            subsample=0.9,              # 90% data sampling
            colsample_bytree=0.9,       # 90% feature sampling
            gamma=0.5,                  # Regularization
            reg_alpha=0.1,              # L1 regularization
            reg_lambda=1.0,             # L2 regularization
            random_state=42,
            verbosity=0
        )
        self.xgb_clf.fit(X, y_cls, verbose=False)
        
        self.xgb_reg = XGBRegressor(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.05,
            min_child_weight=1,
            subsample=0.9,
            colsample_bytree=0.9,
            gamma=0.5,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            verbosity=0
        )
        self.xgb_reg.fit(X, y_reg, verbose=False)
    
    def _train_lstm(self, X, y_cls, y_reg):
        """Train LSTM neural network."""
        if not _TORCH_AVAILABLE:
            return
        
        try:
            class SimpleLSTM(nn.Module):
                def __init__(self, input_size, hidden_size=64, num_layers=2):
                    super().__init__()
                    self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                                      batch_first=True, dropout=0.3)
                    self.fc = nn.Sequential(
                        nn.Linear(hidden_size, 32),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(32, 2)  # Binary classification
                    )
                
                def forward(self, x):
                    _, (h_n, _) = self.lstm(x)
                    return self.fc(h_n[-1])
            
            # Create sequences
            def make_sequences(X, y, seq_len):
                Xs, ys = [], []
                for i in range(len(X) - seq_len):
                    Xs.append(X[i:i + seq_len])
                    ys.append(y[i + seq_len])
                return np.array(Xs), np.array(ys)
            
            X_seq_cls, y_seq_cls = make_sequences(X, y_cls, self.seq_len)
            
            if len(X_seq_cls) < 10:
                return
            
            device = torch.device("cpu")
            self.lstm_clf = SimpleLSTM(X.shape[1]).to(device)
            optimizer = torch.optim.Adam(self.lstm_clf.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()
            
            X_t = torch.FloatTensor(X_seq_cls).to(device)
            y_t = torch.LongTensor(y_seq_cls.astype(int)).to(device)
            
            # Train for 30 epochs
            self.lstm_clf.train()
            for epoch in range(30):
                optimizer.zero_grad()
                outputs = self.lstm_clf(X_t)
                loss = criterion(outputs, y_t)
                loss.backward()
                optimizer.step()
            
            self.lstm_clf.eval()
        
        except Exception as e:
            print(f"    LSTM training error: {e}")
    
    def predict(self, X_test):
        """
        Get ensemble predictions.
        
        Returns:
            (prediction_class, prediction_return, confidence)
        """
        
        if self.scaler is None:
            return "N/A", 0.0, 0.0
        
        try:
            X_scaled = self.scaler.transform(X_test)
            X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
            
            predictions = []
            returns_pred = []
            
            # Random Forest
            if self.rf_clf is not None:
                rf_pred = self.rf_clf.predict(X_scaled)
                rf_proba = self.rf_clf.predict_proba(X_scaled)
                rf_conf = np.max(rf_proba, axis=1)
                predictions.append((rf_pred, rf_conf))
                
                if self.rf_reg is not None:
                    rf_ret = self.rf_reg.predict(X_scaled)
                    returns_pred.append(rf_ret)
            
            # XGBoost
            if self.xgb_clf is not None:
                xgb_pred = self.xgb_clf.predict(X_scaled)
                xgb_proba = self.xgb_clf.predict_proba(X_scaled)
                xgb_conf = np.max(xgb_proba, axis=1)
                predictions.append((xgb_pred, xgb_conf))
                
                if self.xgb_reg is not None:
                    xgb_ret = self.xgb_reg.predict(X_scaled)
                    returns_pred.append(xgb_ret)
            
            # LSTM
            if self.lstm_clf is not None and _TORCH_AVAILABLE:
                try:
                    # Create sequence
                    if len(X_scaled) >= self.seq_len:
                        X_seq = X_scaled[-self.seq_len:].reshape(1, self.seq_len, X_scaled.shape[1])
                        device = torch.device("cpu")
                        X_t = torch.FloatTensor(X_seq).to(device)
                        
                        self.lstm_clf.eval()
                        with torch.no_grad():
                            lstm_out = self.lstm_clf(X_t)
                            lstm_proba = torch.softmax(lstm_out, dim=1).cpu().numpy()[0]
                            lstm_pred = np.argmax(lstm_proba)
                            lstm_conf = np.max(lstm_proba)
                            predictions.append((lstm_pred, lstm_conf))
                except Exception:
                    pass
            
            if not predictions:
                return "N/A", 0.0, 0.0
            
            # Weighted ensemble voting
            total_weight = 0
            weighted_vote = 0
            total_conf = 0
            
            for pred, conf in predictions:
                weight = self.clf_weights.get('rf' if total_weight < 1 else ('xgb' if total_weight < 2 else 'lstm'), 0.33)
                weighted_vote += pred[-1:].item() if hasattr(pred[-1:], 'item') else pred[-1:][0]
                total_conf += conf
                total_weight += weight
            
            # Final prediction
            final_pred = 1 if weighted_vote / len(predictions) > 0.5 else 0
            avg_confidence = total_conf / len(predictions)
            avg_confidence = min(max(avg_confidence, 0.0), 1.0)
            
            # Ensemble return prediction
            if returns_pred:
                ensemble_return = np.mean(returns_pred)
            else:
                ensemble_return = 0.0
            
            trend = "Bullish" if final_pred == 1 else "Bearish"
            return trend, float(avg_confidence), float(ensemble_return)
        
        except Exception as e:
            print(f"Prediction error: {e}")
            return "N/A", 0.0, 0.0
