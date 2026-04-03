"""
Improved Ensemble Model v2
Incorporates accuracy improvements:
- Feature selection (59 → 20 features)
- Balanced class weights
- XGBoost as primary estimator
- Optimal threshold tuning
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False

from modules.feature_engineering import build_features, get_feature_columns


class ImprovedEnsembleV2:
    """
    Enhanced ensemble with accuracy optimization techniques.
    Target: 52-55% accuracy (up from 50%)
    """
    
    def __init__(self):
        self.xgb_clf = None
        self.rf_clf = None
        self.scaler = None
        self.selector = None
        self.optimal_threshold = 0.5
        self.feature_indices = None
    
    def train(self, X_train, y_cls_train):
        """
        Train with all accuracy improvements.
        
        Args:
            X_train: Training features (n_samples, n_features)
            y_cls_train: Binary classification targets
        """
        
        print("  [ENSEMBLE V2] Improved model training...")
        
        # 1. Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_train_scaled = np.nan_to_num(X_train_scaled, nan=0, posinf=0, neginf=0)
        
        # 2. Select best features (59 → 20)
        print("  [FEATURES] Selecting top 20 features...")
        n_features = min(20, X_train.shape[1])
        self.selector = SelectKBest(f_classif, k=n_features)
        X_train_selected = self.selector.fit_transform(X_train_scaled, y_cls_train)
        self.feature_indices = self.selector.get_support(indices=True)
        
        # 3. Compute balanced class weights
        y_cls_train_clean = np.nan_to_num(y_cls_train, nan=0, posinf=0, neginf=0).astype(int)
        class_weights = compute_class_weight('balanced', classes=np.unique(y_cls_train_clean), y=y_cls_train_clean)
        class_weight_dict = {i: w for i, w in enumerate(class_weights)}
        
        print(f"  [WEIGHTS] Class balance: {class_weight_dict}")
        
        # 4. Train XGBoost (primary model - best performer)
        if _XGB_AVAILABLE:
            print("  [XGBOOST] Training primary model...")
            scale_pos_weight = class_weights[0] / max(class_weights[1], 1e-6)
            
            self.xgb_clf = XGBClassifier(
                n_estimators=300,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                scale_pos_weight=scale_pos_weight,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                verbosity=0
            )
            self.xgb_clf.fit(X_train_selected, y_cls_train_clean)
        else:
            print("  [WARN] XGBoost not available, using Random Forest")
            self.xgb_clf = RandomForestClassifier(
                n_estimators=300,
                class_weight=class_weight_dict,
                max_depth=12,
                random_state=42,
                n_jobs=-1
            )
            self.xgb_clf.fit(X_train_selected, y_cls_train_clean)
        
        # 5. Train Random Forest (secondary for ensemble)
        print("  [RANDOM FOREST] Training secondary model...")
        self.rf_clf = RandomForestClassifier(
            n_estimators=300,
            class_weight=class_weight_dict,
            max_depth=12,
            random_state=42,
            n_jobs=-1
        )
        self.rf_clf.fit(X_train_selected, y_cls_train_clean)
        
        print("  [ENSEMBLE V2] Training complete")
    
    def optimize_threshold(self, X_val, y_val):
        """
        Optimize classification threshold for F1 score.
        
        Args:
            X_val: Validation features
            y_val: Validation targets
        """
        
        if self.scaler is None or self.selector is None:
            return
        
        print("  [THRESHOLD] Optimizing classification boundary...")
        
        try:
            X_val_scaled = self.scaler.transform(X_val)
            X_val_scaled = np.nan_to_num(X_val_scaled, nan=0, posinf=0, neginf=0)
            X_val_selected = X_val_scaled[:, self.feature_indices]
            
            # Get probabilities from XGBoost
            y_proba = self.xgb_clf.predict_proba(X_val_selected)[:, 1]
            
            y_val_clean = np.nan_to_num(y_val, nan=0, posinf=0, neginf=0).astype(int)
            
            # Test thresholds
            best_f1 = 0
            best_threshold = 0.5
            
            for thresh in np.linspace(0.2, 0.8, 31):
                y_pred = (y_proba >= thresh).astype(int)
                
                if len(np.unique(y_pred)) < 2:
                    continue
                
                from sklearn.metrics import f1_score
                f1 = f1_score(y_val_clean, y_pred, zero_division=0)
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = thresh
            
            self.optimal_threshold = best_threshold
            print(f"  [THRESHOLD] Optimal: {best_threshold:.3f} (F1: {best_f1:.3f})")
        
        except Exception as e:
            print(f"  [WARN] Threshold optimization failed: {e}")
    
    def predict(self, X_test):
        """
        Make predictions with optimized threshold.
        
        Returns:
            (trend_str: "Bullish"/"Bearish", confidence: 0-1)
        """
        
        if self.scaler is None or self.xgb_clf is None:
            return "N/A", 0.0
        
        try:
            X_scaled = self.scaler.transform(X_test)
            X_scaled = np.nan_to_num(X_scaled, nan=0, posinf=0, neginf=0)
            
            # Apply feature selection
            if self.feature_indices is not None:
                X_selected = X_scaled[:, self.feature_indices]
            else:
                X_selected = X_scaled
            
            # Get predictions from XGBoost
            y_proba = self.xgb_clf.predict_proba(X_selected)[:, 1]
            
            # Get prediction from Random Forest for ensemble
            y_rf_proba = self.rf_clf.predict_proba(X_selected)[:, 1]
            
            # Ensemble: 70% XGBoost, 30% RF
            ensemble_proba = 0.7 * y_proba[-1] + 0.3 * y_rf_proba[-1]
            
            # Apply optimized threshold
            pred = 1 if ensemble_proba >= self.optimal_threshold else 0
            confidence = abs(ensemble_proba - 0.5) * 2  # Scale to 0-1
            confidence = min(max(confidence, 0), 1)
            
            trend = "Bullish" if pred == 1 else "Bearish"
            return trend, float(confidence)
        
        except Exception as e:
            print(f"  [ERROR] Prediction failed: {e}")
            return "N/A", 0.0


def create_improved_models(X_train, y_train, X_val=None, y_val=None):
    """
    Create and train improved models.
    
    Returns:
        ImprovedEnsembleV2 instance, ready for predictions
    """
    
    ensemble = ImprovedEnsembleV2()
    ensemble.train(X_train, y_train)
    
    if X_val is not None and y_val is not None:
        ensemble.optimize_threshold(X_val, y_val)
    
    return ensemble
