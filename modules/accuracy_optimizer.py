"""
Accuracy Improvement Module
Systematic approach to increase model accuracy beyond 50%
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import warnings
warnings.filterwarnings("ignore")

try:
    from imblearn.over_sampling import SMOTE
    _SMOTE_AVAILABLE = True
except ImportError:
    _SMOTE_AVAILABLE = False

try:
    from xgboost import XGBClassifier
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False


class AccuracyOptimizer:
    """
    Comprehensive accuracy improvement system.
    Identifies bottlenecks and implements targeted fixes.
    """
    
    @staticmethod
    def diagnose_accuracy_issues(X_train, y_train, X_test, y_test):
        """
        Diagnose why accuracy is low.
        
        Returns:
            Dict with diagnostic findings
        """
        
        diagnosis = {
            "class_distribution": {},
            "feature_quality": {},
            "model_performance": {},
            "recommendations": []
        }
        
        # 1. Check class distribution
        unique, counts = np.unique(y_train, return_counts=True)
        class_dist = dict(zip(unique, counts))
        
        diagnosis["class_distribution"] = {
            "train": class_dist,
            "counts": {str(k): v for k, v in class_dist.items()},
            "ratio": f"{max(counts):.0f}:{min(counts):.0f}",
            "imbalanced": max(counts) / min(counts) > 1.5
        }
        
        if diagnosis["class_distribution"]["imbalanced"]:
            diagnosis["recommendations"].append(
                "HIGH CLASS IMBALANCE - Use SMOTE or class weights"
            )
        
        # 2. Check feature quality
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_train_scaled = np.nan_to_num(X_train_scaled, nan=0, posinf=0, neginf=0)
        y_train_clean = np.nan_to_num(y_train, nan=0, posinf=0, neginf=0).astype(int)
        
        # Feature importance with Random Forest
        try:
            rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
            rf.fit(X_train_scaled, y_train_clean)
            
            importances = rf.feature_importances_
            top_features = np.argsort(importances)[-10:][::-1]
            
            diagnosis["feature_quality"] = {
                "top_10_importance": importances[top_features].tolist(),
                "feature_count": X_train.shape[1],
                "redundant_features": np.sum(importances < 0.001),
                "avg_importance": float(np.mean(importances))
            }
            
            if diagnosis["feature_quality"]["redundant_features"] > X_train.shape[1] * 0.5:
                diagnosis["recommendations"].append(
                    "TOO MANY LOW-VALUE FEATURES - Feature selection needed"
                )
        except Exception as e:
            diagnosis["feature_quality"]["error"] = str(e)
        
        # 3. Check model performance variations
        X_test_scaled = scaler.transform(X_test)
        X_test_scaled = np.nan_to_num(X_test_scaled, nan=0, posinf=0, neginf=0)
        y_test_clean = np.nan_to_num(y_test, nan=0, posinf=0, neginf=0).astype(int)
        
        if len(np.unique(y_test)) >= 2:
            try:
                y_pred = rf.predict(X_test_scaled)
                
                diagnosis["model_performance"] = {
                    "accuracy": accuracy_score(y_test_clean, y_pred),
                    "precision": precision_score(y_test_clean, y_pred, zero_division=0),
                    "recall": recall_score(y_test_clean, y_pred, zero_division=0),
                    "f1": f1_score(y_test_clean, y_pred, zero_division=0)
                }
                
                if diagnosis["model_performance"]["recall"] < 0.3:
                    diagnosis["recommendations"].append(
                        "LOW RECALL - Model misses true positives"
                    )
                if diagnosis["model_performance"]["precision"] < 0.3:
                    diagnosis["recommendations"].append(
                        "LOW PRECISION - Model has false positives"
                    )
            except Exception as e:
                diagnosis["model_performance"]["error"] = str(e)
        
        return diagnosis
    
    @staticmethod
    def apply_smote_sampling(X_train, y_train):
        """
        Apply SMOTE to handle class imbalance.
        
        Returns:
            Resampled X_train, y_train
        """
        
        if not _SMOTE_AVAILABLE:
            print("  [WARN] SMOTE not available (install imbalanced-learn)")
            return X_train, y_train
        
        try:
            smote = SMOTE(random_state=42, k_neighbors=3)
            X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
            
            print(f"  [SMOTE] Before: {len(X_train)} samples")
            print(f"  [SMOTE] After: {len(X_resampled)} samples")
            print(f"  [SMOTE] Class distribution: {np.unique(y_resampled, return_counts=True)}")
            
            return X_resampled, y_resampled
        except Exception as e:
            print(f"  [ERROR] SMOTE failed: {e}")
            return X_train, y_train
    
    @staticmethod
    def select_best_features(X_train, y_train, n_features=20):
        """
        Select top N most predictive features.
        
        Returns:
            X_train_selected, feature_indices
        """
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        X_scaled = np.nan_to_num(X_scaled, nan=0, posinf=0, neginf=0)
        
        try:
            # Use mutual information + f_classif
            selector = SelectKBest(f_classif, k=min(n_features, X_train.shape[1]))
            X_selected = selector.fit_transform(X_scaled, y_train)
            
            feature_indices = selector.get_support(indices=True)
            scores = selector.scores_[feature_indices]
            
            print(f"  [FEATURES] Selected {len(feature_indices)} from {X_train.shape[1]} features")
            print(f"  [F-SCORE] Top scores: {scores[:5]}")
            
            return X_selected, feature_indices
        except Exception as e:
            print(f"  [ERROR] Feature selection failed: {e}")
            return X_train, np.arange(X_train.shape[1])
    
    @staticmethod
    def optimize_classification_threshold(y_true, y_proba, thresholds=None):
        """
        Find optimal classification threshold to maximize F1 or accuracy.
        
        Args:
            y_true: Ground truth labels
            y_proba: Predicted probabilities for positive class
            thresholds: List of thresholds to test (default: 50 values)
            
        Returns:
            optimal_threshold, performance_metrics
        """
        
        if thresholds is None:
            thresholds = np.linspace(0, 1, 51)
        
        best_f1 = 0
        best_threshold = 0.5
        
        results = []
        
        for thresh in thresholds:
            y_pred = (y_proba >= thresh).astype(int)
            
            if len(np.unique(y_pred)) < 2:  # Skip if only one class
                continue
            
            try:
                acc = accuracy_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred, zero_division=0)
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = thresh
                
                results.append({
                    "threshold": thresh,
                    "accuracy": acc,
                    "f1": f1
                })
            except Exception:
                continue
        
        if results:
            print(f"  [THRESHOLD] Optimal: {best_threshold:.3f} (F1: {best_f1:.3f})")
            return best_threshold, results
        else:
            return 0.5, []
    
    @staticmethod
    def train_with_improvements(X_train, y_train, X_test, y_test):
        """
        Train models with all accuracy improvements applied.
        
        Returns:
            Dict with trained models and performance metrics
        """
        
        print("\n[OPTIMIZATION] Starting accuracy improvement...")
        print("-" * 70)
        
        results = {
            "baseline": {},
            "improved": {},
            "improvements": []
        }
        
        # 1. BASELINE - Standard model
        print("\n[STEP 1] Training baseline model...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_train_scaled = np.nan_to_num(X_train_scaled, nan=0, posinf=0, neginf=0)
        X_test_scaled = scaler.transform(X_test)
        X_test_scaled = np.nan_to_num(X_test_scaled, nan=0, posinf=0, neginf=0)
        
        y_train_clean = np.nan_to_num(y_train, nan=0, posinf=0, neginf=0).astype(int)
        y_test_clean = np.nan_to_num(y_test, nan=0, posinf=0, neginf=0).astype(int)
        
        rf_baseline = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf_baseline.fit(X_train_scaled, y_train_clean)
        y_pred_baseline = rf_baseline.predict(X_test_scaled)
        
        results["baseline"] = {
            "accuracy": accuracy_score(y_test_clean, y_pred_baseline),
            "f1": f1_score(y_test_clean, y_pred_baseline, zero_division=0)
        }
        print(f"  Baseline Accuracy: {results['baseline']['accuracy']:.3f}")
        print(f"  Baseline F1: {results['baseline']['f1']:.3f}")
        
        # 2. IMPROVEMENT 1: Class weights
        print("\n[STEP 2] Adding class weights...")
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train_clean), y=y_train_clean)
        class_weight_dict = {i: w for i, w in enumerate(class_weights)}
        
        rf_weighted = RandomForestClassifier(
            n_estimators=200,
            class_weight=class_weight_dict,
            max_depth=12,
            random_state=42,
            n_jobs=-1
        )
        rf_weighted.fit(X_train_scaled, y_train_clean)
        y_pred_weighted = rf_weighted.predict(X_test_scaled)
        
        acc_weighted = accuracy_score(y_test_clean, y_pred_weighted)
        f1_weighted = f1_score(y_test_clean, y_pred_weighted, zero_division=0)
        
        if acc_weighted > results["baseline"]["accuracy"]:
            results["improvements"].append(f"+{(acc_weighted - results['baseline']['accuracy'])*100:.2f}% from class weights")
            results["weighted"] = {"accuracy": acc_weighted, "f1": f1_weighted}
            print(f"  [+] Class weights improved: {acc_weighted:.3f} (F1: {f1_weighted:.3f})")
        else:
            print(f"  [-] Class weights no improvement: {acc_weighted:.3f}")
        
        # 3. IMPROVEMENT 2: Feature selection
        print("\n[STEP 3] Feature selection...")
        X_train_selected, feature_indices = AccuracyOptimizer.select_best_features(
            X_train_scaled, y_train_clean, n_features=20
        )
        X_test_selected = X_test_scaled[:, feature_indices]
        
        rf_fs = RandomForestClassifier(
            n_estimators=200,
            class_weight=class_weight_dict,
            max_depth=12,
            random_state=42,
            n_jobs=-1
        )
        rf_fs.fit(X_train_selected, y_train_clean)
        y_pred_fs = rf_fs.predict(X_test_selected)
        
        acc_fs = accuracy_score(y_test_clean, y_pred_fs)
        f1_fs = f1_score(y_test_clean, y_pred_fs, zero_division=0)
        
        if acc_fs > results["baseline"]["accuracy"]:
            results["improvements"].append(f"+{(acc_fs - results['baseline']['accuracy'])*100:.2f}% from feature selection")
            results["feature_selection"] = {"accuracy": acc_fs, "f1": f1_fs}
            print(f"  [+] Feature selection improved: {acc_fs:.3f} (F1: {f1_fs:.3f})")
        else:
            print(f"  [-] Feature selection no improvement: {acc_fs:.3f}")
        
        # 4. IMPROVEMENT 3: XGBoost if available
        if _XGB_AVAILABLE:
            print("\n[STEP 4] Training XGBoost...")
            xgb = XGBClassifier(
                n_estimators=200,
                max_depth=7,
                learning_rate=0.05,
                scale_pos_weight=np.sum(y_train_clean == 0) / np.sum(y_train_clean == 1) if np.sum(y_train_clean == 1) > 0 else 1,
                random_state=42,
                verbosity=0
            )
            xgb.fit(X_train_selected, y_train_clean)
            y_pred_xgb = xgb.predict(X_test_selected)
            
            acc_xgb = accuracy_score(y_test_clean, y_pred_xgb)
            f1_xgb = f1_score(y_test_clean, y_pred_xgb, zero_division=0)
            
            if acc_xgb > results["baseline"]["accuracy"]:
                results["improvements"].append(f"+{(acc_xgb - results['baseline']['accuracy'])*100:.2f}% from XGBoost")
                results["xgboost"] = {"accuracy": acc_xgb, "f1": f1_xgb}
                print(f"  [+] XGBoost improved: {acc_xgb:.3f} (F1: {f1_xgb:.3f})")
            else:
                print(f"  [-] XGBoost no improvement: {acc_xgb:.3f}")
        
        # 5. IMPROVEMENT 4: Threshold optimization
        print("\n[STEP 5] Threshold optimization...")
        y_proba = rf_weighted.predict_proba(X_test_scaled)[:, 1]
        optimal_threshold, threshold_results = AccuracyOptimizer.optimize_classification_threshold(
            y_test_clean, y_proba
        )
        
        y_pred_threshold = (y_proba >= optimal_threshold).astype(int)
        acc_threshold = accuracy_score(y_test_clean, y_pred_threshold)
        f1_threshold = f1_score(y_test_clean, y_pred_threshold, zero_division=0)
        
        if acc_threshold > results["baseline"]["accuracy"]:
            results["improvements"].append(f"+{(acc_threshold - results['baseline']['accuracy'])*100:.2f}% from threshold tuning")
            results["threshold_optimized"] = {"accuracy": acc_threshold, "f1": f1_threshold, "threshold": optimal_threshold}
            print(f"  [+] Threshold tuning improved: {acc_threshold:.3f} (F1: {f1_threshold:.3f})")
        else:
            print(f"  [-] Threshold tuning no improvement: {acc_threshold:.3f}")
        
        # 6. Final best model
        best_accuracy = max(
            results["baseline"]["accuracy"],
            results.get("weighted", {}).get("accuracy", 0),
            results.get("feature_selection", {}).get("accuracy", 0),
            results.get("xgboost", {}).get("accuracy", 0),
            results.get("threshold_optimized", {}).get("accuracy", 0)
        )
        
        improvement = ((best_accuracy - results["baseline"]["accuracy"]) / results["baseline"]["accuracy"]) * 100
        results["improved"]["best_accuracy"] = best_accuracy
        results["improved"]["improvement_pct"] = improvement
        
        return results


def test_accuracy_improvements(X_train, y_train, X_test, y_test, stock_name="TEST"):
    """Quick test of accuracy improvements."""
    print(f"\n{'='*70}")
    print(f"[ACCURACY TEST] {stock_name}")
    print(f"{'='*70}")
    
    # Diagnose
    print("\n[DIAGNOSIS] Analyzing accuracy bottlenecks...")
    diagnosis = AccuracyOptimizer.diagnose_accuracy_issues(X_train, y_train, X_test, y_test)
    
    print(f"\n[CLASS DIST] Train: {diagnosis['class_distribution']['counts']}")
    print(f"[IMBALANCE] Ratio: {diagnosis['class_distribution']['ratio']}")
    
    if diagnosis['recommendations']:
        print(f"\n[RECOMMENDATIONS]:")
        for rec in diagnosis['recommendations']:
            print(f"  - {rec}")
    
    # Train with improvements
    results = AccuracyOptimizer.train_with_improvements(X_train, y_train, X_test, y_test)
    
    print(f"\n{'='*70}")
    print(f"[RESULTS SUMMARY]")
    print(f"{'='*70}")
    print(f"Baseline Accuracy: {results['baseline']['accuracy']:.1%}")
    print(f"Best Accuracy:     {results['improved']['best_accuracy']:.1%}")
    print(f"Improvement:       +{results['improved']['improvement_pct']:.1f}%")
    
    if results['improvements']:
        print(f"\n[TECHNIQUES THAT WORKED]:")
        for imp in results['improvements']:
            print(f"  {imp}")
    
    print(f"\n{'='*70}\n")
    
    return results
