import numpy as np
import pandas as pd
from scipy.stats import randint, loguniform
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import (
    roc_auc_score, average_precision_score, classification_report,
    confusion_matrix, precision_recall_curve
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

from base import compute_features, FILES, H

def best_threshold_by_f1(y_true, y_prob):
    """
    Return the best threshold for a given set of predictions, y_prob, which maximizes the F1-score.
    
    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        True labels.
    y_prob : array-like, shape (n_samples,)
        Predicted probabilities.
    
    Returns
    -------
    thr : float
        The best threshold.
    f1 : float
        The best F1-score.
    """
    P, R, T = precision_recall_curve(y_true, y_prob)
    if len(T) == 0:
        return 0.5, 0.0
    F1 = (2 * P * R / (P + R + 1e-12))[:-1]
    ix = int(np.argmax(F1))
    return float(T[ix]), float(F1[ix])

def time_cv_with_positives(X, y, n_splits=5, gap=H, min_pos=1):
    """
    Return TimeSeriesSplit CV folds which contain positives in both train and validation.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target vector.
    n_splits : int, optional
        Number of CV folds. Default is 5.
    gap : int, optional
        Gap, in samples, between train and validation. Default is H.
    min_pos : int, optional
        Minimum number of positives in both train and validation. Default is 1.

    Returns
    -------
    splits : list of tuples
        List of train and validation indices for each CV fold.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    splits = []
    for tr, va in tscv.split(X):
        y_tr, y_va = y.iloc[tr], y.iloc[va]
        if (y_tr.nunique() == 2) and (y_va.nunique() == 2) and (y_tr.sum() >= min_pos) and (y_va.sum() >= min_pos):
            splits.append((tr, va))
    if not splits:
        raise RuntimeError("No valid CV folds contain positives in train and validation.")
    return splits

def report_metrics(name, y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    base = y_true.mean()
    print(f"\n=== {name} @ thr={threshold:.3f} ===")
    print(f"ROC-AUC: {roc_auc_score(y_true, y_prob):.3f}")
    print(f"PR-AUC:  {average_precision_score(y_true, y_prob):.3f}  (baseline={base:.3f})")
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))
    print("Classification report:\n", classification_report(y_true, y_pred, digits=3))

def run_search(name, estimator, search_space, X_tr, y_tr, cv_splits, n_iter, random_state):
    print(f"\n[{name} Tuning]")
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=search_space,
        n_iter=n_iter,
        scoring="average_precision",
        cv=cv_splits,
        refit=True,
        verbose=0,
        random_state=random_state,
        n_jobs=-1,
        error_score="raise",
    )
    search.fit(X_tr, y_tr)
    print(f"[{name} Tuning] Best params:")
    print(search.best_params_)
    print(f"[{name} Tuning] Best CV PR-AUC: {search.best_score_:.3f}")
    return search

def main(n_iter=60, random_state=42):
    X, y = compute_features(FILES, horizon=H)
    n = len(X)
    i_tr = int(n * 0.6)
    i_va = int(n * 0.8)
    X_tr, y_tr = X.iloc[:i_tr], y.iloc[:i_tr]
    X_va, y_va = X.iloc[i_tr:i_va], y.iloc[i_tr:i_va]
    X_te, y_te = X.iloc[i_va:], y.iloc[i_va:]

    print(f"Samples: total={n} | train={len(X_tr)} | val={len(X_va)} | test={len(X_te)}")
    print(f"Positives by split: train={y_tr.sum()}, val={y_va.sum()}, test={y_te.sum()}")

    # Time-aware CV folds with at least one positive in train/val
    cv_splits = time_cv_with_positives(X_tr.reset_index(drop=True), y_tr.reset_index(drop=True), n_splits=5, gap=H)

    results = []

    # Logistic Regression search space (scaled)
    logit_base = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight="balanced",
            max_iter=5000,
            solver="lbfgs",
            random_state=random_state,
        )),
    ])
    logit_space = {"clf__C": loguniform(1e-3, 10)}
    logit_search = run_search("Logit", logit_base, logit_space, X_tr, y_tr, cv_splits, n_iter, random_state)
    logit_best = logit_search.best_estimator_
    logit_cal = CalibratedClassifierCV(logit_best, method="sigmoid", cv="prefit")
    logit_cal.fit(X_va, y_va)
    p_val_log = logit_cal.predict_proba(X_va)[:, 1]
    thr_log, f1_log = best_threshold_by_f1(y_va, p_val_log)
    p_test_log = logit_cal.predict_proba(X_te)[:, 1]
    print(f"\n[Logit] Validation PR-AUC: {average_precision_score(y_va, p_val_log):.3f} | "
          f"Best-F1 thr: {thr_log:.3f} (F1={f1_log:.3f})")
    report_metrics("Logit (tuned)", y_te, p_test_log, thr_log)
    results.append({"model": "Logit", "val_pr": average_precision_score(y_va, p_val_log),
                    "thr": thr_log, "test_pr": average_precision_score(y_te, p_test_log)})

    # Logistic Regression + PCA search space
    pca_logit_base = Pipeline([
        ("scale", StandardScaler()),
        ("pca", PCA(svd_solver="full")),
        ("clf", LogisticRegression(
            class_weight="balanced",
            max_iter=5000,
            solver="lbfgs",
            random_state=random_state,
        )),
    ])
    pca_logit_space = {
        "pca__n_components": [0.7, 0.8, 0.9, 0.95],
        "clf__C": loguniform(1e-3, 10),
    }
    pca_search = run_search("Logit+PCA", pca_logit_base, pca_logit_space, X_tr, y_tr, cv_splits, n_iter, random_state)
    pca_best = pca_search.best_estimator_
    pca_cal = CalibratedClassifierCV(pca_best, method="sigmoid", cv="prefit")
    pca_cal.fit(X_va, y_va)
    p_val_pca = pca_cal.predict_proba(X_va)[:, 1]
    thr_pca, f1_pca = best_threshold_by_f1(y_va, p_val_pca)
    p_test_pca = pca_cal.predict_proba(X_te)[:, 1]
    print(f"\n[Logit+PCA] Validation PR-AUC: {average_precision_score(y_va, p_val_pca):.3f} | "
          f"Best-F1 thr: {thr_pca:.3f} (F1={f1_pca:.3f})")
    report_metrics("Logit+PCA (tuned)", y_te, p_test_pca, thr_pca)
    results.append({"model": "Logit+PCA", "val_pr": average_precision_score(y_va, p_val_pca),
                    "thr": thr_pca, "test_pr": average_precision_score(y_te, p_test_pca)})

    # RF search space
    rf_base = RandomForestClassifier(
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1
    )
    rf_space = {
        "n_estimators": randint(300, 2000),
        "max_depth": randint(3, 12),
        "min_samples_leaf": randint(1, 40),
        "min_samples_split": randint(2, 40),
        "max_features": ["sqrt", "log2", None],
        "bootstrap": [True, False],
    }
    rf_search = run_search("RF", rf_base, rf_space, X_tr, y_tr, cv_splits, n_iter, random_state)
    rf_best = rf_search.best_estimator_
    rf_cal = CalibratedClassifierCV(rf_best, method="sigmoid", cv="prefit")
    rf_cal.fit(X_va, y_va)
    p_val_rf = rf_cal.predict_proba(X_va)[:, 1]
    thr_rf, f1_rf = best_threshold_by_f1(y_va, p_val_rf)
    p_test_rf = rf_cal.predict_proba(X_te)[:, 1]
    print(f"\n[RF] Validation PR-AUC: {average_precision_score(y_va, p_val_rf):.3f} | "
          f"Best-F1 thr: {thr_rf:.3f} (F1={f1_rf:.3f})")
    report_metrics("Random Forest (tuned)", y_te, p_test_rf, thr_rf)
    results.append({"model": "RF", "val_pr": average_precision_score(y_va, p_val_rf),
                    "thr": thr_rf, "test_pr": average_precision_score(y_te, p_test_rf)})

    # XGBoost search space
    pos, neg = int((y_tr==1).sum()), int((y_tr==0).sum())
    xgb_base = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        scale_pos_weight=neg/max(pos,1),
        random_state=random_state,
        n_jobs=-1,
        early_stopping_rounds=None,  # handled via CV, not inside the search
    )
    xgb_space = {
        "n_estimators": randint(200, 800),
        "learning_rate": [0.01, 0.02, 0.05, 0.1],
        "max_depth": randint(2, 6),
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "min_child_weight": [1, 3, 5, 10],
        "gamma": [0, 0.1, 0.3],
        "reg_lambda": [0.5, 1.0, 2.0],
    }
    xgb_search = run_search("XGB", xgb_base, xgb_space, X_tr, y_tr, cv_splits, n_iter, random_state)
    xgb_best = xgb_search.best_estimator_
    xgb_cal = CalibratedClassifierCV(xgb_best, method="sigmoid", cv="prefit")
    xgb_cal.fit(X_va, y_va)
    p_val_xgb = xgb_cal.predict_proba(X_va)[:, 1]
    thr_xgb, f1_xgb = best_threshold_by_f1(y_va, p_val_xgb)
    p_test_xgb = xgb_cal.predict_proba(X_te)[:, 1]
    print(f"\n[XGB] Validation PR-AUC: {average_precision_score(y_va, p_val_xgb):.3f} | "
          f"Best-F1 thr: {thr_xgb:.3f} (F1={f1_xgb:.3f})")
    report_metrics("XGBoost (tuned)", y_te, p_test_xgb, thr_xgb)
    results.append({"model": "XGB", "val_pr": average_precision_score(y_va, p_val_xgb),
                    "thr": thr_xgb, "test_pr": average_precision_score(y_te, p_test_xgb)})

    # Stacking (RF + XGB -> Logistic) using tuned bases
    stack_rf = RandomForestClassifier(**rf_best.get_params())
    stack_xgb = XGBClassifier(**xgb_best.get_params())
    stack_base = StackingClassifier(
        estimators=[("rf", stack_rf), ("xgb", stack_xgb)],
        final_estimator=LogisticRegression(
            class_weight="balanced", max_iter=5000, solver="lbfgs", random_state=random_state
        ),
        stack_method="predict_proba",
        n_jobs=-1,
    )
    stack_space = {"final_estimator__C": loguniform(1e-3, 10)}
    stack_search = run_search("Stacking", stack_base, stack_space, X_tr, y_tr, cv_splits, n_iter, random_state)
    stack_best = stack_search.best_estimator_
    stack_cal = CalibratedClassifierCV(stack_best, method="sigmoid", cv="prefit")
    stack_cal.fit(X_va, y_va)
    p_val_stack = stack_cal.predict_proba(X_va)[:, 1]
    thr_stack, f1_stack = best_threshold_by_f1(y_va, p_val_stack)
    p_test_stack = stack_cal.predict_proba(X_te)[:, 1]
    print(f"\n[Stacking] Validation PR-AUC: {average_precision_score(y_va, p_val_stack):.3f} | "
          f"Best-F1 thr: {thr_stack:.3f} (F1={f1_stack:.3f})")
    report_metrics("Stacking (tuned)", y_te, p_test_stack, thr_stack)
    results.append({"model": "Stacking", "val_pr": average_precision_score(y_va, p_val_stack),
                    "thr": thr_stack, "test_pr": average_precision_score(y_te, p_test_stack)})

    print("\n[Summary] Validation/Test PR-AUC and thresholds")
    df_res = pd.DataFrame(results)
    print(df_res)
    best = df_res.sort_values("val_pr", ascending=False).iloc[0]
    print(f"\n[Choice] Select '{best['model']}' based on highest validation PR-AUC ({best['val_pr']:.3f}); test PR-AUC={best['test_pr']:.3f}, thr={best['thr']:.3f}")

    try:
        pd.DataFrame(logit_search.cv_results_).to_csv("tuning/logit_tuning_cv_results.csv", index=False)
        pd.DataFrame(pca_search.cv_results_).to_csv("tuning/logit_pca_tuning_cv_results.csv", index=False)
        pd.DataFrame(rf_search.cv_results_).to_csv("tuning/rf_tuning_cv_results.csv", index=False)
        pd.DataFrame(xgb_search.cv_results_).to_csv("tuning/xgb_tuning_cv_results.csv", index=False)
        pd.DataFrame(stack_search.cv_results_).to_csv("tuning/stacking_tuning_cv_results.csv", index=False)
        importances = pd.Series(rf_best.feature_importances_, index=X_tr.columns).sort_values(ascending=False)
        importances.to_csv("tuning/rf_tuned_feature_importance.csv")
        try:
            xgb_importance = pd.Series(xgb_best.feature_importances_, index=X_tr.columns).sort_values(ascending=False)
            xgb_importance.to_csv("tuning/xgb_tuned_feature_importance.csv")
        except Exception:
            pass
        print("\nSaved tuning results and feature importances to tuning/.")
    except Exception as e:
        print(f"(Skip saving artifacts) {e}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-iter", type=int, default=60, help="Number of random search iterations")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    args = ap.parse_args()
    main(n_iter=args.n_iter, random_state=args.seed)


"""
OUTPUT:
Samples: total=505 | train=303 | val=101 | test=101
Positives by split: train=29, val=8, test=3

[Logit Tuning]
[Logit Tuning] Best params:
{'clf__C': np.float64(0.005337032762603957)}
[Logit Tuning] Best CV PR-AUC: 0.850
/Users/benjaminemily/Desktop/ESILV/S7/ML/Project/venv/lib/python3.11/site-packages/sklearn/calibration.py:330: FutureWarning: The `cv='prefit'` option is deprecated in 1.6 and will be removed in 1.8. You can use CalibratedClassifierCV(FrozenEstimator(estimator)) instead.
  warnings.warn(

[Logit] Validation PR-AUC: 0.840 | Best-F1 thr: 0.687 (F1=0.769)

=== Logit (tuned) @ thr=0.687 ===
ROC-AUC: 0.677
PR-AUC:  0.060  (baseline=0.030)
Confusion matrix:
 [[71 27]
 [ 1  2]]
Classification report:
               precision    recall  f1-score   support

           0      0.986     0.724     0.835        98
           1      0.069     0.667     0.125         3

    accuracy                          0.723       101
   macro avg      0.528     0.696     0.480       101
weighted avg      0.959     0.723     0.814       101


[Logit+PCA Tuning]
[Logit+PCA Tuning] Best params:
{'clf__C': np.float64(0.4609877941534894), 'pca__n_components': 0.95}
[Logit+PCA Tuning] Best CV PR-AUC: 0.856
/Users/benjaminemily/Desktop/ESILV/S7/ML/Project/venv/lib/python3.11/site-packages/sklearn/calibration.py:330: FutureWarning: The `cv='prefit'` option is deprecated in 1.6 and will be removed in 1.8. You can use CalibratedClassifierCV(FrozenEstimator(estimator)) instead.
  warnings.warn(

[Logit+PCA] Validation PR-AUC: 0.624 | Best-F1 thr: 0.395 (F1=0.667)

=== Logit+PCA (tuned) @ thr=0.395 ===
ROC-AUC: 0.724
PR-AUC:  0.066  (baseline=0.030)
Confusion matrix:
 [[55 43]
 [ 0  3]]
Classification report:
               precision    recall  f1-score   support

           0      1.000     0.561     0.719        98
           1      0.065     1.000     0.122         3

    accuracy                          0.574       101
   macro avg      0.533     0.781     0.421       101
weighted avg      0.972     0.574     0.701       101


[RF Tuning]
[RF Tuning] Best params:
{'bootstrap': True, 'max_depth': 3, 'max_features': 'sqrt', 'min_samples_leaf': 27, 'min_samples_split': 14, 'n_estimators': 1582}
[RF Tuning] Best CV PR-AUC: 0.805
/Users/benjaminemily/Desktop/ESILV/S7/ML/Project/venv/lib/python3.11/site-packages/sklearn/calibration.py:330: FutureWarning: The `cv='prefit'` option is deprecated in 1.6 and will be removed in 1.8. You can use CalibratedClassifierCV(FrozenEstimator(estimator)) instead.
  warnings.warn(

[RF] Validation PR-AUC: 0.975 | Best-F1 thr: 0.560 (F1=0.933)

=== Random Forest (tuned) @ thr=0.560 ===
ROC-AUC: 0.721
PR-AUC:  0.067  (baseline=0.030)
Confusion matrix:
 [[44 54]
 [ 0  3]]
Classification report:
               precision    recall  f1-score   support

           0      1.000     0.449     0.620        98
           1      0.053     1.000     0.100         3

    accuracy                          0.465       101
   macro avg      0.526     0.724     0.360       101
weighted avg      0.972     0.465     0.604       101


[XGB Tuning]
[XGB Tuning] Best params:
{'colsample_bytree': 0.8, 'gamma': 0, 'learning_rate': 0.1, 'max_depth': 2, 'min_child_weight': 3, 'n_estimators': 696, 'reg_lambda': 1.0, 'subsample': 0.8}
[XGB Tuning] Best CV PR-AUC: 0.677
/Users/benjaminemily/Desktop/ESILV/S7/ML/Project/venv/lib/python3.11/site-packages/sklearn/calibration.py:330: FutureWarning: The `cv='prefit'` option is deprecated in 1.6 and will be removed in 1.8. You can use CalibratedClassifierCV(FrozenEstimator(estimator)) instead.
  warnings.warn(

[XGB] Validation PR-AUC: 0.858 | Best-F1 thr: 0.213 (F1=0.857)

=== XGBoost (tuned) @ thr=0.213 ===
ROC-AUC: 0.636
PR-AUC:  0.051  (baseline=0.030)
Confusion matrix:
 [[53 45]
 [ 0  3]]
Classification report:
               precision    recall  f1-score   support

           0      1.000     0.541     0.702        98
           1      0.062     1.000     0.118         3

    accuracy                          0.554       101
   macro avg      0.531     0.770     0.410       101
weighted avg      0.972     0.554     0.685       101


[Stacking Tuning]
[Stacking Tuning] Best params:
{'final_estimator__C': np.float64(7.579479953348009)}
[Stacking Tuning] Best CV PR-AUC: 0.705
/Users/benjaminemily/Desktop/ESILV/S7/ML/Project/venv/lib/python3.11/site-packages/sklearn/calibration.py:330: FutureWarning: The `cv='prefit'` option is deprecated in 1.6 and will be removed in 1.8. You can use CalibratedClassifierCV(FrozenEstimator(estimator)) instead.
  warnings.warn(

[Stacking] Validation PR-AUC: 0.975 | Best-F1 thr: 0.542 (F1=0.933)

=== Stacking (tuned) @ thr=0.542 ===
ROC-AUC: 0.714
PR-AUC:  0.065  (baseline=0.030)
Confusion matrix:
 [[43 55]
 [ 0  3]]
Classification report:
               precision    recall  f1-score   support

           0      1.000     0.439     0.610        98
           1      0.052     1.000     0.098         3

    accuracy                          0.455       101
   macro avg      0.526     0.719     0.354       101
weighted avg      0.972     0.455     0.595       101


[Summary] Validation/Test PR-AUC and thresholds
       model    val_pr       thr   test_pr
0      Logit  0.840074  0.687427  0.060027
1  Logit+PCA  0.623621  0.394503  0.066270
2         RF  0.975000  0.560254  0.067151
3        XGB  0.857955  0.213271  0.051474
4   Stacking  0.975000  0.542116  0.064871

[Choice] Select 'RF' based on highest validation PR-AUC (0.975); test PR-AUC=0.067, thr=0.560

Saved tuning results and feature importances to tuning/.
"""
