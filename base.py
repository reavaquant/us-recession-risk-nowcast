import pandas as pd
import numpy as np

from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, precision_recall_curve, confusion_matrix, balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.calibration import CalibratedClassifierCV

from lstm import train_lstm_sequence


FILES = {
    "recession": "data/us_recession.csv",
    "unrate": "data/unrate.csv",
    "sentiment": "data/umcsent.csv",
    "indpro": "data/indpro.csv",
    "housing": "data/houst.csv",
    "cpi": "data/cpi.csv",
    "treasury_spread": "data/1030treasury_spreads.csv",
}

H = 6

def _load_series(path, series_name):
    """
    Load a series from a file and return it as a pandas Series.

    Parameters
    ----------
    path : str
        Path to the file containing the series.
    series_name : str
        Name to give to the series.

    Returns
    -------
    pd.Series
        Loaded series.

    Raises
    ------
    ValueError
        If the file at `path` does not contain exactly one column
        of data (in addition to the "observation_date" column).
    """
    data = pd.read_csv(path, parse_dates=["observation_date"])
    value_cols = [c for c in data.columns if c != "observation_date"]
    if len(value_cols) != 1:
        raise ValueError(f"{path} should have exactly one column of data")
    series = data.rename(columns={value_cols[0]: series_name}).set_index("observation_date")
    if series_name == "treasury_spread":
        series = series.resample("MS").mean() # resemple to monthly data as it is in daily originally
    return series

def compute_features(files, horizon=H):
    """
    Compute features from a set of series.

    Parameters
    ----------
    files : dict
        Mapping of series names to paths of files containing the series.
    horizon : int, optional
        Number of months to predict the recession. Defaults to H.

    Returns
    -------
    pd.DataFrame
        Concatenated features.
    pd.Series
        Target to predict (recession).
    """
    df = pd.concat([_load_series(p, n) for n,p in files.items()], axis=1, join="inner").sort_index()
    df["sentiment"] = df["sentiment"].ffill().shift(1)
    df["target"] = df["recession"].shift(-horizon).astype(float)

    feats = []
    for col in ["indpro","cpi","housing","sentiment"]:
        s = df[col]
        block = pd.DataFrame(index=df.index)
        block[f"{col}_yoy"] = np.log(s).diff(12)   # YoY log change (stable)
        block[f"{col}_m1"] = np.log(s).diff(1)    # 1m log change
        for L in [3,6,12]:
            block[f"{col}_lag{L}"] = s.shift(L)
        feats.append(block)

    for col in ["unrate","treasury_spread"]:
        s = df[col]
        block = pd.DataFrame(index=df.index)
        block[f"{col}_d1"] = s.diff(1)
        block[f"{col}_d12"] = s.diff(12)
        for L in [3,6,12]:
            block[f"{col}_lag{L}"] = s.shift(L)
        feats.append(block)

    X = pd.concat(feats, axis=1).replace([np.inf,-np.inf], np.nan)
    data = pd.concat([X, df["target"]], axis=1).dropna()
    y = data.pop("target").astype(int)
    return data, y

def report_metrics(name, y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    base_rate = y_true.mean()  # baseline for PR-AUC
    print(f"\n=== {name} @ threshold={threshold:.3f} ===")
    print(f"ROC-AUC: {roc_auc_score(y_true, y_prob):.3f}")
    print(f"PR-AUC: {average_precision_score(y_true, y_prob):.3f}  (baseline={base_rate:.3f})")
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))
    print("Classification report:\n", classification_report(y_true, y_pred, digits=3))

def best_threshold_by_f1(y_true, y_prob):
    """
    Find the best threshold to maximize the F1-score between the true labels, y_true, and the predicted probabilities, y_prob.

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
    p, r, thr = precision_recall_curve(y_true, y_prob)
    f1 = (2 * p * r / (p + r + 1e-12))[:-1]  # drop last point (no threshold)
    if len(f1) == 0:
        return 0.5, 0.0
    ix = int(np.argmax(f1))
    return float(thr[ix]), float(f1[ix])

def rolling_cv_scores(model, X, y, n_splits=5, gap=H):
    """
    Compute rolling cross-validation scores for a given model and data.

    Parameters
    ----------
    model : sklearn.base.BaseEstimator
        Model to evaluate.
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target vector.
    n_splits : int, optional
        Number of CV folds. Default is 5.
    gap : int, optional
        Gap, in samples, between train and validation. Default is H.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the rolling cross-validation scores.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    rows = []
    for k, (tr, va) in enumerate(tscv.split(X)):
        y_tr, y_va = y.iloc[tr], y.iloc[va]
        if y_tr.nunique() < 2 or y_va.nunique() < 2:
            rows.append({"fold": k+1, "roc_auc": np.nan, "pr_auc": np.nan, "base_rate": y_va.mean()})
            continue

        m = clone(model)  # IMPORTANT: fresh model per fold

        if isinstance(m, XGBClassifier):
            # Ensure proper metric; give it a validation set for early stopping
            m.set_params(eval_metric="aucpr", early_stopping_rounds=50)
            m.fit(X.iloc[tr], y_tr,
                  eval_set=[(X.iloc[va], y_va)],
                  verbose=False)
        else:
            # Disable any lingering early stopping params if present
            try:
                m.set_params(early_stopping_rounds=None)
            except Exception:
                pass
            m.fit(X.iloc[tr], y_tr)

        p = m.predict_proba(X.iloc[va])[:, 1]
        rows.append({
            "fold": k+1,
            "roc_auc": roc_auc_score(y_va, p),
            "pr_auc": average_precision_score(y_va, p),
            "base_rate": y_va.mean(),
        })
    return pd.DataFrame(rows)

def sliding_test_scores(model, X, y, window=0.2, step=0.05):
    """
    Evaluate a model on a sliding window of the dataset.

    Parameters
    ----------
    model : sklearn estimator
        Model to evaluate.
    X : pd.DataFrame
        DataFrame containing the feature data.
    y : pd.Series
        Series containing the target data.
    window : float, optional
        Fraction of the dataset to use as the sliding window. Defaults to 0.2.
    step : float, optional
        Fraction of the dataset to use as the step size. Defaults to 0.05.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the evaluation metrics for each window.
    """
    n = len(X)
    w = int(n*window)
    s = int(n*step)
    rows = []
    for start in range(0, n - w + 1, s):
        end = start + w
        X_tr, y_tr = X.iloc[:start], y.iloc[:start]
        X_te, y_te = X.iloc[start:end], y.iloc[start:end]

        if start < max(50, 3*H) or y_tr.nunique() < 2 or y_te.nunique() < 2:
            continue

        i_split = int(len(X_tr) * 0.8)
        X_tr_in, y_tr_in = X_tr.iloc[:i_split], y_tr.iloc[:i_split]
        X_val_in, y_val_in = X_tr.iloc[i_split:], y_tr.iloc[i_split:]
        has_both_val_classes = y_val_in.nunique() > 1

        m = clone(model)
        if isinstance(m, XGBClassifier):
            m.set_params(eval_metric="aucpr")
            if has_both_val_classes:
                m.set_params(early_stopping_rounds=50)
                m.fit(X_tr_in, y_tr_in,
                      eval_set=[(X_val_in, y_val_in)],
                      verbose=False)
            else:
                m.set_params(early_stopping_rounds=None)
                m.fit(X_tr_in, y_tr_in, verbose=False)
        else:
            try:
                m.set_params(early_stopping_rounds=None)
            except Exception:
                pass
            m.fit(X_tr_in, y_tr_in)

        if not has_both_val_classes:
            thr = 0.5  # no PR curve possible, fallback threshold
        else:
            p_val = m.predict_proba(X_val_in)[:, 1]
            P, R, T = precision_recall_curve(y_val_in, p_val)
            if len(T) == 0:
                thr = 0.5
            else:
                F1 = (2*P*R/(P+R+1e-12))[:-1]
                thr = float(T[int(np.argmax(F1))])

        p_te = m.predict_proba(X_te)[:, 1]
        y_hat = (p_te >= thr).astype(int)
        rows.append({
            "start": X_te.index.min(), "end": X_te.index.max(),
            "positives": int(y_te.sum()),
            "roc_auc": roc_auc_score(y_te, p_te),
            "pr_auc": average_precision_score(y_te, p_te),
            "bal_acc": balanced_accuracy_score(y_te, y_hat),
            "f1": f1_score(y_te, y_hat, zero_division=0),
            "thr": thr,
            "base_rate": y_te.mean(),
        })
    return pd.DataFrame(rows)


def main():
    X, y = compute_features(FILES, horizon=H)
    n = len(X)
    i_tr = int(n * 0.6)
    i_va = int(n * 0.8)
    # Chronological split to avoid lookahead bias
    X_tr, y_tr = X.iloc[:i_tr], y.iloc[:i_tr]
    X_va, y_va = X.iloc[i_tr:i_va], y.iloc[i_tr:i_va]
    X_te, y_te = X.iloc[i_va:], y.iloc[i_va:]

    print(f"Samples: total={n} | train={len(X_tr)} | val={len(X_va)} | test={len(X_te)}")
    print(f"Positives by split: train={y_tr.sum()}, val={y_va.sum()}, test={y_te.sum()}")


    logit = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=5000, C=0.5, solver="lbfgs", random_state=42)),
    ])
    pca_logit = Pipeline([
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=0.9, svd_solver="full")),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=5000, C=0.5, solver="lbfgs", random_state=42)),
    ])
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=6, min_samples_leaf=10,
        class_weight="balanced_subsample", random_state=42, n_jobs=-1
    )
    pos, neg = int((y_tr==1).sum()), int((y_tr==0).sum())
    xgb = XGBClassifier(
        n_estimators=200, learning_rate=0.02, max_depth=3,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        scale_pos_weight=neg/max(pos,1), tree_method="hist",
        eval_metric="aucpr", early_stopping_rounds=20,random_state=42
    )
    stacking = StackingClassifier(
        estimators=[
            ("rf", clone(rf)),
            ("xgb", clone(xgb).set_params(early_stopping_rounds=None)),
        ],
        final_estimator=LogisticRegression(
            class_weight="balanced", max_iter=5000, C=0.5, solver="lbfgs", random_state=42
        ),
        stack_method="predict_proba",
        n_jobs=-1,
    )

    # 4) Fit on TRAIN; choose threshold on VALIDATION (max F1)
    # Logistic
    logit.fit(X_tr, y_tr)
    logit_cal = CalibratedClassifierCV(logit, method="sigmoid", cv="prefit")
    logit_cal.fit(X_va, y_va)
    p_val_log = logit_cal.predict_proba(X_va)[:, 1]
    thr_log, f1_log = best_threshold_by_f1(y_va, p_val_log)
    print(f"\n[Logit] Validation PR-AUC: {average_precision_score(y_va, p_val_log):.3f} | Best-F1 thr: {thr_log:.3f} (F1={f1_log:.3f})")

    # Logistic + PCA
    pca_logit.fit(X_tr, y_tr)
    pcal_cal = CalibratedClassifierCV(pca_logit, method="sigmoid", cv="prefit")
    pcal_cal.fit(X_va, y_va)
    p_val_pca = pcal_cal.predict_proba(X_va)[:, 1]
    thr_pca, f1_pca = best_threshold_by_f1(y_va, p_val_pca)
    print(f"[Logit+PCA] Validation PR-AUC: {average_precision_score(y_va, p_val_pca):.3f} | Best-F1 thr: {thr_pca:.3f} (F1={f1_pca:.3f})")

    # XGB (early stopping on validation)
    xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    xgb_cal = CalibratedClassifierCV(xgb, method="sigmoid", cv="prefit")
    xgb_cal.fit(X_va, y_va)
    p_val_xgb = xgb_cal.predict_proba(X_va)[:, 1]
    thr_xgb, f1_xgb = best_threshold_by_f1(y_va, p_val_xgb)
    print(f"[XGB] Validation PR-AUC: {average_precision_score(y_va, p_val_xgb):.3f} | Best-F1 thr: {thr_xgb:.3f} (F1={f1_xgb:.3f})")

    # RF
    rf.fit(X_tr, y_tr)
    rf_cal = CalibratedClassifierCV(rf, method="sigmoid", cv="prefit")
    rf_cal.fit(X_va, y_va)
    p_val_rf = rf_cal.predict_proba(X_va)[:, 1]
    thr_rf, f1_rf = best_threshold_by_f1(y_va, p_val_rf)
    print(f"[RF]  Validation PR-AUC: {average_precision_score(y_va, p_val_rf):.3f} | Best-F1 thr: {thr_rf:.3f} (F1={f1_rf:.3f})")

    # Stacking ensemble (RF + XGB -> Logistic)
    stacking.fit(X_tr, y_tr)
    stack_cal = CalibratedClassifierCV(stacking, method="sigmoid", cv="prefit")
    stack_cal.fit(X_va, y_va)
    p_val_stack = stack_cal.predict_proba(X_va)[:, 1]
    thr_stack, f1_stack = best_threshold_by_f1(y_va, p_val_stack)
    print(f"[Stacking RF+XGB->Logit] Validation PR-AUC: {average_precision_score(y_va, p_val_stack):.3f} | Best-F1 thr: {thr_stack:.3f} (F1={f1_stack:.3f})")


    report_metrics("Logistic Regression", y_te, logit_cal.predict_proba(X_te)[:, 1], thr_log)
    report_metrics("XGBoost", y_te, xgb_cal.predict_proba(X_te)[:, 1], thr_xgb)
    report_metrics("Random Forest", y_te, rf_cal.predict_proba(X_te)[:, 1], thr_rf)
    report_metrics("Logistic Regression + PCA", y_te, pcal_cal.predict_proba(X_te)[:, 1], thr_pca)
    report_metrics("Stacking (RF+XGB -> Logit)", y_te, stack_cal.predict_proba(X_te)[:, 1], thr_stack)

    # Quick recap of validation/test PR-AUCs and chosen thresholds
    summary = []
    def add_summary(name, val_probs, thr, model):
        summary.append({
            "model": name,
            "val_pr": average_precision_score(y_va, val_probs),
            "val_thr": thr,
            "test_pr": average_precision_score(y_te, model.predict_proba(X_te)[:, 1]),
        })
    add_summary("Logit", p_val_log, thr_log, logit_cal)
    add_summary("Logit+PCA", p_val_pca, thr_pca, pcal_cal)
    add_summary("RandomForest", p_val_rf, thr_rf, rf_cal)
    add_summary("XGBoost", p_val_xgb, thr_xgb, xgb_cal)
    add_summary("Stacking", p_val_stack, thr_stack, stack_cal)
    print("\n[Summary] Validation/Test PR-AUC and thresholds")
    print(pd.DataFrame(summary))

    print("\n[MAIN] Starting LSTM...", flush=True)
    try:
        lstm_res = train_lstm_sequence(X, y, i_tr=i_tr, i_va=i_va, lookback=18, hidden_size=16, num_layers=1, dropout=0.1, lr=1e-3, batch_size=64, max_epochs=20, patience=3)
        print(f"[LSTM seq2one] Validation PR-AUC: {lstm_res['val_pr_auc']:.3f} | "
          f"Best-F1 thr: {lstm_res['threshold']:.3f} (F1={lstm_res['val_f1']:.3f}) | "
          f"calibrated={lstm_res['calibrated']}", flush=True)
        if len(np.unique(lstm_res["test_targets"])) < 2:
            print("[LSTM] Not enough positive/negative examples in the test window to compute full metrics.")
        else:
            report_metrics("LSTM (sequence)", lstm_res["test_targets"], lstm_res["test_probs"], lstm_res["threshold"])
        print("[LSTM] Note: very few positives; validation can look perfect but test recall is low, interpret with caution.")
    except Exception as e:
        print(f"[LSTM] skipped due to error: {e}", flush=True)

    # Robustness: rolling CV on first 80% (train+val)
    X80, y80 = X.iloc[:i_va], y.iloc[:i_va]
    cv_log = rolling_cv_scores(logit, X80, y80, n_splits=3, gap=H)
    cv_xgb = rolling_cv_scores(xgb, X80, y80, n_splits=3, gap=H)
    cv_rf = rolling_cv_scores(rf, X80, y80, n_splits=3, gap=H)
    cvlogpca = rolling_cv_scores(pca_logit, X80, y80, n_splits=3, gap=H)
    cv_stack = rolling_cv_scores(stacking, X80, y80, n_splits=3, gap=H)

    print("\n-- Rolling CV (mean ± std) on first 80% --")
    for name, dfcv in [("Logit", cv_log), ("XGB", cv_xgb), ("RF", cv_rf), ("Logit+PCA", cvlogpca)]:
        m = dfcv[["roc_auc","pr_auc","base_rate"]].mean()
        s = dfcv[["roc_auc","pr_auc","base_rate"]].std()
        print(f"{name}: ROC-AUC {m['roc_auc']:.3f}±{s['roc_auc']:.3f} | PR-AUC {m['pr_auc']:.3f}±{s['pr_auc']:.3f} | base {m['base_rate']:.3f}")

    # Robustness: sliding tests over the full span
    st_log = sliding_test_scores(logit, X, y, window=0.2, step=0.05)
    st_xgb = sliding_test_scores(xgb, X, y, window=0.2, step=0.05)
    st_rf = sliding_test_scores(rf, X, y, window=0.2, step=0.05)
    st_logpca = sliding_test_scores(pca_logit, X, y, window=0.2, step=0.05)
    st_stack = sliding_test_scores(stacking, X, y, window=0.2, step=0.05)

    def summarize_sliding(name, dfst):
        if len(dfst)==0:
            print(f"{name}: not enough data to run sliding windows.")
            return
        m = dfst[["roc_auc","pr_auc","bal_acc","f1","base_rate"]].mean()
        s = dfst[["roc_auc","pr_auc","bal_acc","f1","base_rate"]].std()
        print(f"{name} (sliding): ROC-AUC {m['roc_auc']:.3f}±{s['roc_auc']:.3f} | "
              f"PR-AUC {m['pr_auc']:.3f}±{s['pr_auc']:.3f} | "
              f"BalAcc {m['bal_acc']:.3f}±{s['bal_acc']:.3f} | "
              f"F1 {m['f1']:.3f}±{s['f1']:.3f} | base {m['base_rate']:.3f}")

    print("\n-- Sliding window tests (20% window, 5% step) --")
    summarize_sliding("Logit", st_log)
    summarize_sliding("XGB", st_xgb)
    summarize_sliding("RF", st_rf)
    summarize_sliding("Logit+PCA", st_logpca)
    summarize_sliding("Stacking", st_stack)


if __name__ == "__main__":
    main()


"""
Base models and Ensemble model with more advanced models use by literature (Stacking model: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3624931 Machine Learning, the Treasury Yield Curve and Recession Forecasting | LSTM Model: https://arxiv.org/abs/2310.17571)

OUTPUT:
Samples: total=505 | train=303 | val=101 | test=101
Positives by split: train=29, val=8, test=3

[Logit] Validation PR-AUC: 0.590 | Best-F1 thr: 0.518 (F1=0.615)
[Logit+PCA] Validation PR-AUC: 0.633 | Best-F1 thr: 0.160 (F1=0.714)
[XGB] Validation PR-AUC: 0.889 | Best-F1 thr: 0.444 (F1=0.941)
[RF]  Validation PR-AUC: 0.975 | Best-F1 thr: 0.266 (F1=0.933)
[Stacking RF+XGB->Logit] Validation PR-AUC: 0.986 | Best-F1 thr: 0.240 (F1=0.941)

=== Logistic Regression @ threshold=0.518 ===
ROC-AUC: 0.711
PR-AUC: 0.064  (baseline=0.030)
Confusion matrix:
 [[33 65]
 [ 0  3]]
Classification report:
               precision    recall  f1-score   support

           0      1.000     0.337     0.504        98
           1      0.044     1.000     0.085         3

    accuracy                          0.356       101
   macro avg      0.522     0.668     0.294       101
weighted avg      0.972     0.356     0.491       101


=== XGBoost @ threshold=0.444 ===
ROC-AUC: 0.653
PR-AUC: 0.075  (baseline=0.030)
Confusion matrix:
 [[39 59]
 [ 0  3]]
Classification report:
               precision    recall  f1-score   support

           0      1.000     0.398     0.569        98
           1      0.048     1.000     0.092         3

    accuracy                          0.416       101
   macro avg      0.524     0.699     0.331       101
weighted avg      0.972     0.416     0.555       101


=== Random Forest @ threshold=0.266 ===
ROC-AUC: 0.684
PR-AUC: 0.059  (baseline=0.030)
Confusion matrix:
 [[47 51]
 [ 0  3]]
Classification report:
               precision    recall  f1-score   support

           0      1.000     0.480     0.648        98
           1      0.056     1.000     0.105         3

    accuracy                          0.495       101
   macro avg      0.528     0.740     0.377       101
weighted avg      0.972     0.495     0.632       101


=== Logistic Regression + PCA @ threshold=0.160 ===
ROC-AUC: 0.731
PR-AUC: 0.068  (baseline=0.030)
Confusion matrix:
 [[31 67]
 [ 0  3]]
Classification report:
               precision    recall  f1-score   support

           0      1.000     0.316     0.481        98
           1      0.043     1.000     0.082         3

    accuracy                          0.337       101
   macro avg      0.521     0.658     0.281       101
weighted avg      0.972     0.337     0.469       101


=== Stacking (RF+XGB -> Logit) @ threshold=0.240 ===
ROC-AUC: 0.667
PR-AUC: 0.056  (baseline=0.030)
Confusion matrix:
 [[29 69]
 [ 0  3]]
Classification report:
               precision    recall  f1-score   support

           0      1.000     0.296     0.457        98
           1      0.042     1.000     0.080         3

    accuracy                          0.317       101
   macro avg      0.521     0.648     0.268       101
weighted avg      0.972     0.317     0.446       101


[Summary] Validation/Test PR-AUC and thresholds
          model    val_pr   val_thr   test_pr
0         Logit  0.590196  0.518338  0.063657
1     Logit+PCA  0.633289  0.160230  0.067621
2  RandomForest  0.975000  0.266466  0.058971
3       XGBoost  0.888889  0.444092  0.075000
4      Stacking  0.986111  0.239919  0.056116

[MAIN] Starting LSTM...
[LSTM] start training: max_epochs=20, batches/epoch=5
[LSTM] epoch 000 val_pr=0.1747 best=0.1747 bad=0
[LSTM] epoch 001 val_pr=0.1672 best=0.1747 bad=1
[LSTM] epoch 002 val_pr=0.1644 best=0.1747 bad=2
[LSTM] epoch 003 val_pr=0.1705 best=0.1747 bad=3
[LSTM seq2one] Validation PR-AUC: 0.195 | Best-F1 thr: 0.200 (F1=0.296) | calibrated=True

=== LSTM (sequence) @ threshold=0.200 ===
ROC-AUC: 0.781
PR-AUC: 0.070  (baseline=0.030)
Confusion matrix:
 [[58 40]
 [ 0  3]]
Classification report:
               precision    recall  f1-score   support

         0.0      1.000     0.592     0.744        98
         1.0      0.070     1.000     0.130         3

    accuracy                          0.604       101
   macro avg      0.535     0.796     0.437       101
weighted avg      0.972     0.604     0.725       101

[LSTM] Note: very few positives; validation can look perfect but test recall is low, interpret with caution. We will not optimise LSTM hyperparameters as it too unstable.

-- Rolling CV (mean ± std) on first 80% --
Logit: ROC-AUC 0.825±0.073 | PR-AUC 0.578±0.058 | base 0.092
XGB: ROC-AUC 0.947±0.071 | PR-AUC 0.810±0.191 | base 0.092
RF: ROC-AUC 0.947±0.058 | PR-AUC 0.821±0.123 | base 0.092
Logit+PCA: ROC-AUC 0.786±0.150 | PR-AUC 0.568±0.125 | base 0.092

-- Sliding window tests (20% window, 5% step) --
Logit (sliding): ROC-AUC 0.865±0.074 | PR-AUC 0.529±0.246 | BalAcc 0.632±0.097 | F1 0.218±0.186 | base 0.110
XGB (sliding): ROC-AUC 0.846±0.089 | PR-AUC 0.375±0.263 | BalAcc 0.709±0.114 | F1 0.329±0.165 | base 0.110
RF (sliding): ROC-AUC 0.901±0.076 | PR-AUC 0.530±0.269 | BalAcc 0.781±0.104 | F1 0.375±0.228 | base 0.110
Logit+PCA (sliding): ROC-AUC 0.888±0.102 | PR-AUC 0.621±0.310 | BalAcc 0.722±0.128 | F1 0.320±0.213 | base 0.110
Stacking (sliding): ROC-AUC 0.890±0.082 | PR-AUC 0.486±0.265 | BalAcc 0.746±0.118 | F1 0.377±0.229 | base 0.110
"""
