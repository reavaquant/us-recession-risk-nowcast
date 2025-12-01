# Recession Risk Nowcasting

Small ML project that nowcasts US recession risk (NBER indicator) from monthly macro/financial series: industrial production, CPI, housing starts, sentiment, unemployment, and the 10Y-3M Treasury spread. It builds engineered features, trains several classic classifiers (Logit, Logit+PCA, Random Forest, XGBoost, Stacking), calibrates their probabilities, and compares them with a small sequence-to-one LSTM.

## Data expected
Place these CSVs under `data/`, each with columns `observation_date` and one value column:
- `us_recession.csv` (target)
- `unrate.csv`
- `umcsent.csv`
- `indpro.csv`
- `houst.csv`
- `cpi.csv`
- `1030treasury_spreads.csv` (resampled to monthly inside the code)

The target is shifted by `H=6` months (`base.py`). Missing sentiment is forward filled and lagged by 1 month. Feature engineering uses log changes and lags for level variables, and diffs/lags for rates/spreads.

## Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Quick runs
- Baseline training/evaluation (time split 60/20/20, threshold picked by Fbeta with beta=0.5):
  ```bash
  python base.py
  ```
- Randomized hyperparameter search + calibration (Logit, Logit+PCA, RF, XGB, Stacking). Adjust `--n-iter`/`--seed` as needed:
  ```bash
  python optimization.py --n-iter 60 --seed 42
  ```
  Tuning CV results and feature importances are saved to `tuning/` as CSV files.
- Exploratory analysis: open `eda.ipynb` in Jupyter.

## What each script does
- `base.py`: builds features, splits chronologically, fits Logit/Logit+PCA/RF/XGB/Stacking, calibrates with sigmoid, selects thresholds by Fbeta, prints ROC/PR/confusion matrices, runs rolling CV and sliding-window tests, then trains a small LSTM for comparison.
- `optimization.py`: same data flow but adds RandomizedSearchCV to tune the classic models before calibration, then summarizes validation/test PR-AUC and sliding-window metrics.
- `calibrator.py`: tiny wrapper to calibrate a prefit model (sigmoid or isotonic) and expose `predict_proba`.
- `lstm.py`: PyTorch sequence-to-one LSTM trainer for monthly windows.
- `TODO.md`: short backlog/ideas.

## Notes and cautions
- The held-out test window is tiny (few positives), so PR-AUC and precision on that split are noisy; rely more on validation, rolling CV, and sliding-window diagnostics.
- Thresholding uses precision-leaning Fbeta (beta=0.5). Adjust `BETA` in code if you want to favor recall.
- Models are not checkpointed; only tuning artifacts and feature importances are written to `tuning/`.
