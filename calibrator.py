import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

class PrefitCalibratedModel:
    def __init__(self, base_estimator, method="sigmoid"):
        """
        Initialize a PrefitCalibratedModel.

        Parameters
        ----------
        base_estimator : estimator
            The underlying estimator to calibrate.
        method : str, default="sigmoid"
            The method to use for calibration. Can be "sigmoid" or "isotonic".
        """
        self.base_estimator = base_estimator
        self.method = method

    def fit(self, X_cal, y_cal):
        """
        Fit the calibrated model to the given data.

        Parameters
        ----------
        X_cal : array-like of shape (n_samples, n_features)
            The data to fit the calibrated model to.
        y_cal : array-like of shape (n_samples,)
            The true labels of the data to fit the calibrated model to.

        Returns
        -------
        self : PrefitCalibratedModel
            The calibrated model, fitted to the data.
        """
        raw = self.base_estimator.predict_proba(X_cal)[:, 1]
        if self.method == "sigmoid":
            cal = LogisticRegression(solver="lbfgs", max_iter=1000)
            cal.fit(raw.reshape(-1, 1), y_cal)
            self._predict_cal = lambda r: cal.predict_proba(r.reshape(-1, 1))[:, 1]
            self.calibrator = cal
        else:
            cal = IsotonicRegression(out_of_bounds="clip")
            cal.fit(raw, y_cal)
            self._predict_cal = lambda r: cal.predict(r)
            self.calibrator = cal
        return self

    def predict_proba(self, X):
        """
        Predict the probability of the positive class given the input data X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data to predict probabilities for.

        Returns
        -------
        proba : array-like of shape (n_samples, 2)
            The predicted probabilities of the positive class.

        """
        raw = self.base_estimator.predict_proba(X)[:, 1]
        cal = self._predict_cal(raw)
        return np.vstack([1 - cal, cal]).T