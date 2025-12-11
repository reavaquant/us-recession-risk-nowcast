import copy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
torch.set_num_threads(1)
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression

class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        """
        Initialize a FocalLoss instance.

        Parameters:
        alpha (float, optional): hyperparameter for focal loss. Defaults to 0.25.
        gamma (float, optional): hyperparameter for focal loss. Defaults to 2.0.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        """
        Compute the focal loss for the given logits and targets.

        Parameters:
        logits (torch.Tensor): input logits
        targets (torch.Tensor): target labels

        Returns:
        torch.Tensor: computed loss
        """
        probs = torch.sigmoid(logits)
        ce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * (1 - p_t) ** self.gamma * ce
        return loss.mean()

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, dropout=0.2):
        """
        Initialize an LSTMClassifier instance.

        Parameters:
        input_size (int): number of input features
        hidden_size (int, optional): number of hidden units in the LSTM layer. Defaults to 64.
        num_layers (int, optional): number of LSTM layers. Defaults to 1.
        dropout (float, optional): dropout probability after each LSTM layer (except the last layer). Defaults to 0.2.
        """
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """
        Forward pass of the LSTM model.

        Parameters
        ----------
        x : torch.Tensor
            Input sequence of shape (batch_size, seq_len, input_size)

        Returns
        -------
        logits : torch.Tensor
            Output probabilities of shape (batch_size, seq_len)
        """
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        logits = self.fc(self.dropout(last_hidden))
        return logits.squeeze(1)

def _predict_loader(model, loader, device):
    """
    Predict probabilities and targets for a given model and loader.

    Parameters
    ----------
    model : nn.Module
        The model to predict with.
    loader : DataLoader
        The loader to iterate over.
    device : torch.device
        The device to run the predictions on.

    Returns
    -------
    probs : np.ndarray
        The predicted probabilities.
    targets : np.ndarray
        The true targets.
    """
    probs, targets = [], []
    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            prob = torch.sigmoid(logits)
            probs.append(prob.cpu().numpy())
            targets.append(yb.cpu().numpy())
    if not probs:
        return np.array([]), np.array([])
    return np.concatenate(probs), np.concatenate(targets)

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

def safe_average_precision(y_true, y_prob):
    """Return PR-AUC, falling back to the base rate if only one class is present."""
    if len(np.unique(y_true)) < 2:
        return float(np.mean(y_true))
    return float(average_precision_score(y_true, y_prob))

def make_sequence_arrays(X, y, lookback):
    """
    Roll windows over the timeline to build sequence-to-one samples for the LSTM.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (time ordered).
    y : pd.Series
        Binary labels aligned with X.
    lookback : int
        Number of past steps to include in each sequence.

    Returns
    -------
    seq_X : np.ndarray
        Array of shape (n_seq, lookback, n_features).
    seq_y : np.ndarray
        Array of shape (n_seq,) containing labels for each sequence.
    positions : np.ndarray
        Indices of X that correspond to each sequence label.
    timestamps : np.ndarray
        Datetime index values aligned with seq_y.
    """
    if lookback >= len(X):
        raise ValueError("lookback must be smaller than the dataset length.")
    X_arr = X.values.astype(np.float32)
    y_arr = y.values.astype(np.float32)
    seq_X, seq_y, positions, timestamps = [], [], [], []
    for i in range(lookback, len(X_arr)):
        seq_X.append(X_arr[i - lookback:i])
        seq_y.append(y_arr[i])
        positions.append(i)
        timestamps.append(X.index[i])
    return np.stack(seq_X), np.array(seq_y), np.array(positions), np.array(timestamps)

def train_lstm_sequence(X, y, i_tr, i_va, lookback=18, hidden_size=16, num_layers=1, dropout=0.1, lr=1e-3, batch_size=32, max_epochs=50, patience=5, min_delta=1e-3, use_focal=True, focal_alpha=0.25, focal_gamma=2.0,):
    """
    Train an LSTM model with the given hyperparameters on the given sequential data.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (time ordered).
    y : pd.Series
        Binary labels aligned with X.
    i_tr : int
        Index of the last training sample.
    i_va : int
        Index of the last validation sample.
    lookback : int, optional
        Number of past steps to include in each sequence. Default is 18.
    hidden_size : int, optional
        Number of hidden units in the LSTM layer. Default is 16.
    num_layers : int, optional
        Number of LSTM layers. Default is 1.
    dropout : float, optional
        Dropout rate for the LSTM layer. Default is 0.1.
    lr : float, optional
        Learning rate for the Adam optimizer. Default is 1e-3.
    batch_size : int, optional
        Batch size for the Adam optimizer. Default is 32.
    max_epochs : int, optional
        Maximum number of epochs to train the model. Default is 50.
    patience : int, optional
        Number of epochs to wait for improvement before stopping the training. Default is 5.
    min_delta : float, optional
        Minimum improvement in the validation set required to update the best model. Default is 1e-3.
    use_focal : bool, optional
        Whether to use focal loss. Default is True.
    focal_alpha : float, optional
        Alpha parameter for focal loss. Default is 0.25.
    focal_gamma : float, optional
        Gamma parameter for focal loss. Default is 2.0.

    Returns
    -------
    dict
        Dictionary containing the training history, best validation probabilities, and best validation labels.
    """
    if i_tr <= lookback:
        raise ValueError("lookback is too large compared to the training window.")

    scaler = StandardScaler()
    scaler.fit(X.iloc[:i_tr])
    X_scaled = pd.DataFrame(scaler.transform(X), index=X.index, columns=X.columns)

    seq_X, seq_y, positions, timestamps = make_sequence_arrays(X_scaled, y, lookback)
    mask_tr = positions < i_tr
    mask_va = (positions >= i_tr) & (positions < i_va)
    mask_te = positions >= i_va

    X_tr_seq, y_tr_seq = seq_X[mask_tr], seq_y[mask_tr]
    X_va_seq, y_va_seq = seq_X[mask_va], seq_y[mask_va]
    X_te_seq, y_te_seq = seq_X[mask_te], seq_y[mask_te]

    if len(X_tr_seq) == 0 or len(X_va_seq) == 0 or len(X_te_seq) == 0:
        raise RuntimeError("Not enough sequential data to build train/val/test LSTM splits.")

    device = torch.device("cpu")
    model = LSTMClassifier(input_size=X.shape[1], hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if use_focal:
        criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    else:
        pos = float(y_tr_seq.sum())
        neg = float(len(y_tr_seq) - pos)
        pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_loader = DataLoader(SequenceDataset(X_tr_seq, y_tr_seq), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(SequenceDataset(X_va_seq, y_va_seq), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(SequenceDataset(X_te_seq, y_te_seq), batch_size=batch_size, shuffle=False)

    best_state = None
    best_val_probs, best_val_targets = None, None
    best_pr = -np.inf
    bad_epochs = 0

    print(f"[LSTM] start training: max_epochs={max_epochs}, batches/epoch={len(train_loader)}", flush=True)
    for epoch in range(max_epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        val_probs, val_targets = _predict_loader(model, val_loader, device)
        if len(val_targets) == 0:
            break
        val_pr = safe_average_precision(val_targets, val_probs)
        if val_pr > best_pr + min_delta:
            best_pr = val_pr
            bad_epochs = 0
            best_state = copy.deepcopy(model.state_dict())
            best_val_probs, best_val_targets = val_probs, val_targets
        else:
            bad_epochs += 1
        if epoch % 1 == 0:  # <- chaque epoch
            print(f"[LSTM] epoch {epoch:03d} val_pr={val_pr:.4f} best={best_pr:.4f} bad={bad_epochs}", flush=True)
        if bad_epochs >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    val_probs, val_targets = best_val_probs, best_val_targets
    test_probs, test_targets = _predict_loader(model, test_loader, device)

    calibrator = None
    calibrated = False
    if val_probs is not None and len(np.unique(val_targets)) >= 2:
        try:
            calibrator = IsotonicRegression(out_of_bounds="clip")
            calibrator.fit(val_probs, val_targets)
            val_probs = calibrator.transform(val_probs)
            if len(test_probs):
                test_probs = calibrator.transform(test_probs)
            calibrated = True
        except Exception as e:
            print(f"(Skip isotonic calibration) {e}")

    thr, f1_val = best_threshold_by_f1(val_targets, val_probs)
    return {
        "val_pr_auc": safe_average_precision(val_targets, val_probs),
        "test_pr_auc": safe_average_precision(test_targets, test_probs),
        "val_probs": val_probs,
        "val_targets": val_targets,
        "test_probs": test_probs,
        "test_targets": test_targets,
        "threshold": thr,
        "val_f1": f1_val,
        "calibrated": calibrated,
        "timestamps": {
            "train": timestamps[mask_tr],
            "val": timestamps[mask_va],
            "test": timestamps[mask_te],
        },
    }