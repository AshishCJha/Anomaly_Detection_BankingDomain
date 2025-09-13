# banking_cnn_anomaly_end2end.py
# End-to-end: synthetic data (100k sequences) + 1D CNN Autoencoder anomaly detection + zipped outputs

import os
import math
import random
import zipfile
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score

# =========================
# Config
# =========================
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

TOTAL_SAMPLES = 100_000           # total sequences (train + val + test)
SEQ_LEN = 64                      # transactions per sequence
N_FEATURES = 4                    # amount, time_gap, merchant_cat, country_risk
TRAIN_NORMAL = 80_000             # normal-only training
VAL_NORMAL = 10_000               # normal-only validation (for threshold)
TEST_MIXED = TOTAL_SAMPLES - TRAIN_NORMAL - VAL_NORMAL  # 10,000
TEST_ANOMALY_RATE = 0.2           # 20% anomalies in test

BATCH_SIZE = 256
EPOCHS = 20
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PCT_FOR_THRESH = 0.995            # 99.5th percentile of val reconstruction error

# =========================
# Synthetic Banking Data Generator
# =========================
@dataclass
class GenParams:
    seq_len: int = SEQ_LEN
    n_features: int = N_FEATURES

def _seasonal(n, period, amp=1.0, phase=0.0):
    t = np.arange(n)
    return amp * np.sin(2*np.pi*(t/period + phase))

def _clip01(x):
    return np.clip(x, 0, 1)

def generate_normal_sequence(p: GenParams) -> np.ndarray:
    """
    Features (scaled ~[0,1]):
      0) amount:        daily/weekly seasonality + noise
      1) time_gap:      time between tx (normalized), higher => slower activity
      2) merchant_cat:  smooth drift around a center
      3) country_risk:  mostly low with small spikes
    """
    L = p.seq_len

    amt = 0.25 + 0.1 * _seasonal(L, period=24, amp=1.0, phase=np.random.rand()) + 0.05*np.random.randn(L)
    amt = _clip01(amt)

    time_gap = 0.6 + 0.15 * _seasonal(L, period=12, amp=1.0, phase=np.random.rand()) + 0.05*np.random.randn(L)
    time_gap = _clip01(time_gap)

    cat_center = np.random.uniform(0.3, 0.7)
    merchant_cat = cat_center + 0.05 * _seasonal(L, period=18, amp=1.0, phase=np.random.rand()) + 0.03*np.random.randn(L)
    merchant_cat = _clip01(merchant_cat)

    base_risk = np.random.uniform(0.05, 0.2)
    country_risk = base_risk + 0.05 * np.maximum(0, np.random.randn(L))
    country_risk = _clip01(country_risk)

    x = np.stack([amt, time_gap, merchant_cat, country_risk], axis=0)  # (C, L)
    return x.astype(np.float32)

def inject_anomalies(x: np.ndarray) -> np.ndarray:
    """
    Inject anomalies in a random window:
      - High-value bursts (amount spikes)
      - Rapid-fire tx (very small time gaps)
      - Category jump noise
      - High-risk country bursts
    """
    C, L = x.shape
    y = x.copy()

    anomaly_types = np.random.choice(
        ["amt_spike", "rapid_fire", "cat_jump", "risk_burst"],
        size=np.random.randint(1, 3), replace=False
    )

    start = np.random.randint(0, L//2)
    end = min(L, start + np.random.randint(L//8, L//3))

    if "amt_spike" in anomaly_types:
        y[0, start:end] = _clip01(y[0, start:end] + np.random.uniform(0.5, 1.0))

    if "rapid_fire" in anomaly_types:
        y[1, start:end] = _clip01(y[1, start:end] * np.random.uniform(0.0, 0.15))

    if "cat_jump" in anomaly_types:
        y[2, start:end] = _clip01(np.random.uniform(0.0, 1.0, size=end-start))

    if "risk_burst" in anomaly_types:
        y[3, start:end] = _clip01(y[3, start:end] + np.random.uniform(0.5, 1.0))

    y[:, start:end] = _clip01(y[:, start:end] + 0.05*np.random.randn(C, end-start).astype(np.float32))
    return y

def make_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      X_train (Ntr, C, L), y_train (zeros)
      X_val   (Nval, C, L), y_val (zeros)
      X_test  (Nte, C, L), y_test (0/1)
    """
    p = GenParams()

    # Train/Val normals
    X_train = np.stack([generate_normal_sequence(p) for _ in range(TRAIN_NORMAL)], axis=0)
    y_train = np.zeros(TRAIN_NORMAL, dtype=np.int64)

    X_val = np.stack([generate_normal_sequence(p) for _ in range(VAL_NORMAL)], axis=0)
    y_val = np.zeros(VAL_NORMAL, dtype=np.int64)

    # Test mixed
    n_anom = int(TEST_MIXED * TEST_ANOMALY_RATE)
    n_norm = TEST_MIXED - n_anom

    X_test_norm = np.stack([generate_normal_sequence(p) for _ in range(n_norm)], axis=0)
    X_test_anom = np.stack([inject_anomalies(generate_normal_sequence(p)) for _ in range(n_anom)], axis=0)

    X_test = np.concatenate([X_test_norm, X_test_anom], axis=0)
    y_test = np.concatenate([np.zeros(n_norm, dtype=np.int64), np.ones(n_anom, dtype=np.int64)], axis=0)

    # Shuffle test set
    idx = np.random.permutation(len(X_test))
    return X_train, y_train, X_val, y_val, X_test[idx], y_test[idx]

# =========================
# 1D CNN Autoencoder
# =========================
class CNNAE1D(nn.Module):
    def __init__(self, in_ch=N_FEATURES, latent_dim=64):
        super().__init__()
        # Encoder: (B, C, L)
        self.enc = nn.Sequential(
            nn.Conv1d(in_ch, 32, kernel_size=5, stride=2, padding=2),  # L -> L/2
            nn.BatchNorm1d(32), nn.ReLU(True),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),     # L/2 -> L/4
            nn.BatchNorm1d(64), nn.ReLU(True),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),    # L/4 -> L/8
            nn.BatchNorm1d(128), nn.ReLU(True),
        )
        L8 = SEQ_LEN // 8
        self.to_latent = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * L8, latent_dim),
        )
        self.from_latent = nn.Sequential(
            nn.Linear(latent_dim, 128 * L8),
            nn.ReLU(True),
        )
        self.dec = nn.Sequential(
            nn.Unflatten(1, (128, L8)),
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),  # L/8 -> L/4
            nn.BatchNorm1d(64), nn.ReLU(True),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),   # L/4 -> L/2
            nn.BatchNorm1d(32), nn.ReLU(True),
            nn.ConvTranspose1d(32, in_ch, kernel_size=4, stride=2, padding=1),# L/2 -> L
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.enc(x)
        z = self.to_latent(h)
        return z

    def decode(self, z):
        h = self.from_latent(z)
        x_hat = self.dec(h)
        return x_hat

    def forward(self, x):
        return self.decode(self.encode(x))

# =========================
# Training / Eval Utilities
# =========================
def per_sample_mse(x, x_hat):
    # (B, C, L) -> (B,)
    return ((x_hat - x) ** 2).view(x.size(0), -1).mean(dim=1)

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total = 0.0; n = 0
    for xb, _ in loader:
        xb = xb.to(DEVICE)
        optimizer.zero_grad()
        xh = model(xb)
        loss_map = criterion(xh, xb)
        loss = loss_map.mean()
        loss.backward()
        optimizer.step()
        total += loss.item() * xb.size(0)
        n += xb.size(0)
    return total / n

@torch.no_grad()
def collect_errors(model, loader):
    model.eval()
    errs = []
    ys = []
    for xb, yb in loader:
        xb = xb.to(DEVICE)
        xh = model(xb)
        e = per_sample_mse(xb, xh).cpu().numpy()
        errs.append(e)
        ys.append(yb.numpy())
    return np.concatenate(errs), np.concatenate(ys)

def metrics(y_true, y_pred, y_score=None):
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    auc = roc_auc_score(y_true, y_score) if (y_score is not None and len(np.unique(y_true)) > 1) else float('nan')
    return acc, p, r, f1, auc

# =========================
# Saving: model, metrics, CSV data, zip, optional download
# =========================
def save_all(model, thresh, acc, p, r, f1, auc,
             X_train, y_train, X_val, y_val, X_test, y_test):
    # 1) Save model
    torch.save(model.state_dict(), "banking_cnn_autoencoder.pt")

    # 2) Save metrics
    with open("results.txt", "w") as f:
        f.write("Banking CNN Autoencoder Anomaly Detection\n")
        f.write(f"Threshold: {thresh:.6f}\n")
        f.write(f"Accuracy:  {acc:.4f}\n")
        f.write(f"Precision: {p:.4f}\n")
        f.write(f"Recall:    {r:.4f}\n")
        f.write(f"F1-score:  {f1:.4f}\n")
        f.write(f"ROC-AUC:   {auc:.4f}\n")

    # 3) Save datasets to CSV (flatten each sequence)
    def save_csv(X, y, path):
        N, C, L = X.shape
        flat = X.reshape(N, C*L)
        df = pd.DataFrame(flat)
        df["label"] = y
        df.to_csv(path, index=False)

    save_csv(X_train, y_train, "train.csv")
    save_csv(X_val,   y_val,   "val.csv")
    save_csv(X_test,  y_test,  "test.csv")

    # 4) Zip outputs
    with zipfile.ZipFile("banking_anomaly_results.zip", "w") as zf:
        for fname in ["banking_cnn_autoencoder.pt", "results.txt", "train.csv", "val.csv", "test.csv"]:
            zf.write(fname)

    print("\n✔ All files saved to banking_anomaly_results.zip")

# =========================
# Main
# =========================
def main():
    print("Generating synthetic banking dataset...")
    X_train, y_train, X_val, y_val, X_test, y_test = make_dataset()

    tr_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    va_ds = TensorDataset(torch.from_numpy(X_val),   torch.from_numpy(y_val))
    te_ds = TensorDataset(torch.from_numpy(X_test),  torch.from_numpy(y_test))

    tr_ld = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
    va_ld = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    te_ld = DataLoader(te_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    model = CNNAE1D(in_ch=N_FEATURES, latent_dim=64).to(DEVICE)
    criterion = nn.MSELoss(reduction="none")
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    print(f"Training on {len(tr_ds):,} normal sequences...")
    for ep in range(1, EPOCHS+1):
        tr_loss = train_epoch(model, tr_ld, opt, criterion)
        print(f"[Epoch {ep:02d}] train_loss={tr_loss:.6f}")

    # Threshold from validation normals
    val_errs, _ = collect_errors(model, va_ld)
    thresh = float(np.quantile(val_errs, PCT_FOR_THRESH))
    print(f"Chosen threshold (p{PCT_FOR_THRESH*100:.2f} of val errors): {thresh:.6g}")

    # Evaluate on mixed test
    test_errs, test_y = collect_errors(model, te_ld)
    test_pred = (test_errs > thresh).astype(np.int64)

    acc, p, r, f1, auc = metrics(test_y, test_pred, y_score=test_errs)
    print(f"\n=== Test Results (mixed set of {len(test_y):,} with ~{TEST_ANOMALY_RATE*100:.0f}% anomalies) ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {p:.4f}")
    print(f"Recall:    {r:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")

    # Save everything
    save_all(model, thresh, acc, p, r, f1, auc,
             X_train, y_train, X_val, y_val, X_test, y_test)

if __name__ == "__main__":
    main()

    # Optional: auto-download if running in Google Colab
    try:
        from google.colab import files
        files.download("banking_anomaly_results.zip")
    except Exception:
        # On local Python/Jupyter or other environments, just grab the zip from the working directory
        print("Tip: In Colab this auto-downloads. Otherwise, find 'banking_anomaly_results.zip' in your working directory.")
