import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
import os
from PIL import Image
import numpy as np
from torchvision import datasets, transforms

def set_seed(seed: int):
    """Reproducible results."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def calculate_metrics(y_true, y_pred, scores):
    return {
        'AUC': roc_auc_score(y_true, scores),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1': f1_score(y_true, y_pred),
        'TP': confusion_matrix(y_true, y_pred)[1, 1],
        'FP': confusion_matrix(y_true, y_pred)[0, 1],
        'TN': confusion_matrix(y_true, y_pred)[0, 0],
        'FN': confusion_matrix(y_true, y_pred)[1, 0],
    }

class Autoencoder(nn.Module):
    """Simple fully‐connected Autoencoder for 28×28 inputs."""
    def __init__(self, input_dim=28*28, hidden_dims=(128, 32)):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (batch, 1, 28, 28) or (batch, 784)
        if x.ndim == 4:
            x = x.view(x.size(0), -1)
        z = self.encoder(x)
        return self.decoder(z)

class AutoencoderAnomalyDetector:
    """
    Wrapper that trains the AE on 'normal' data, computes a threshold
    on reconstruction error, and then predicts anomalies.
    """
    def __init__(self,
                 input_dim=28*28,
                 hidden_dims=(128, 32),
                 lr=1e-4,
                 threshold_percentile=95,
                 device: str = None,
                 seed: int = None):
        if seed is not None:
            set_seed(seed)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold_percentile = threshold_percentile

        self.model = Autoencoder(input_dim, hidden_dims).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.threshold = None

    def fit(self, X_train: np.ndarray, epochs=20, batch_size=256, verbose=True):
        """
        Train on normal data and set the anomaly threshold.
        X_train: array of shape (n_samples, 1, 28, 28) or (n_samples, 784)
        """
        self.model.train()
        X = torch.tensor(X_train, dtype=torch.float32, device=self.device)
        if X.ndim == 4:
            X = X.view(X.size(0), -1)
        n = X.size(0)

        for epoch in range(1, epochs + 1):
            perm = torch.randperm(n, device=self.device)
            epoch_loss = 0.0
            for i in range(0, n, batch_size):
                idx = perm[i : i + batch_size]
                batch = X[idx]
                self.optimizer.zero_grad()
                recon = self.model(batch)
                loss = self.criterion(recon, batch)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item() * batch.size(0)
            if verbose:
                print(f"Epoch {epoch}/{epochs}  Loss: {epoch_loss/n:.6f}")

        # compute reconstruction errors on train set
        self.model.eval()
        with torch.no_grad():
            recon = self.model(X).cpu().numpy()
        orig = X.cpu().numpy()
        errors = np.mean((recon - orig)**2, axis=1)
        self.threshold = np.percentile(errors, self.threshold_percentile)

    def predict(self, X: np.ndarray, return_score: bool = False):
        """
        Predict anomalies: labels or raw reconstruction-error scores.
        X: array of shape (n_samples, 1, 28, 28) or (n_samples, 784)
        """
        if self.threshold is None:
            raise RuntimeError("Detector not trained. Call fit() first.")

        self.model.eval()
        Xt = torch.tensor(X, dtype=torch.float32, device=self.device)
        if Xt.ndim == 4:
            Xt = Xt.view(Xt.size(0), -1)
        with torch.no_grad():
            recon = self.model(Xt).cpu().numpy()
        orig = Xt.cpu().numpy()
        errors = np.mean((recon - orig)**2, axis=1)
        labels = (errors > self.threshold).astype(int)
        return (errors if return_score else labels)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray):
        """
        Compute a suite of metrics on labeled test set using calculate_metrics.
        """
        # get raw scores and binary predictions
        scores = self.predict(X_test, return_score=True)
        y_pred = (scores > self.threshold).astype(int)
        # calculate and return metrics dict
        return calculate_metrics(y_test, y_pred, scores)


