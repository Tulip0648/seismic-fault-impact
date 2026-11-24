"""
Module 4: Deep Tabular Residual MLP (PyTorch) with a scikit-learn style API.

Why a wrapper class?  Every other model in Module 4 is an sklearn estimator, so the
same CV / OOF / 5x2 significance-test code can run on all of them.  This class gives
the PyTorch network the same `fit / predict / predict_proba` interface.

It lives in a .py file (not in the notebook) so that a fitted network can be pickled
inside `end_to_end_pipeline.joblib` and loaded from any working directory.
"""

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight


class ResidualBlock(nn.Module):
    """x -> x + Dropout(Linear(GELU(Linear(BatchNorm(x)))))   (pre-norm residual block)."""

    def __init__(self, width: int, dropout: float, expansion: int = 2):
        super().__init__()
        self.norm = nn.BatchNorm1d(width)
        self.fc1 = nn.Linear(width, width * expansion)
        self.fc2 = nn.Linear(width * expansion, width)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.norm(x)
        h = self.drop(F.gelu(self.fc1(h)))
        h = self.drop(self.fc2(h))
        return x + h


class TabularResNet(nn.Module):
    """Input projection -> N residual blocks -> BatchNorm -> GELU -> linear head."""

    def __init__(self, n_features: int, n_classes: int, width: int = 128, n_blocks: int = 3, dropout: float = 0.3):
        super().__init__()
        self.inp = nn.Linear(n_features, width)
        self.blocks = nn.Sequential(*[ResidualBlock(width, dropout) for _ in range(n_blocks)])
        self.out_norm = nn.BatchNorm1d(width)
        self.head = nn.Linear(width, n_classes)

    def forward(self, x):
        h = self.blocks(self.inp(x))
        return self.head(F.gelu(self.out_norm(h)))


def focal_loss(logits, target, alpha=None, gamma=2.0):
    """Multi-class focal loss: down-weights easy (mostly 'green') examples."""
    log_p = F.log_softmax(logits, dim=1)
    log_pt = log_p.gather(1, target[:, None]).squeeze(1)
    loss = -((1 - log_pt.exp()) ** gamma) * log_pt
    if alpha is not None:
        loss = loss * alpha[target]
    return loss.mean()


class TabularResNetClassifier(BaseEstimator, ClassifierMixin):
    """sklearn-compatible wrapper around TabularResNet.

    - Standardises inputs internally (scaler fit on the training data of each `fit`).
    - Class-weighted cross-entropy (`loss="weighted_ce"`) or focal loss (`loss="focal"`).
    - AdamW + cosine-annealing learning-rate schedule.
    - Early stopping on a validation set: `eval_set=(X_val, y_val)` if given, otherwise a
      stratified `val_fraction` hold-out carved from the training data.
    """

    def __init__(self, width=128, n_blocks=3, dropout=0.3, lr=1e-3, weight_decay=1e-4,
                 epochs=200, batch_size=256, patience=25, loss="weighted_ce", focal_gamma=2.0,
                 val_fraction=0.15, random_state=42, device="auto", verbose=False):
        self.width = width
        self.n_blocks = n_blocks
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.loss = loss
        self.focal_gamma = focal_gamma
        self.val_fraction = val_fraction
        self.random_state = random_state
        self.device = device
        self.verbose = verbose

    # ------------------------------------------------------------------ helpers
    def _device(self):
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def _loss(self, logits, target, alpha):
        if self.loss == "focal":
            return focal_loss(logits, target, alpha, self.focal_gamma)
        return F.cross_entropy(logits, target, weight=alpha)

    def _to_tensor(self, X):
        X = np.asarray(X, dtype=np.float32)
        return torch.from_numpy(self.scaler_.transform(X).astype(np.float32))

    # ------------------------------------------------------------------ sklearn API
    def fit(self, X, y, eval_set=None):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y).astype(np.int64)
        self.classes_ = np.unique(y)
        y_idx = np.searchsorted(self.classes_, y)
        self.n_features_in_ = X.shape[1]

        if eval_set is None:
            X, X_val, y_idx, y_val = train_test_split(X, y_idx, test_size=self.val_fraction,
                                                      stratify=y_idx, random_state=self.random_state)
        else:
            X_val = np.asarray(eval_set[0], dtype=np.float32)
            y_val = np.searchsorted(self.classes_, np.asarray(eval_set[1]).astype(np.int64))

        self.scaler_ = StandardScaler().fit(X)
        dev = self._device()
        Xt, yt = self._to_tensor(X).to(dev), torch.from_numpy(y_idx).to(dev)
        Xv, yv = self._to_tensor(X_val).to(dev), torch.from_numpy(y_val).to(dev)

        weights = compute_class_weight("balanced", classes=np.arange(len(self.classes_)), y=y_idx)
        alpha = torch.tensor(weights, dtype=torch.float32, device=dev)

        net = TabularResNet(X.shape[1], len(self.classes_), self.width, self.n_blocks, self.dropout).to(dev)
        # foreach=False: the multi-tensor (foreach) AdamW path is pathologically slow on some CPU builds
        # of PyTorch (each step dispatches through torch.distributed tensor strategies). Same maths.
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.weight_decay, foreach=False)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        gen = torch.Generator(device="cpu").manual_seed(self.random_state)

        best_loss, best_state, best_epoch, bad = np.inf, None, 0, 0
        self.history_ = []
        n = len(Xt)
        for epoch in range(self.epochs):
            net.train()
            perm = torch.randperm(n, generator=gen).to(dev)
            running = 0.0
            for start in range(0, n, self.batch_size):
                idx = perm[start:start + self.batch_size]
                if len(idx) < 2:          # BatchNorm needs >= 2 rows
                    continue
                opt.zero_grad()
                loss = self._loss(net(Xt[idx]), yt[idx], alpha)
                loss.backward()
                opt.step()
                running += loss.item() * len(idx)
            sched.step()

            net.eval()
            with torch.no_grad():
                val_loss = self._loss(net(Xv), yv, alpha).item()
            self.history_.append({"epoch": epoch + 1, "train_loss": running / n, "val_loss": val_loss})
            if val_loss < best_loss - 1e-5:
                best_loss, best_epoch, bad = val_loss, epoch + 1, 0
                best_state = copy.deepcopy(net.state_dict())
            else:
                bad += 1
                if bad >= self.patience:
                    break
            if self.verbose and (epoch + 1) % 20 == 0:
                print(f"epoch {epoch + 1:4d}  train {running / n:.4f}  val {val_loss:.4f}")

        net.load_state_dict(best_state)
        self.best_epoch_ = best_epoch
        self.best_val_loss_ = best_loss
        self.net_ = net.cpu().eval()      # keep on CPU so the fitted object pickles anywhere
        return self

    def predict_proba(self, X):
        with torch.no_grad():
            logits = self.net_(self._to_tensor(X))
        return F.softmax(logits, dim=1).numpy().astype(np.float64)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    # ------------------------------------------------------------------ .pt export
    def save_pt(self, path, feature_names=None):
        """Save weights + everything needed to rebuild the model (spec: best_mlp_aug.pt)."""
        torch.save({
            "state_dict": self.net_.state_dict(),
            "params": self.get_params(),
            "n_features": self.n_features_in_,
            "classes": self.classes_.tolist(),
            "scaler_mean": self.scaler_.mean_.tolist(),
            "scaler_scale": self.scaler_.scale_.tolist(),
            "feature_names": list(feature_names) if feature_names is not None else None,
            "best_epoch": self.best_epoch_,
        }, path)

    @classmethod
    def load_pt(cls, path):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        obj = cls(**ckpt["params"])
        obj.classes_ = np.asarray(ckpt["classes"])
        obj.n_features_in_ = ckpt["n_features"]
        obj.scaler_ = StandardScaler()
        obj.scaler_.mean_ = np.asarray(ckpt["scaler_mean"])
        obj.scaler_.scale_ = np.asarray(ckpt["scaler_scale"])
        obj.scaler_.var_ = obj.scaler_.scale_ ** 2
        obj.scaler_.n_features_in_ = ckpt["n_features"]
        net = TabularResNet(ckpt["n_features"], len(obj.classes_), obj.width, obj.n_blocks, obj.dropout)
        net.load_state_dict(ckpt["state_dict"])
        obj.net_ = net.eval()
        obj.best_epoch_ = ckpt.get("best_epoch")
        return obj
