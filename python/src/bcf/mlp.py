from typing import Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ---- Generic MLP ------------------------------------------------------------
class MLP(nn.Module):
    def __init__(
        self, in_dim, out_dim=1, hidden=(128, 64), activation=nn.ReLU, dropout=0.0
    ):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), activation()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = h
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ---- Combiner for joint training of fx + gv --------------------------
class FxPlusGv(nn.Module):
    """
    Wraps two modules with signatures fx(X)->(N,), gv(V)->(N,),
    returns fx(X) + gv(V).
    """

    def __init__(self, fx: MLP, gv: MLP):
        super().__init__()
        self.fx = fx
        self.gv = gv

    def forward(self, X, V):
        return self.fx(X) + self.gv(V)


# ---- Training functions ----------------------------------------
def train_fx_gv(
    fx: MLP,
    gv: MLP,
    X: torch.Tensor,
    V: torch.Tensor,
    y: torch.Tensor,
    epochs=100,
    batch_size=256,
    lr=1e-3,
    weight_decay=0e-4,
    device="cpu",
    verbose=True,
):
    fx.to(device)
    gv.to(device)
    model = FxPlusGv(fx, gv).to(device)

    ds = TensorDataset(X, V, y)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    # opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    opt = make_opt_fx_gv(fx=fx, gv=gv, lr=lr, wd=weight_decay)

    loss_fn = nn.MSELoss()

    history = {"train_mse": []}
    model.train()
    if verbose:
        print("Train MSE Phase I -------------")
    for epoch in range(epochs):
        loss_sum, loss_fx, loss_gv, nobs = 0.0, 0.0, 0.0, 0
        for Xb, Vb, yb in dl:
            Xb, Vb, yb = Xb.to(device), Vb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(Xb, Vb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

            # ---- save loss for fx + gv
            nobs += Xb.size(0)
            loss_sum += float(loss.item()) * Xb.size(0)

            # ---- compute loss for fx
            loss = loss_fn(fx(Xb), yb)
            loss_fx += float(loss.item()) * Xb.size(0)

            # ---- compute loss for gv
            loss = loss_fn(gv(Vb), yb)
            loss_gv += float(loss.item()) * Xb.size(0)

        # compute average training loss across the epochs
        train_mse = loss_sum / max(nobs, 1)
        train_fx_mse = loss_fx / max(nobs, 1)
        train_gv_mse = loss_gv / max(nobs, 1)

        history["train_mse"].append(train_mse)

        if verbose and (epoch % 50 == 0):
            print(
                f"Epoch {epoch}/{epochs} - fx+gv={train_mse:.6f} - fx= {train_fx_mse:.6f} - gv={train_gv_mse:.6f}"
            )


def train_fx_imp(
    fx: MLP,
    fx_imp: MLP,
    X: torch.Tensor,  # full X for fx
    RX: torch.Tensor,  # projected continuous features for fx_imp
    y: torch.Tensor,
    epochs=100,
    batch_size=256,
    lr=1e-3,
    weight_decay=0e-4,
    device="cpu",
    verbose=True,
):
    fx.to(device)
    fx_imp.to(device)
    fx.eval()  # freeze fx
    for p in fx.parameters():
        p.requires_grad_(False)

    # precompute residual target y2 = y - f(X)
    with torch.no_grad():
        y2 = y.to(device) - fx(X.to(device))

    ds = TensorDataset(RX, y2)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    # opt = torch.optim.AdamW(fx_imp.parameters(), lr=lr, weight_decay=weight_decay)
    opt = make_opt_fx(fx=fx_imp, lr=lr, wd=weight_decay)
    loss_fn = nn.MSELoss()

    fx_imp.train()
    if verbose:
        print("Train MSE Phase II -------------")
    for epoch in range(epochs):
        loss_sum, nobs = 0.0, 0
        for RXb, y2b in dl:
            RXb, y2b = RXb.to(device), y2b.to(device)
            opt.zero_grad()
            pred = fx_imp(RXb)
            loss = loss_fn(pred, y2b)
            loss.backward()
            opt.step()
            nobs += RXb.size(0)
            loss_sum += float(loss.item()) * RXb.size(0)

        # compute average training loss across the epochs
        train_mse = loss_sum / max(nobs, 1)
        if verbose and (epoch % 50 == 0):
            print(f"Epoch {epoch}/{epochs} - fx_imp={train_mse:.6f}")


def train_fx(
    fx: MLP,
    X: torch.Tensor,  # full X for fx
    y: torch.Tensor,
    epochs=100,
    batch_size=256,
    lr=1e-3,
    weight_decay=0e-4,
    device="cpu",
    verbose=True,
):
    fx.to(device)

    # precompute residual target y2 = y - f(X)
    ds = TensorDataset(X, y)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    # opt = torch.optim.AdamW(fx.parameters(), lr=lr, weight_decay=weight_decay)
    opt = make_opt_fx(fx=fx, lr=lr, wd=weight_decay)
    loss_fn = nn.MSELoss()

    fx.train()
    if verbose:
        print("Train MSE -------------")
    for epoch in range(epochs):
        loss_sum, nobs = 0.0, 0
        for Xb, yb in dl:
            Xb, yb = Xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = fx(Xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            nobs += Xb.size(0)
            loss_sum += float(loss.item()) * Xb.size(0)

        # compute average training loss across the epochs
        train_mse = loss_sum / max(nobs, 1)
        if verbose and (epoch % 50 == 0):
            print(f"Epoch {epoch}/{epochs} - fx={train_mse:.6f}")


# ---- Prediction functions ------------------
@torch.no_grad()
def predict_full(
    fx: MLP,
    fx_imp: MLP,
    X: torch.Tensor,
    RX: torch.Tensor,
    y_mean: float = 0.0,
    device="cpu",
) -> np.ndarray:
    fx.eval()
    fx_imp.eval()
    fx.to(device)
    fx_imp.to(device)
    out = y_mean + fx(X.to(device)) + fx_imp(RX.to(device))
    return out.detach().squeeze(-1).cpu().numpy()


@torch.no_grad()
def predict_fx_only(
    fx: MLP, X: torch.Tensor, y_mean: float = 0.0, device="cpu"
) -> np.ndarray:
    fx.eval()
    fx.to(device)
    out = y_mean + fx(X.to(device))
    return out.detach().squeeze(-1).cpu().numpy()


# ---- helpers
def param_groups_no_bias_decay(module, wd):
    decay, no_decay = [], []
    for n, p in module.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if n.endswith("bias") else decay).append(p)
    return [
        dict(params=decay, weight_decay=wd),
        dict(params=no_decay, weight_decay=0.0),
    ]


# Joint optimizer for f(X) and g(V)
def make_opt_fx_gv(fx, gv, lr=1e-3, wd=1.0):
    groups = param_groups_no_bias_decay(fx, wd) + param_groups_no_bias_decay(gv, wd)
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.999), eps=1e-8)


# Optimizer for f(X) and f_imp(X)
def make_opt_fx(fx, lr=1e-3, wd=1.0):
    groups = param_groups_no_bias_decay(fx, wd)
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.999), eps=1e-8)
