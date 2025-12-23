# Adapted from: https://github.com/kohpangwei/group_DRO (MIT License)
import os
import types
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset


class MyLossComputer:

    def __init__(self, n_groups: int, step_size: float, device=None):
        self.n_groups = n_groups
        self.step_size = step_size
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.adv_probs = torch.ones(n_groups, device=self.device) / n_groups  # q
        return None

    def loss(self, yhat, y, group_idx=None):
        """
        yhat: (N,) predictions
        y:    (N,) targets
        group_idx: (N, ) long tensor with values in {0, ..., n_groups-1}
        """
        # ensure float tensors
        y = y.float()
        yhat = yhat.float()

        # Make shapes match
        if yhat.dim() == 2 and yhat.size(-1) == 1:  # if it is (N, 1)
            yhat = yhat.squeeze(-1)  # transform to (N,)
        if y.dim() == 2 and y.size(-1) == 1:  # if it is (N, 1)
            y = y.squeeze(-1)  # transform to (N,)

        # per-sample loss: (N, )
        per_sample_losses = (y - yhat) ** 2

        # per-group average loss
        group_loss, group_count = self.compute_group_avg(per_sample_losses, group_idx)

        # compute overall loss
        actual_loss, _ = self.compute_robust_loss(group_loss, group_count)

        return actual_loss  # Sum q_g * group_loss

    def compute_group_avg(self, losses, group_idx):
        """
        losses: (N,) per-sample losses
        group_idx: (N,) long
        returns:
        group_loss: (G,) mean loss per group (0 for empty groups)
        group_count: (G,) counts per group
        """
        device = losses.device
        G = self.n_groups
        # sums per group
        loss_sum = torch.zeros(G, device=device).scatter_add_(0, group_idx, losses)
        # counts per group
        ones = torch.ones_like(losses, device=device)
        count = torch.zeros(G, device=device).scatter_add_(0, group_idx, ones)
        # mean with zero-safe denom
        denom = torch.clamp(count, min=1.0)
        group_mean = loss_sum / denom
        return group_mean, count

    def compute_robust_loss(self, group_loss, group_count):
        """
        Exponentiated-gradient update of adv_probs (q), then robust loss = q·group_loss.
        """
        # Optional: only let observed groups influence the update
        observed = (group_count > 0).float()

        with torch.no_grad():
            adjusted = group_loss * observed  # unseen groups -> factor = exp(0) = 1
            self.adv_probs *= torch.exp(self.step_size * adjusted.detach())
            self.adv_probs /= self.adv_probs.sum().clamp(min=1e-12)

        robust = (group_loss * self.adv_probs).sum()
        return robust, self.adv_probs


def make_loader_from_numpy(
    X_np: np.ndarray,
    y_np: np.ndarray,
    g_np: np.ndarray,
    batch_size=256,
    shuffle=True,
    device=None,
):
    """
    X_np: (N, ...) float
    y_np: (N, ...) float (match your model’s output shape)
    g_np: (N,) ints in {0,...,G-1}
    """

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X = torch.from_numpy(X_np).float()
    y = torch.from_numpy(y_np).float()
    g = torch.from_numpy(g_np).long()

    ds = TensorDataset(X, y, g)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
    return loader, device


def train_step(model, optimizer, loss_computer: MyLossComputer, batch, device):
    model.train()

    x, y, g = (t.to(device) for t in batch)

    optimizer.zero_grad()
    yhat = model(x)

    loss = loss_computer.loss(yhat, y, g)  # GroupDRO objective: Sum q_g * loss_g
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total, n = 0.0, 0
    for x, y, _ in loader:
        x, y = x.to(device).float(), y.to(device).float()
        yhat = model(x)

        # Make shapes match
        if yhat.dim() == 2 and yhat.size(-1) == 1:  # if it is (N, 1)
            yhat = yhat.squeeze(-1)  # transform to (N,)
        if y.dim() == 2 and y.size(-1) == 1:  # if it is (N, 1)
            y = y.squeeze(-1)  # transform to (N,)

        total += ((yhat - y) ** 2).sum().item()
        n += x.size(0)
    return total / max(n, 1)


def fit_groupdro(
    model,
    train_loader,
    val_loader,
    n_groups,
    epochs=20,
    lr=1e-3,
    step_size=0.01,
    wd=0.0,
    device=None,
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loss_computer = MyLossComputer(
        n_groups=n_groups, step_size=step_size, device=device
    )

    best_val = float("inf")
    for ep in range(1, epochs + 1):
        # --- train ---
        running, nsteps = 0.0, 0
        for batch in train_loader:
            running += train_step(model, optimizer, loss_computer, batch, device)
            nsteps += 1
        train_obj = running / max(nsteps, 1)

        # --- validate (plain MSE) ---
        val_mse = eval_epoch(model, val_loader, device)

        if ep % 50 == 0:
            print(
                f"[{ep:03d}] train_obj={train_obj:.4f} | val_mse={val_mse:.4f} | q={np.round(loss_computer.adv_probs.detach().cpu().numpy(),4)}"
            )

        # if val_mse < best_val:
        #     best_val = val_mse
        #     torch.save(model.state_dict(), "best_groupdro.pt")


# %%
def create_groups(Z: np.ndarray, level: int = 0) -> np.ndarray:
    """
    Return labels in 0..(2^d - 1) for d = Z.shape[1], using recursive median splits.
    Example: for d=2 it returns 4 groups.
    """
    n, d = Z.shape
    labels = np.zeros(n, dtype=int)

    def rec(idx: np.ndarray, lvl: int):
        axis = lvl
        left_mask = Z[idx, axis] <= np.median(Z[idx, axis])
        left_idx = idx[left_mask]
        right_idx = idx[~left_mask]
        # set bit for this level on the right branch
        labels[right_idx] |= 1 << lvl
        if lvl == d - 1:
            return
        rec(left_idx, lvl + 1)
        rec(right_idx, lvl + 1)

    rec(np.arange(n), level)
    return labels


# %%
