"""MLP model, training loop, and evaluation for 15Paper experiments."""

import numpy as np
import torch
import torch.nn as nn


class MLP(nn.Module):
    """Simple feedforward MLP with Tanh activation."""

    def __init__(self, input_dim=1, hidden_dim=64, num_layers=3, output_dim=1):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


def train_model(model, t_train, y_train, loss_fn, epochs=5000, lr=1e-3,
                patience=500, val_fraction=0.2, verbose=False):
    """Train MLP with early stopping on validation loss.

    Parameters
    ----------
    model : MLP
    t_train : np.ndarray, shape (N,)
    y_train : np.ndarray, shape (N,)
    loss_fn : callable(model_output, y_target, t_input) -> scalar loss
        Signature allows implicit losses that depend on t.
    epochs : int
    lr : float
    patience : int
    val_fraction : float
    verbose : bool

    Returns
    -------
    train_losses : list
    val_losses : list
    """
    N = len(t_train)
    n_val = max(1, int(N * val_fraction))
    n_train = N - n_val

    # Shuffle and split
    idx = np.random.permutation(N)
    idx_tr, idx_val = idx[:n_train], idx[n_train:]

    t_tr = torch.tensor(t_train[idx_tr], dtype=torch.float32).unsqueeze(1)
    y_tr = torch.tensor(y_train[idx_tr], dtype=torch.float32).unsqueeze(1)
    t_val = torch.tensor(t_train[idx_val], dtype=torch.float32).unsqueeze(1)
    y_val = torch.tensor(y_train[idx_val], dtype=torch.float32).unsqueeze(1)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    best_state = None
    wait = 0
    train_losses = []
    val_losses = []
    initial_loss = None

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(t_tr)
        loss = loss_fn(pred, y_tr, t_tr)

        if initial_loss is None:
            initial_loss = loss.item()

        # Divergence check
        if loss.item() > 100 * initial_loss:
            if verbose:
                print(f"  Training diverged at epoch {epoch}")
            return train_losses, val_losses

        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        # Validation
        model.eval()
        with torch.no_grad():
            pred_val = model(t_val)
            vloss = loss_fn(pred_val, y_val, t_val).item()
        val_losses.append(vloss)

        if vloss < best_val_loss:
            best_val_loss = vloss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                if verbose:
                    print(f"  Early stop at epoch {epoch}, val_loss={best_val_loss:.2e}")
                break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    return train_losses, val_losses


def evaluate_model(model, t_test, f_true, reconstruct_fn=None):
    """Evaluate model on test set.

    Parameters
    ----------
    model : MLP
    t_test : np.ndarray, shape (M,)
    f_true : np.ndarray, shape (M,) — ground truth f*(t)
    reconstruct_fn : callable or None
        If not None, applies reconstruction: f_hat = reconstruct_fn(model_output, t_test).
        For explicit model, pass None.

    Returns
    -------
    mse : float
    predictions : np.ndarray
    """
    model.eval()
    t_tensor = torch.tensor(t_test, dtype=torch.float32).unsqueeze(1)
    with torch.no_grad():
        pred = model(t_tensor).squeeze().numpy()

    if reconstruct_fn is not None:
        pred = reconstruct_fn(pred, t_test)

    mse = np.mean((f_true - pred) ** 2)
    return mse, pred
