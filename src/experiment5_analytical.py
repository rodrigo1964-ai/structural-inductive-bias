"""Experiment 5: Closed-form analytical example (Upgrade 2).

Target: f*(t) = (1 + 0.5*sin(3t)) * exp(-t),  t in [0, 8]
Known structure: F(t,y,z) = y*exp(t) - z  =>  h*(t) = 1 + 0.5*sin(3t)
beta = 1 (since dF/dz = -1)

All quantities (S(F), beta, theoretical ratio) are calculable analytically
via FFT, and compared with the empirical MLP training ratio.
"""

import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from models import MLP, train_model, evaluate_model
from utils import (set_seed, set_figure_style, save_figure, save_results,
                   single_column_fig, BLUE, ORANGE, GREEN,
                   FIGURES_DIR, RESULTS_DIR, PROJECT_ROOT)
from barron_analysis import barron_norm

# ─── System parameters ───
T_MAX = 8.0
LAMBDA = 1.0

# ─── Hyperparameters (identical to Experiment 1) ───
SAMPLE_SIZES = [10, 20, 50, 100, 200, 500, 1000]
N_SEEDS = 20
N_TEST = 10000
SIGMA = 0.01
EPOCHS = 5000
LR = 1e-3
PATIENCE = 500
HIDDEN_DIM = 64
NUM_LAYERS = 3


def h_star(t):
    """Implicit target: h*(t) = 1 + 0.5*sin(3t)."""
    return 1.0 + 0.5 * np.sin(3 * t)


def f_star(t):
    """Original target: f*(t) = h*(t) * exp(-t)."""
    return h_star(t) * np.exp(-LAMBDA * t)


def explicit_loss(pred, y_target, t_input):
    return torch.mean((pred - y_target) ** 2)


def implicit_loss(pred, y_target, t_input):
    """Loss: (MLP(t)*exp(-t) - y)^2. MLP learns h*(t)."""
    f_hat = pred * torch.exp(-LAMBDA * t_input)
    return torch.mean((f_hat - y_target) ** 2)


def implicit_reconstruct(mlp_output, t_test):
    return mlp_output * np.exp(-LAMBDA * t_test)


def run_experiment():
    print("=" * 60)
    print("Experiment 5: Closed-form Analytical Example")
    print("=" * 60)

    # ── Step 1: Compute Barron norms ──
    t_dense = np.linspace(0, T_MAX, N_TEST)
    f_vals = f_star(t_dense)
    h_vals = h_star(t_dense)

    C_fstar = barron_norm(f_vals, T_MAX)
    C_hstar = barron_norm(h_vals, T_MAX)
    S_F = C_fstar / C_hstar
    beta = 1.0
    theoretical_ratio = (beta ** 2) / (S_F ** 2)

    print(f"\nBarron norm C_{{f*}} = {C_fstar:.6f}")
    print(f"Barron norm C_{{h*}} = {C_hstar:.6f}")
    print(f"Structural content S(F) = C_{{f*}}/C_{{h*}} = {S_F:.4f}")
    print(f"beta = {beta}")
    print(f"Theoretical ratio N_imp/N_exp = 1/S(F)^2 = {theoretical_ratio:.6f}")

    # ── Step 2: MLP training ──
    t_test = t_dense
    f_true = f_vals

    mse_explicit = np.zeros((len(SAMPLE_SIZES), N_SEEDS))
    mse_implicit = np.zeros((len(SAMPLE_SIZES), N_SEEDS))

    for i, N in enumerate(SAMPLE_SIZES):
        print(f"\n  N = {N}")
        for s in range(N_SEEDS):
            set_seed(s)
            t_train = np.sort(np.random.uniform(0, T_MAX, N))
            y_train = f_star(t_train) + np.random.randn(N) * SIGMA

            lr = 1e-4 if N < 30 else LR
            epochs = 8000 if N < 30 else EPOCHS

            # Explicit
            set_seed(s)
            model_exp = MLP(hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS)
            train_model(model_exp, t_train, y_train, explicit_loss,
                        epochs=epochs, lr=lr, patience=PATIENCE)
            mse_e, _ = evaluate_model(model_exp, t_test, f_true)
            mse_explicit[i, s] = mse_e

            # Implicit
            set_seed(s)
            model_imp = MLP(hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS)
            train_model(model_imp, t_train, y_train, implicit_loss,
                        epochs=epochs, lr=lr, patience=PATIENCE)
            mse_i, _ = evaluate_model(model_imp, t_test, f_true, implicit_reconstruct)
            mse_implicit[i, s] = mse_i

            if (s + 1) % 5 == 0:
                print(f"    seed {s+1}/{N_SEEDS}: exp={mse_e:.2e}, imp={mse_i:.2e}")

    # Save results
    os.makedirs(os.path.join(RESULTS_DIR, 'exp5'), exist_ok=True)
    save_results('experiment5.npz', 'exp5',
                 sample_sizes=np.array(SAMPLE_SIZES),
                 mse_explicit=mse_explicit,
                 mse_implicit=mse_implicit,
                 C_fstar=np.array(C_fstar),
                 C_hstar=np.array(C_hstar),
                 S_F=np.array(S_F),
                 theoretical_ratio=np.array(theoretical_ratio))

    # ── Step 3: Compute empirical ratio ──
    empirical_ratio = compute_empirical_ratio(mse_explicit, mse_implicit)

    print(f"\n{'='*60}")
    print(f"ANALYTICAL EXAMPLE SUMMARY")
    print(f"{'='*60}")
    print(f"Barron norm C_{{f*}} = {C_fstar:.6f}")
    print(f"Barron norm C_{{h*}} = {C_hstar:.6f}")
    print(f"Structural content S(F) = C_{{f*}}/C_{{h*}} = {S_F:.4f}")
    print(f"beta = {beta}")
    print(f"Theoretical ratio N_imp/N_exp = 1/S(F)^2 = {theoretical_ratio:.6f}")
    print(f"Empirical ratio N_imp/N_exp = {empirical_ratio:.6f}")
    if empirical_ratio > 0:
        print(f"Ratio of ratios (empirical/theoretical) = {empirical_ratio/theoretical_ratio:.2f}")

    # ── Step 4: Plot ──
    plot_learning_curves(mse_explicit, mse_implicit, S_F, theoretical_ratio,
                         empirical_ratio)

    print("\nExperiment 5 complete.")
    return S_F, theoretical_ratio, empirical_ratio


def compute_empirical_ratio(mse_explicit, mse_implicit):
    """Compute empirical N_imp/N_exp ratio via MSE threshold interpolation."""
    N = np.array(SAMPLE_SIZES, dtype=float)
    mean_exp = mse_explicit.mean(axis=1)
    mean_imp = mse_implicit.mean(axis=1)

    ratios = []
    for i in range(len(N)):
        target_mse = mean_imp[i]
        if target_mse < mean_exp[-1] or target_mse > mean_exp[0]:
            continue
        log_mse_exp = np.log(mean_exp)
        log_N = np.log(N)
        log_target = np.log(target_mse)
        for j in range(len(N) - 1):
            if log_mse_exp[j] >= log_target >= log_mse_exp[j + 1]:
                alpha = (log_target - log_mse_exp[j]) / (log_mse_exp[j + 1] - log_mse_exp[j])
                N_exp_interp = np.exp(log_N[j] + alpha * (log_N[j + 1] - log_N[j]))
                ratios.append(N[i] / N_exp_interp)
                break

    return np.mean(ratios) if ratios else 0.0


def plot_learning_curves(mse_explicit, mse_implicit, S_F, theo_ratio, emp_ratio):
    set_figure_style()
    fig = single_column_fig(0.85)
    ax = fig.add_subplot(111)

    N = np.array(SAMPLE_SIZES)
    mean_exp = mse_explicit.mean(axis=1)
    std_exp = mse_explicit.std(axis=1)
    mean_imp = mse_implicit.mean(axis=1)
    std_imp = mse_implicit.std(axis=1)

    ax.loglog(N, mean_exp, 'o-', color=BLUE, label='Explicit', linewidth=1.5, markersize=4)
    ax.fill_between(N, mean_exp - std_exp, mean_exp + std_exp, color=BLUE, alpha=0.2)
    ax.loglog(N, mean_imp, 's--', color=ORANGE, label='Implicit', linewidth=1.5, markersize=4)
    ax.fill_between(N, mean_imp - std_imp, mean_imp + std_imp, color=ORANGE, alpha=0.2)

    # Annotate with ratios
    ax.text(0.05, 0.05,
            f'$\\mathcal{{S}}(F) = {S_F:.2f}$\n'
            f'Theo. ratio $= {theo_ratio:.4f}$\n'
            f'Emp. ratio $= {emp_ratio:.4f}$',
            transform=ax.transAxes, fontsize=8, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel('Training samples $N$')
    ax.set_ylabel('Test MSE')
    ax.legend()
    save_figure(fig, 'fig_exp5_analytical.pdf')


if __name__ == '__main__':
    run_experiment()
