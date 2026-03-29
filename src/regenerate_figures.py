"""Regenerate all publication-quality figures from saved .npz data."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils import (set_figure_style, save_figure, single_column_fig, double_column_fig,
                   BLUE, ORANGE, GREEN, CYAN, PURPLE, RESULTS_DIR)
from systems import f_star_exp1, T_MAX_EXP1, pendulum_reference, T_MAX_EXP3
from models import MLP, train_model, evaluate_model
import os


def log_band(data_2d):
    """Compute geometric mean and ±1 std band in log space.

    For log-scale plots, arithmetic mean ± std is wrong.
    Instead: compute mean and std of log(data), then exponentiate.
    """
    log_data = np.log(data_2d + 1e-30)  # avoid log(0)
    log_mean = log_data.mean(axis=-1)
    log_std = log_data.std(axis=-1)
    center = np.exp(log_mean)
    lo = np.exp(log_mean - log_std)
    hi = np.exp(log_mean + log_std)
    return center, lo, hi


# ═══════════════════════════════════════════════════════════════
# Experiment 1: Learning curves
# ═══════════════════════════════════════════════════════════════

def fig_exp1_learning_curves():
    d = np.load(os.path.join(RESULTS_DIR, 'exp1', 'experiment1.npz'))
    N = d['sample_sizes']
    me, mi = d['mse_explicit'], d['mse_implicit']

    set_figure_style()
    fig = single_column_fig(0.85)
    ax = fig.add_subplot(111)

    center_exp, lo_exp, hi_exp = log_band(me)
    center_imp, lo_imp, hi_imp = log_band(mi)

    ax.loglog(N, center_exp, 'o-', color=BLUE, label='Explicit', lw=1.5, ms=4, zorder=3)
    ax.fill_between(N, lo_exp, hi_exp, color=BLUE, alpha=0.15)

    ax.loglog(N, center_imp, 's--', color=ORANGE, label='Implicit', lw=1.5, ms=4, zorder=3)
    ax.fill_between(N, lo_imp, hi_imp, color=ORANGE, alpha=0.15)

    ax.set_xlabel('Training samples $N$')
    ax.set_ylabel('Test MSE')
    ax.legend(loc='upper right')
    ax.set_xlim(8, 1200)
    save_figure(fig, 'fig_exp1_learning_curves.pdf')


def fig_exp1_predictions():
    """Regenerate predictions plot at N=50 (uses seed 0)."""
    from experiment1 import (implicit_loss, implicit_reconstruct, explicit_loss,
                             HIDDEN_DIM, NUM_LAYERS, EPOCHS, LR, PATIENCE, SIGMA)
    from utils import set_seed

    t_test = np.linspace(0, T_MAX_EXP1, 10000)
    f_true = f_star_exp1(t_test)
    N_plot = 50

    set_seed(0)
    t_train = np.sort(np.random.uniform(0, T_MAX_EXP1, N_plot))
    y_train = f_star_exp1(t_train) + np.random.randn(N_plot) * SIGMA

    set_seed(0)
    model_exp = MLP(hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS)
    train_model(model_exp, t_train, y_train, explicit_loss, epochs=EPOCHS, lr=LR, patience=PATIENCE)
    _, pred_exp = evaluate_model(model_exp, t_test, f_true)

    set_seed(0)
    model_imp = MLP(hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS)
    train_model(model_imp, t_train, y_train, implicit_loss, epochs=EPOCHS, lr=LR, patience=PATIENCE)
    _, pred_imp = evaluate_model(model_imp, t_test, f_true, implicit_reconstruct)

    set_figure_style()
    fig = single_column_fig(0.75)
    ax = fig.add_subplot(111)

    ax.plot(t_test, f_true, '-', color='black', label='$f^*(t)$', lw=1.5)
    ax.plot(t_test, pred_exp, '-', color=BLUE, label='Explicit', lw=1.0, alpha=0.85)
    ax.plot(t_test, pred_imp, '--', color=ORANGE, label='Implicit', lw=1.0, alpha=0.85)
    ax.plot(t_train, y_train, '.', color='gray', ms=3, label='Training data', zorder=0)
    ax.set_xlabel('$t$')
    ax.set_ylabel('$f(t)$')
    ax.legend(loc='upper right', ncol=2)
    save_figure(fig, 'fig_exp1_predictions.pdf')


def fig_exp1_ratio():
    d = np.load(os.path.join(RESULTS_DIR, 'exp1', 'experiment1.npz'))
    N = d['sample_sizes'].astype(float)
    me, mi = d['mse_explicit'], d['mse_implicit']
    # Use geometric means for log-scale interpolation
    mean_exp = np.exp(np.log(me + 1e-30).mean(axis=1))
    mean_imp = np.exp(np.log(mi + 1e-30).mean(axis=1))

    # For each MSE threshold from implicit, find N_exp via log-linear interpolation
    ratios, N_imp_vals = [], []
    log_N = np.log(N)
    log_mse_exp = np.log(mean_exp)

    for i in range(len(N)):
        target = mean_imp[i]
        if target >= mean_exp[0] or target <= mean_exp[-1]:
            continue
        log_t = np.log(target)
        for j in range(len(N) - 1):
            if log_mse_exp[j] >= log_t >= log_mse_exp[j + 1]:
                alpha = (log_t - log_mse_exp[j]) / (log_mse_exp[j + 1] - log_mse_exp[j])
                N_exp_interp = np.exp(log_N[j] + alpha * (log_N[j + 1] - log_N[j]))
                ratios.append(N[i] / N_exp_interp)
                N_imp_vals.append(N[i])
                break

    set_figure_style()
    fig = single_column_fig(0.85)
    ax = fig.add_subplot(111)

    if ratios:
        ax.plot(N_imp_vals, ratios, 'o-', color=BLUE, lw=1.5, ms=5,
                label='Empirical $N_{\\rm imp}/N_{\\rm exp}$')
        mean_r = np.mean(ratios)
        ax.axhline(y=mean_r, color=ORANGE, ls='--', lw=1.2,
                    label=f'Mean ratio = {mean_r:.2f}')
        ax.axhline(y=1.0, color='gray', ls=':', lw=0.8, alpha=0.5)
        ax.set_xlabel('$N_{\\rm imp}$')
        ax.set_ylabel('$N_{\\rm imp} / N_{\\rm exp}$')
        ax.legend()

    save_figure(fig, 'fig_exp1_ratio.pdf')


# ═══════════════════════════════════════════════════════════════
# Experiment 2: Learning curves + beta effect
# ═══════════════════════════════════════════════════════════════

def fig_exp2_learning_curves():
    d = np.load(os.path.join(RESULTS_DIR, 'exp2', 'experiment2.npz'))
    N = d['sample_sizes']
    me, mi = d['mse_explicit'], d['mse_implicit']
    beta = d['beta'][0]

    set_figure_style()
    fig = single_column_fig(0.85)
    ax = fig.add_subplot(111)

    center_exp, lo_exp, hi_exp = log_band(me)
    center_imp, lo_imp, hi_imp = log_band(mi)

    ax.loglog(N, center_exp, 'o-', color=BLUE, label='Explicit', lw=1.5, ms=4, zorder=3)
    ax.fill_between(N, lo_exp, hi_exp, color=BLUE, alpha=0.15)

    ax.loglog(N, center_imp, 's--', color=ORANGE,
              label=f'Implicit ($\\beta={beta:.1f}$)', lw=1.5, ms=4, zorder=3)
    ax.fill_between(N, lo_imp, hi_imp, color=ORANGE, alpha=0.15)

    ax.set_xlabel('Training samples $N$')
    ax.set_ylabel('Test MSE')
    ax.legend()
    ax.set_xlim(8, 1200)
    save_figure(fig, 'fig_exp2_learning_curves.pdf')


def fig_exp2_beta_effect():
    d = np.load(os.path.join(RESULTS_DIR, 'exp2', 'experiment2.npz'))
    N = d['sample_sizes']
    me, mi = d['mse_explicit'], d['mse_implicit']

    set_figure_style()
    fig = single_column_fig(0.85)
    ax = fig.add_subplot(111)

    # Use median instead of mean (more robust to outliers)
    med_exp = np.median(me, axis=1)
    med_imp = np.median(mi, axis=1)
    advantage = med_exp / med_imp

    ax.semilogx(N, advantage, 'o-', color=GREEN, lw=1.5, ms=4, label='Empirical advantage (median)')
    ax.axhline(y=1.0, color='gray', ls=':', lw=0.8, label='No advantage')
    ax.set_xlabel('Training samples $N$')
    ax.set_ylabel('MSE$_{\\rm exp}$ / MSE$_{\\rm imp}$')
    ax.legend()
    ax.set_xlim(8, 1200)
    save_figure(fig, 'fig_exp2_beta_effect.pdf')


# ═══════════════════════════════════════════════════════════════
# Experiment 3: MSE vs K, residuals, predictions
# ═══════════════════════════════════════════════════════════════

def fig_exp3_mse_vs_K():
    d = np.load(os.path.join(RESULTS_DIR, 'exp3', 'experiment3.npz'))
    K = d['K_values']
    me = d['mse_explicit']
    mi = d['mse_implicit']

    set_figure_style()
    fig = single_column_fig(0.85)
    ax = fig.add_subplot(111)

    # Explicit: geometric stats (1D array -> need reshape)
    center_exp, lo_exp, hi_exp = log_band(me.reshape(1, -1))

    # Implicit: per-K geometric stats
    center_imp, lo_imp, hi_imp = log_band(mi)

    # Explicit: horizontal band
    ax.axhline(y=center_exp[0], color=BLUE, lw=1.5, label='Explicit', zorder=3)
    ax.axhspan(lo_exp[0], hi_exp[0], color=BLUE, alpha=0.12)

    # Implicit: decreasing curve
    ax.semilogy(K, center_imp, 's--', color=ORANGE, label='Implicit (HAM residual)',
                lw=1.5, ms=5, zorder=3)
    ax.fill_between(K, lo_imp, hi_imp, color=ORANGE, alpha=0.15)

    ax.set_xlabel('HAM order $K$')
    ax.set_ylabel('Test MSE')
    ax.set_xticks(K)
    ax.legend(loc='upper right')
    save_figure(fig, 'fig_exp3_mse_vs_K.pdf')


def fig_exp3_residuals():
    d = np.load(os.path.join(RESULTS_DIR, 'exp3', 'experiment3.npz'))
    t_test = d['t_test']
    u_ref = d['u_ref_test']
    S_K = d['S_K_on_test']  # shape (7, N_TEST)
    K_vals = d['K_values']

    set_figure_style()
    fig = single_column_fig(0.85)
    ax = fig.add_subplot(111)

    colors = [BLUE, GREEN, ORANGE, PURPLE]
    for idx, K in enumerate([0, 2, 4, 6]):
        ki = list(K_vals).index(K)
        residual = u_ref - S_K[ki]
        ax.plot(t_test, residual, color=colors[idx], label=f'$K={K}$', lw=1.0, alpha=0.85)

    ax.set_xlabel('$t$')
    ax.set_ylabel('Residual $h^*_K(t)$')
    ax.legend()
    save_figure(fig, 'fig_exp3_residuals.pdf')


def fig_exp3_predictions():
    """Predictions at K=3 (requires retraining)."""
    from experiment3 import (make_implicit_loss, make_reconstruct, explicit_loss,
                             HIDDEN_DIM, NUM_LAYERS, EPOCHS, LR, PATIENCE, SIGMA, N_TRAIN)
    from ham import compute_ham_terms
    from scipy.interpolate import interp1d
    from utils import set_seed
    import torch

    t_test = np.linspace(0, T_MAX_EXP3, 10000)
    u_ref = pendulum_reference(t_test)

    t_ham = np.linspace(0, T_MAX_EXP3, 20000)
    _, partial_sums = compute_ham_terms(3, t_ham)
    S3_fn = interp1d(t_ham, partial_sums[3], kind='cubic', fill_value='extrapolate')

    set_seed(0)
    t_train = np.sort(np.random.uniform(0, T_MAX_EXP3, N_TRAIN))
    y_train = pendulum_reference(t_train) + np.random.randn(N_TRAIN) * SIGMA

    # Explicit
    set_seed(0)
    model_exp = MLP(hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS)
    train_model(model_exp, t_train, y_train, explicit_loss, epochs=EPOCHS, lr=LR, patience=PATIENCE)
    _, pred_exp = evaluate_model(model_exp, t_test, u_ref)

    # Implicit K=3
    loss_fn = make_implicit_loss(S3_fn)
    recon_fn = make_reconstruct(S3_fn)
    set_seed(0)
    model_imp = MLP(hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS)
    train_model(model_imp, t_train, y_train, loss_fn, epochs=EPOCHS, lr=LR, patience=PATIENCE)
    _, pred_imp = evaluate_model(model_imp, t_test, u_ref, recon_fn)

    set_figure_style()
    fig = single_column_fig(0.75)
    ax = fig.add_subplot(111)
    ax.plot(t_test, u_ref, '-', color='black', label='Reference', lw=1.5)
    ax.plot(t_test, pred_exp, '-', color=BLUE, label='Explicit', lw=1.0, alpha=0.85)
    ax.plot(t_test, pred_imp, '--', color=ORANGE, label='Implicit ($K=3$)', lw=1.0, alpha=0.85)
    ax.set_xlabel('$t$')
    ax.set_ylabel('$u(t)$')
    ax.legend()
    save_figure(fig, 'fig_exp3_predictions.pdf')


# ═══════════════════════════════════════════════════════════════
# Experiment 4: Trivial F
# ═══════════════════════════════════════════════════════════════

def fig_exp4_trivial():
    d = np.load(os.path.join(RESULTS_DIR, 'exp4', 'experiment4.npz'))
    N = d['sample_sizes']
    me, mi = d['mse_explicit'], d['mse_implicit']

    set_figure_style()
    fig = single_column_fig(0.85)
    ax = fig.add_subplot(111)

    center_exp, lo_exp, hi_exp = log_band(me)
    center_imp, lo_imp, hi_imp = log_band(mi)

    ax.loglog(N, center_exp, 'o-', color=BLUE, label='Explicit', lw=1.5, ms=4, zorder=3)
    ax.fill_between(N, lo_exp, hi_exp, color=BLUE, alpha=0.15)

    ax.loglog(N, center_imp, 's--', color=ORANGE, label='Implicit ($\\mathcal{S}=1$)',
              lw=1.5, ms=4, zorder=3)
    ax.fill_between(N, lo_imp, hi_imp, color=ORANGE, alpha=0.15)

    ax.set_xlabel('Training samples $N$')
    ax.set_ylabel('Test MSE')
    ax.legend()
    ax.set_xlim(8, 1200)
    save_figure(fig, 'fig_exp4_trivial.pdf')


if __name__ == '__main__':
    print("Regenerating all figures from saved data...\n")

    print("Experiment 1:")
    fig_exp1_learning_curves()
    fig_exp1_ratio()

    print("\nExperiment 2:")
    fig_exp2_learning_curves()
    fig_exp2_beta_effect()

    print("\nExperiment 3:")
    fig_exp3_mse_vs_K()
    fig_exp3_residuals()

    print("\nExperiment 4:")
    fig_exp4_trivial()

    print("\nFigures requiring retraining (slower):")
    print("  Exp 1 predictions...")
    fig_exp1_predictions()
    print("  Exp 3 predictions...")
    fig_exp3_predictions()

    print("\nAll figures regenerated.")
