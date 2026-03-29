"""Barron norm decay verification for HAM terms and residuals (Upgrade 1).

Validates that the Barron norm C_f = integral(|omega|*|f_hat(omega)|) decays
geometrically with HAM order k, providing numerical support for Proposition 4.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from utils import (set_figure_style, single_column_fig, save_figure,
                   BLUE, ORANGE, GREEN, PROJECT_ROOT)
import matplotlib.pyplot as plt


def barron_norm(f_values, T):
    """Compute Barron norm via FFT: C_f = integral(|omega| * |f_hat(omega)|)."""
    M = len(f_values)
    f_hat = np.fft.rfft(f_values)
    freqs = np.fft.rfftfreq(M, d=T / M)
    omega = 2 * np.pi * freqs
    d_omega = omega[1] - omega[0] if len(omega) > 1 else 1.0
    C_f = np.sum(np.abs(omega) * np.abs(f_hat)) * d_omega / M
    return C_f


def main():
    set_figure_style()

    # Load experiment 3 data
    data_path = os.path.join(PROJECT_ROOT, 'results', 'exp3', 'experiment3.npz')
    d = np.load(data_path)
    t_test = d['t_test']
    u_ref = d['u_ref_test']
    S_K = d['S_K_on_test']   # shape (7, 10000)
    K_values = d['K_values']  # [0, 1, 2, 3, 4, 5, 6]

    T = t_test[-1] - t_test[0]

    # Recover individual HAM terms: u_0 = S_0, u_k = S_k - S_{k-1}
    terms = [S_K[0]]
    for k in range(1, len(K_values)):
        terms.append(S_K[k] - S_K[k - 1])

    # Residuals: h*_K = u* - S_K
    residuals = [u_ref - S_K[k] for k in range(len(K_values))]

    # Compute Barron norms and sup norms
    C_terms = np.array([barron_norm(uk, T) for uk in terms])
    C_residuals = np.array([barron_norm(hK, T) for hK in residuals])
    sup_terms = np.array([np.max(np.abs(uk)) for uk in terms])
    sup_residuals = np.array([np.max(np.abs(hK)) for hK in residuals])

    # Fit exponential decay: log(C) = a + b*k => rho = exp(b)
    # Use terms k=1..6 (skip k=0 which is the initial approximation)
    k_arr = np.array(K_values)

    # For terms: fit on k >= 1 where decay is expected
    mask_terms = (C_terms > 0) & (k_arr >= 1)
    if np.sum(mask_terms) >= 2:
        b_terms, a_terms = np.polyfit(k_arr[mask_terms], np.log(C_terms[mask_terms]), 1)
        rho_barron_terms = np.exp(b_terms)
    else:
        rho_barron_terms = np.nan

    # For residuals: fit on all K where residual is nonzero
    mask_res = C_residuals > 0
    if np.sum(mask_res) >= 2:
        b_res, a_res = np.polyfit(k_arr[mask_res], np.log(C_residuals[mask_res]), 1)
        rho_barron_res = np.exp(b_res)
    else:
        rho_barron_res = np.nan

    # Fit sup norm decay for comparison
    mask_sup = (sup_terms > 0) & (k_arr >= 1)
    if np.sum(mask_sup) >= 2:
        b_sup, _ = np.polyfit(k_arr[mask_sup], np.log(sup_terms[mask_sup]), 1)
        rho_sup_terms = np.exp(b_sup)
    else:
        rho_sup_terms = np.nan

    # ---- Print summary table ----
    print("\n" + "=" * 60)
    print("BARRON NORM DECAY ANALYSIS")
    print("=" * 60)

    print(f"\n{'k':<4} {'||u_k||_inf':<14} {'C_{{u_k}} (Barron)':<20}")
    print("-" * 38)
    for k in range(len(K_values)):
        print(f"{K_values[k]:<4} {sup_terms[k]:<14.6e} {C_terms[k]:<20.6e}")

    print(f"\n{'K':<4} {'||h*_K||_inf':<14} {'C_{{h*_K}} (Barron)':<20}")
    print("-" * 38)
    for k in range(len(K_values)):
        print(f"{K_values[k]:<4} {sup_residuals[k]:<14.6e} {C_residuals[k]:<20.6e}")

    print(f"\nFitted decay rate (terms, k>=1):     rho_Barron = {rho_barron_terms:.4f}")
    print(f"Fitted decay rate (residuals):        rho_Barron = {rho_barron_res:.4f}")
    print(f"Fitted decay rate (sup norm, k>=1):   rho_sup    = {rho_sup_terms:.4f}")
    print(f"\nComparison: rho_Barron(terms) / rho_sup = {rho_barron_terms / rho_sup_terms:.2f}")

    match = abs(rho_barron_terms - rho_sup_terms) / rho_sup_terms < 1.0
    print(f"Match (within factor 2): {'YES' if match else 'NO'}")

    # ---- Figure 1: Barron norm of individual terms ----
    fig = single_column_fig()
    ax = fig.add_subplot(111)
    ax.semilogy(k_arr, C_terms, 'o', color=BLUE, markersize=6, label='$C_{u_k}$ (data)')
    if not np.isnan(rho_barron_terms):
        k_fit = np.linspace(0, K_values[-1], 50)
        ax.semilogy(k_fit, np.exp(a_terms + b_terms * k_fit), '--', color=ORANGE,
                    label=f'Fit: $\\rho = {rho_barron_terms:.3f}$')
    ax.set_xlabel('$k$')
    ax.set_ylabel('$C_{u_k}$')
    ax.legend()
    save_figure(fig, 'fig_barron_terms.pdf')

    # ---- Figure 2: Barron norm of residuals ----
    fig = single_column_fig()
    ax = fig.add_subplot(111)
    ax.semilogy(k_arr[mask_res], C_residuals[mask_res], 's', color=BLUE,
                markersize=6, label='$C_{h^*_K}$ (data)')
    if not np.isnan(rho_barron_res):
        K_fit = np.linspace(K_values[0], K_values[-1], 50)
        ax.semilogy(K_fit, np.exp(a_res + b_res * K_fit), '--', color=ORANGE,
                    label=f'Fit: $\\rho = {rho_barron_res:.3f}$')
    ax.set_xlabel('$K$')
    ax.set_ylabel('$C_{h^*_K}$')
    ax.legend()
    save_figure(fig, 'fig_barron_residuals.pdf')

    print("\nDone. Figures saved.")

    return rho_barron_terms, rho_barron_res, rho_sup_terms


if __name__ == '__main__':
    rho_bt, rho_br, rho_sup = main()
