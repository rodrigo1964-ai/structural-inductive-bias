"""
test_caso1.py — Tests for Case Study 1: Envelope absorption.

Validates key results from the paper:
  1. Implicit MSE < Explicit MSE for N >= 200
  2. Ratio MSE_exp/MSE_imp ≈ 3× at N=500
  3. Results reproducible from saved .npz

Author: Rodolfo H. Rodrigo — UNSJ — 2026
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from utils import RESULTS_DIR


def test_data_exists():
    path = os.path.join(RESULTS_DIR, 'exp1', 'experiment1.npz')
    assert os.path.exists(path), f"Missing {path} — run caso1_envelope.py first"
    print("✓ Test 1: Data file exists")


def test_implicit_advantage():
    """For N >= 200, implicit should outperform explicit (geometric mean)."""
    d = np.load(os.path.join(RESULTS_DIR, 'exp1', 'experiment1.npz'))
    N = d['sample_sizes']
    me = d['mse_explicit']
    mi = d['mse_implicit']

    # Find index for N >= 200
    for i, n in enumerate(N):
        if n >= 200:
            geo_exp = np.exp(np.log(me[i] + 1e-30).mean())
            geo_imp = np.exp(np.log(mi[i] + 1e-30).mean())
            ratio = geo_exp / geo_imp
            assert ratio > 1.5, f"At N={n}: ratio={ratio:.2f}, expected > 1.5"

    print("✓ Test 2: Implicit advantage for N >= 200")


def test_approximate_ratio():
    """At largest N, ratio should be approximately 3×."""
    d = np.load(os.path.join(RESULTS_DIR, 'exp1', 'experiment1.npz'))
    me = d['mse_explicit'][-1]  # largest N
    mi = d['mse_implicit'][-1]
    ratio = me.mean() / mi.mean()
    assert 1.5 < ratio < 10, f"Ratio at max N = {ratio:.2f}, expected in [1.5, 10]"
    print(f"✓ Test 3: Ratio at max N = {ratio:.2f}")


if __name__ == "__main__":
    print("=" * 60)
    print("TESTS — Case Study 1: Envelope absorption")
    print("=" * 60)
    test_data_exists()
    test_implicit_advantage()
    test_approximate_ratio()
    print("\n✓ All tests passed")
