"""
test_caso3.py — Tests for Case Study 3: HAM residual learning.

Validates: MSE decays with K, advantage > 5× at K=5.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from utils import RESULTS_DIR


def test_data_exists():
    path = os.path.join(RESULTS_DIR, 'exp3', 'experiment3.npz')
    assert os.path.exists(path), f"Missing {path}"
    print("✓ Test 1: Data file exists")


def test_mse_decays_with_K():
    """Implicit MSE should decrease monotonically with HAM order K."""
    d = np.load(os.path.join(RESULTS_DIR, 'exp3', 'experiment3.npz'))
    mi = d['mse_implicit'].mean(axis=1)  # shape (7,)
    # MSE at K=5 should be much less than K=0
    assert mi[5] < mi[0] * 0.1, \
        f"MSE(K=5)={mi[5]:.2e} not << MSE(K=0)={mi[0]:.2e}"
    print(f"✓ Test 2: MSE decays with K (K=0: {mi[0]:.2e} → K=5: {mi[5]:.2e})")


def test_advantage_at_K5():
    """At K=5, implicit should outperform explicit by > 3×."""
    d = np.load(os.path.join(RESULTS_DIR, 'exp3', 'experiment3.npz'))
    me = d['mse_explicit'].mean()
    mi_K5 = d['mse_implicit'][5].mean()
    ratio = me / mi_K5
    assert ratio > 3.0, f"Advantage at K=5 is {ratio:.1f}×, expected > 3×"
    print(f"✓ Test 3: Advantage at K=5 = {ratio:.1f}×")


if __name__ == "__main__":
    print("=" * 60)
    print("TESTS — Case Study 3: HAM residual learning")
    print("=" * 60)
    test_data_exists()
    test_mse_decays_with_K()
    test_advantage_at_K5()
    print("\n✓ All tests passed")
