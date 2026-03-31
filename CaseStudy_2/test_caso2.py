"""
test_caso2.py — Tests for Case Study 2: Nonlinearity absorption.

Validates: advantage is inconsistent (β=4.48 negates path norm reduction).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from utils import RESULTS_DIR


def test_data_exists():
    path = os.path.join(RESULTS_DIR, 'exp2', 'experiment2.npz')
    assert os.path.exists(path), f"Missing {path}"
    print("✓ Test 1: Data file exists")


def test_inconsistent_advantage():
    """Advantage should NOT be consistent across all N (gain condition violated)."""
    d = np.load(os.path.join(RESULTS_DIR, 'exp2', 'experiment2.npz'))
    me = d['mse_explicit'].mean(axis=1)
    mi = d['mse_implicit'].mean(axis=1)
    ratios = me / mi

    # At least some N where implicit is worse (ratio < 1)
    n_worse = np.sum(ratios < 1.0)
    assert n_worse >= 1, f"Expected inconsistent advantage, but implicit won everywhere"
    print(f"✓ Test 2: Inconsistent advantage ({n_worse} sizes where implicit is worse)")


if __name__ == "__main__":
    print("=" * 60)
    print("TESTS — Case Study 2: Nonlinearity absorption")
    print("=" * 60)
    test_data_exists()
    test_inconsistent_advantage()
    print("\n✓ All tests passed")
