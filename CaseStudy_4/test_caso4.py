"""
test_caso4.py — Tests for Case Study 4: Counterexample.

Validates: MSE ratio ≈ 1.0 for all N (no advantage when S(F)=1).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from utils import RESULTS_DIR


def test_data_exists():
    path = os.path.join(RESULTS_DIR, 'exp4', 'experiment4.npz')
    assert os.path.exists(path), f"Missing {path}"
    print("✓ Test 1: Data file exists")


def test_ratio_is_one():
    """MSE_exp / MSE_imp should be ≈ 1.0 for all N."""
    d = np.load(os.path.join(RESULTS_DIR, 'exp4', 'experiment4.npz'))
    me = d['mse_explicit'].mean(axis=1)
    mi = d['mse_implicit'].mean(axis=1)
    ratios = me / mi
    assert np.all(np.abs(ratios - 1.0) < 0.5), \
        f"Ratios deviate from 1.0: {ratios}"
    print(f"✓ Test 2: Ratio ≈ 1.0 (range [{ratios.min():.3f}, {ratios.max():.3f}])")


if __name__ == "__main__":
    print("=" * 60)
    print("TESTS — Case Study 4: Counterexample")
    print("=" * 60)
    test_data_exists()
    test_ratio_is_one()
    print("\n✓ All tests passed")
