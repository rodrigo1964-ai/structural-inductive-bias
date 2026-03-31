"""
test_caso5.py — Tests for Case Study 5: Analytical verification.

Validates: S(F) > 1, empirical ratio confirms theoretical prediction.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from utils import RESULTS_DIR


def test_data_exists():
    path = os.path.join(RESULTS_DIR, 'exp5', 'experiment5.npz')
    assert os.path.exists(path), f"Missing {path}"
    print("✓ Test 1: Data file exists")


def test_structural_content():
    """S(F) = C_{f*}/C_{h*} should be > 1 (meaningful structural content)."""
    d = np.load(os.path.join(RESULTS_DIR, 'exp5', 'experiment5.npz'))
    S_F = float(d['S_F'])
    assert S_F > 1.5, f"S(F) = {S_F:.2f}, expected > 1.5"
    print(f"✓ Test 2: S(F) = {S_F:.2f} > 1")


def test_empirical_confirms_theory():
    """Empirical ratio should be < 1 (implicit needs fewer samples)."""
    d = np.load(os.path.join(RESULTS_DIR, 'exp5', 'experiment5.npz'))
    me = d['mse_explicit'][-1].mean()
    mi = d['mse_implicit'][-1].mean()
    assert mi < me, f"Implicit MSE ({mi:.2e}) should be < Explicit ({me:.2e})"
    print(f"✓ Test 3: Implicit outperforms explicit at max N")


if __name__ == "__main__":
    print("=" * 60)
    print("TESTS — Case Study 5: Analytical verification")
    print("=" * 60)
    test_data_exists()
    test_structural_content()
    test_empirical_confirms_theory()
    print("\n✓ All tests passed")
