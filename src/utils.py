"""Plotting utilities, seed management, and logging for 15Paper experiments."""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
FIGURES_DIR = os.path.join(PROJECT_ROOT, 'figures')

# Colorblind-safe palette
BLUE = '#0072B2'
ORANGE = '#D55E00'
GREEN = '#009E73'
PURPLE = '#CC79A7'
YELLOW = '#F0E442'
CYAN = '#56B4E9'
RED = '#E69F00'

# Elsevier/Neurocomputing figure settings
def set_figure_style():
    """Configure matplotlib for publication-quality figures."""
    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 11,
        'legend.fontsize': 9,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman', 'Times New Roman', 'DejaVu Serif'],
        'text.usetex': False,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.color': 'gray',
    })


def set_seed(seed):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def single_column_fig(height_ratio=0.75):
    """Create a single-column figure (8.5 cm wide)."""
    w = 8.5 / 2.54  # cm to inches
    return plt.figure(figsize=(w, w * height_ratio))


def double_column_fig(height_ratio=0.45):
    """Create a double-column figure (17 cm wide)."""
    w = 17.0 / 2.54
    return plt.figure(figsize=(w, w * height_ratio))


def save_figure(fig, name, subdir=None):
    """Save figure as PDF in figures/ directory."""
    if subdir:
        path = os.path.join(FIGURES_DIR, subdir, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
    else:
        path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, format='pdf')
    plt.close(fig)
    print(f"  Saved: {path}")


def save_results(filename, subdir, **arrays):
    """Save numpy arrays as .npz file."""
    path = os.path.join(RESULTS_DIR, subdir, filename)
    np.savez(path, **arrays)
    print(f"  Saved: {path}")
