# Structural Inductive Bias Reduces Sample Complexity in Implicit Neural Models for Dynamical Systems

**Submitted to** *Neurocomputing* (Elsevier), March 2026.

**Author:** Rodolfo H. Rodrigo — UNSJ / INAUT-CONICET, San Juan, Argentina

---

## Reproducibility Guide

### Requirements
```bash
Python >= 3.10
pip install -r requirements.txt   # torch, numpy, scipy, matplotlib
```

All experiments use 20 random seeds (0–19) for statistical reproducibility.
Results were generated with Python 3.12, PyTorch 2.2, NumPy 1.26, SciPy 1.12
on Ubuntu 24.04.

### Reproduce everything at once
```bash
chmod +x run_all.sh
./run_all.sh              # Tests + figure regeneration (fast, uses saved .npz)
./run_all.sh --tests      # Validation tests only (~1 min)
./run_all.sh --figures    # Regenerate all figures from saved data (~2 min)
./run_all.sh --full       # Re-run ALL experiments from scratch (~2 hours)
```

### Reproduce individual figures and tables

| Paper element | Command | Output |
|:---|:---|:---|
| **Fig. 1** — Learning curves (β=1) | `cd CaseStudy_1 && python generate_figures.py` | `figures/fig_exp1_learning_curves.pdf` |
| **Fig. 2** — Predictions (N=50) | (same as above) | `figures/fig_exp1_predictions.pdf` |
| **Fig. 3** — Sample ratio | (same as above) | `figures/fig_exp1_ratio.pdf` |
| **Fig. 4** — β effect (β=4.48) | `cd CaseStudy_2 && python generate_figures.py` | `figures/fig_exp2_learning_curves.pdf`, `fig_exp2_beta_effect.pdf` |
| **Fig. 5** — MSE vs HAM order K | `cd CaseStudy_3 && python generate_figures.py` | `figures/fig_exp3_mse_vs_K.pdf` |
| **Fig. 6** — HAM residuals | (same as above) | `figures/fig_exp3_residuals.pdf`, `fig_exp3_predictions.pdf` |
| **Fig. 7** — Barron norm decay | (same as above) | `figures/fig_barron_terms.pdf`, `fig_barron_residuals.pdf` |
| **Fig. 8** — Counterexample S=1 | `cd CaseStudy_4 && python generate_figures.py` | `figures/fig_exp4_trivial.pdf` |
| **Fig. 9** — Analytical verification | `cd CaseStudy_5 && python generate_figures.py` | `figures/fig_exp5_analytical.pdf` |
| **Table I** — Summary | All 5 experiments produce the data; table assembled in LaTeX | See §X.6 |

**Note:** Figure numbers above are approximate — check the manuscript for exact numbering.

---

## Repository Structure
```
15Paper/
│
├── CaseStudy_1/              # §X.1: Exponential envelope absorption (β=1)
│   ├── caso1_envelope.py     #   → 3× advantage, validates Prop. factorization
│   ├── generate_figures.py   #   → fig_exp1_learning_curves, predictions, ratio
│   └── test_caso1.py
│
├── CaseStudy_2/              # §X.2: Nonlinearity absorption (β=4.48)
│   ├── caso2_nonlinearity.py #   → inconsistent advantage, validates gain condition
│   ├── generate_figures.py   #   → fig_exp2_learning_curves, beta_effect
│   └── test_caso2.py
│
├── CaseStudy_3/              # §X.3: HAM residual learning (pendulum)
│   ├── caso3_ham_residual.py #   → 7× advantage at K=5, validates Prop. HAM
│   ├── generate_figures.py   #   → fig_exp3_*, fig_barron_*
│   └── test_caso3.py
│
├── CaseStudy_4/              # §X.4: Counterexample (S(F)=1)
│   ├── caso4_counterexample.py # → ratio ≡ 1.0, validates Counterexample 1
│   ├── generate_figures.py   #   → fig_exp4_trivial
│   └── test_caso4.py
│
├── CaseStudy_5/              # §X.5: Closed-form analytical verification
│   ├── caso5_analytical.py   #   → S(F)=2.21, ratio comparison
│   ├── generate_figures.py   #   → fig_exp5_analytical
│   └── test_caso5.py
│
├── src/                      # Shared library
│   ├── models.py             #   3-layer MLP (PyTorch), train_model, evaluate_model
│   ├── systems.py            #   Dynamical systems (damped osc, pendulum, exp)
│   ├── ham.py                #   HAM series computation for nonlinear pendulum
│   ├── utils.py              #   Seeds, figure style, save/load utilities
│   ├── barron_analysis.py    #   Barron norm via FFT, decay rate fitting
│   ├── experiment[1-4].py    #   Full experiment pipelines
│   ├── experiment5_analytical.py
│   └── regenerate_figures.py #   Regenerate all figures from saved .npz
│
├── results/                  # Saved experimental data
│   ├── exp1/experiment1.npz  #   MSE arrays (12 sample sizes × 20 seeds)
│   ├── exp2/experiment2.npz
│   ├── exp3/experiment3.npz  #   + HAM partial sums, reference solution
│   ├── exp4/experiment4.npz
│   └── exp5/experiment5.npz  #   + Barron norms, S(F), theoretical ratio
│
├── docs/                     # Manuscript
│   ├── paper_neurocomputing_v3.tex
│   ├── paper_neurocomputing_v3.pdf
│   ├── figures/              #   12 publication PDFs
│   └── Seminario/            #   Teaching handouts (6 LaTeX+PDF pairs)
│
├── run_all.sh                # ./run_all.sh [--tests|--figures|--full]
├── requirements.txt          # torch, numpy, scipy, matplotlib
├── CITATION.cff
└── LICENSE                   # GPL-3.0
```

Folder `regressor/` is a legacy library from earlier papers and is **not required**
to reproduce any result in this paper.

---

## Experimental Summary

| Exp | System | β | S(F) | Advantage | ρ | Validates |
|:---:|:---|:---:|:---:|:---:|:---:|:---|
| 1 | Damped oscillator | 1 | >1 | 3.0× | — | Prop. factorization |
| 2 | exp(g(t)) | 4.48 | >1 | Inconsistent | — | Gain condition |
| 3 | Nonlinear pendulum (HAM) | varies | >1 | 7.1× | 0.45 | Prop. HAM |
| 4 | Trivial F (identity) | 1 | 1 | 1.00× | — | Counterexample 1 |
| 5 | (1+0.5sin3t)·e⁻ᵗ | 1 | 2.21 | 2.2× | — | Corollary datos |

## How it works

The paper proves that **implicit neural models** of the form F(x, y, MLP(x;θ)) = 0
achieve tighter generalization bounds than explicit models y = MLP(x;θ),
provided the structural content metric S(F) = C_{f*}/C_{h*} exceeds β.

Each `CaseStudy_N/` is a thin wrapper around the experiment code in `src/`:

1. `casoN_*.py` runs the full experiment (12 sample sizes × 20 seeds × 2 models).
2. `generate_figures.py` regenerates publication-quality PDFs from saved `.npz` data.
3. `test_casoN.py` validates key numerical claims from the paper.

The core library in `src/` provides:
- `MLP`: 3-layer (64 neurons, Tanh, Xavier init) trained with Adam + early stopping.
- `train_model`: Generic training loop accepting custom loss functions (explicit or implicit).
- `ham.py`: Numerical HAM series u₀, u₁, ..., u_K for the nonlinear pendulum.
- `barron_analysis.py`: Barron norm C_f via FFT and geometric decay rate fitting.

## Citation
```bibtex
@article{rodrigo2026structural,
  title={Structural Inductive Bias Reduces Sample Complexity in
         Implicit Neural Models for Dynamical Systems},
  author={Rodrigo, Rodolfo H.},
  journal={Neurocomputing},
  year={2026},
  note={Submitted}
}
```

## License

GPL-3.0 — See [LICENSE](LICENSE).
