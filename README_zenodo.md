# Structural Inductive Bias Reduces Sample Complexity in Implicit Neural Models for Dynamical Systems

## Description

Source code and experimental data for the paper:

> Rodrigo, R. O. (2026). "Structural Inductive Bias Reduces Sample Complexity in Implicit Neural Models for Dynamical Systems." *Neurocomputing* (submitted).

The paper proves that implicit neural models of the form F(x, y, MLP(x;θ)) = 0, where F encodes known structural knowledge, achieve tighter generalization bounds than explicit models y = MLP(x;θ). The key mechanism is **target transformation**: F changes the learning target from the original function f* to a simpler residual h*, reducing the intrinsic approximation complexity.

## Repository Structure

```
structural-inductive-bias/
├── README.md                          ← This file
├── LICENSE                            ← MIT License
├── requirements.txt                   ← Python dependencies
├── src/                               ← Source code
│   ├── models.py                      ← MLP architecture, training, evaluation
│   ├── systems.py                     ← Dynamical systems definitions
│   ├── ham.py                         ← HAM series for nonlinear pendulum
│   ├── utils.py                       ← Plotting, seeds, figure style
│   ├── experiment1.py                 ← Exponential envelope absorption (β=1)
│   ├── experiment2.py                 ← Nonlinearity absorption (β>1)
│   ├── experiment3.py                 ← HAM residual learning (pendulum)
│   ├── experiment4.py                 ← Counterexample (S(F)=1)
│   ├── experiment5_analytical.py      ← Closed-form analytical verification
│   ├── barron_analysis.py             ← Barron norm FFT verification
│   └── regenerate_figures.py          ← Regenerate all figures from data
└── results/                           ← Experimental data (.npz)
    ├── exp1/experiment1.npz
    ├── exp2/experiment2.npz
    ├── exp3/experiment3.npz
    ├── exp4/experiment4.npz
    └── exp5/experiment5.npz
```

## Experiments

| # | System | F(x,y,z) | β | Key Result |
|---|--------|----------|---|------------|
| 1 | Damped oscillator f* = g(t)exp(-0.5t) | y·exp(0.5t) - z | 1 | 3× advantage |
| 2 | Exponential f* = exp(g(t)) | exp(z) - y | 4.48 | Inconsistent (β penalty) |
| 3 | Nonlinear pendulum (HAM residual) | HAM decomposition | varies | 7× advantage, ρ≈0.45 |
| 4 | Same as Exp 1, trivial F | y - z | 1 | Ratio = 1.000 (control) |
| 5 | Analytical f* = (1+0.5sin3t)exp(-t) | y·exp(t) - z | 1 | S(F)=2.21, verified |

## Reproducing Results

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run experiments

```bash
cd src
python experiment1.py           # ~30 min
python experiment2.py           # ~30 min
python experiment3.py           # ~20 min
python experiment4.py           # ~30 min
python experiment5_analytical.py  # ~15 min
python barron_analysis.py         # ~1 min (uses Exp 3 data)
```

### Regenerate figures only (no retraining)

```bash
cd src
python regenerate_figures.py
```

### Reproducibility

- All experiments use fixed seeds 0–19 (torch + numpy)
- Results are deterministic on CPU
- No GPU required
- Total time: ~2 hours on a modern laptop

## Data Format (.npz files)

**experiment1/2/4.npz:** `sample_sizes` (12,), `mse_explicit` (12,20), `mse_implicit` (12,20)

**experiment3.npz:** `K_values` (7,), `mse_explicit` (20,), `mse_implicit` (7,20), `t_test` (10000,), `u_ref_test` (10000,), `S_K_on_test` (7,10000)

**experiment5.npz:** `sample_sizes` (7,), `mse_explicit` (7,20), `mse_implicit` (7,20), `barron_f_star`, `barron_h_star`, `structural_content`

## Citation

```bibtex
@article{Rodrigo2026structural,
  title={Structural Inductive Bias Reduces Sample Complexity
         in Implicit Neural Models for Dynamical Systems},
  author={Rodrigo, Rodolfo O.},
  journal={Neurocomputing},
  year={2026},
  note={Submitted}
}
```

## License

MIT License — see LICENSE file.

## Contact

Rodolfo O. Rodrigo — rrodrigo@inaut.unsj.edu.ar
Instituto de Automática (INAUT), CONICET–Universidad Nacional de San Juan, Argentina
