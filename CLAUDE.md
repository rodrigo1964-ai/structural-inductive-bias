# CLAUDE.md — 15Paper Numerical Experiments

## Project Identity

**Paper:** "Structural Inductive Bias Reduces Sample Complexity in Implicit Neural Models for Dynamical Systems"
**Target journal:** Neurocomputing (Elsevier)
**Location:** `/home/rodo/15paper/`
**Purpose:** Generate numerical experiments that validate Theorem 1 and Corollary 1 of the paper. The experiments must demonstrate that implicit models with structural inductive bias achieve lower generalization error with fewer training samples than explicit models, and that the sample complexity ratio matches the theoretical prediction.

---

## What the Paper Proves (Context for Claude)

The paper proves that if a known function $F$ transforms the learning target from $f^*$ to a simpler $h^*$ (with $B_{\text{imp}} < B_{\text{exp}}$), then the generalization bound is tighter and the sample complexity ratio is:

$$\frac{N_{\text{imp}}}{N_{\text{exp}}} = \frac{\beta^2 M_{\text{imp}}^2 B_{\text{imp}}^2}{M_{\text{exp}}^2 B_{\text{exp}}^2}$$

The gain condition is $\mathcal{S}(F) > \beta \cdot M_{\text{imp}} / M_{\text{exp}}$ where $\mathcal{S}(F) = B_{\text{exp}} / B_{\text{imp}}$.

**Critical distinction:** This is NOT a PINN. PINNs regularize the loss but the target stays $f^*$. Here the target CHANGES to $h^* \neq f^*$.

---

## Experiment Design

### Experiment 1: Exponential Envelope Absorption (β = 1)

**System:** Damped oscillator
$$f^*(t) = g(t) \cdot e^{-\lambda t}, \quad t \in [0, T]$$
where $\lambda = 0.5$, $T = 10$, and $g(t) = 2 + 0.3\sin(2t) + 0.1\cos(5t)$ (slowly varying modulation, known ground truth for validation).

**Explicit model:** MLP learns $f^*(t)$ directly.
- Loss: $\frac{1}{N}\sum_i (y_i - \text{MLP}(t_i;\theta))^2$

**Implicit model:** MLP learns $h^*(t) = g(t)$ with $F(t, y, z) = y \cdot e^{\lambda t} - z$.
- Loss: $\frac{1}{N}\sum_i (y_i \cdot e^{\lambda t_i} - \text{MLP}(t_i;\theta))^2$
- Note: $\beta = 1$ for this $F$.

**Protocol:**
1. Generate ground truth: $f^*(t_j)$ on a dense grid of 10000 points (test set).
2. For each sample size $N \in \{10, 20, 30, 50, 75, 100, 150, 200, 300, 500, 750, 1000\}$:
   a. Sample $N$ training points uniformly from $[0, T]$.
   b. Add noise: $y_i = f^*(t_i) + \varepsilon_i$, $\varepsilon_i \sim \mathcal{N}(0, \sigma^2)$ with $\sigma = 0.01$.
   c. Train explicit MLP and implicit MLP with identical architecture.
   d. Evaluate generalization error on test set: $\text{MSE}_{\text{test}} = \frac{1}{10000}\sum_j (f^*(t_j) - \hat{f}(t_j))^2$.
   e. For the implicit model, reconstruct $\hat{f}(t) = \text{MLP}(t;\theta) \cdot e^{-\lambda t}$.
   f. Repeat 20 times with different random seeds. Report mean and std of test MSE.

**MLP architecture (IDENTICAL for both):**
- Input: 1 (scalar $t$)
- Hidden layers: 3 layers × 64 neurons
- Activation: Tanh (not ReLU — we need smooth approximation)
- Output: 1
- Optimizer: Adam, lr=1e-3
- Epochs: 5000 (with early stopping on 20% validation split, patience 500)
- Weight initialization: Xavier uniform

**Expected result:** The implicit model should achieve lower test MSE for any given $N$, and the crossover (where explicit reaches the accuracy that implicit achieves at $N_0$) should be at approximately $N_0 \cdot \mathcal{S}(F)^2$.

**Output figures:**
- `fig_exp1_learning_curves.pdf`: Log-log plot of test MSE vs $N$ for both models, with error bands (±1 std). Two curves should be roughly parallel with a horizontal shift.
- `fig_exp1_predictions.pdf`: For $N = 50$, show $f^*(t)$, explicit prediction, implicit prediction on same axes. The implicit should track better.
- `fig_exp1_ratio.pdf`: Plot empirical $N_{\text{imp}}/N_{\text{exp}}$ (sample sizes needed for same MSE threshold) vs the theoretical ratio.

---

### Experiment 2: Nonlinearity Absorption (β > 1)

**System:** Exponentially transformed smooth function
$$f^*(t) = e^{g(t)}, \quad g(t) = \sin(t) + 0.5\cos(2t), \quad t \in [0, 2\pi]$$

**Explicit model:** MLP learns $f^*(t) = e^{g(t)}$ directly.

**Implicit model:** $F(t, y, z) = e^z - y$, so $h^*(t) = g(t) = \ln(f^*(t))$.
- Loss: $\frac{1}{N}\sum_i (e^{\text{MLP}(t_i;\theta)} - y_i)^2$
- Here $\beta = \sup |e^z| = e^{\sup|g|} = e^{1.5} \approx 4.48$.

**Protocol:** Same as Experiment 1 but with $\sigma = 0.05$ (slightly more noise since $f^*$ has larger range).

**Expected result:** The implicit model should still outperform but with a smaller advantage than Experiment 1 due to $\beta > 1$. The gain condition $\mathcal{S}(F) > \beta \cdot M_{\text{imp}}/M_{\text{exp}}$ should be verified numerically.

**Output figures:**
- `fig_exp2_learning_curves.pdf`: Same format as Exp 1.
- `fig_exp2_beta_effect.pdf`: Comparison of the empirical advantage vs the theoretical prediction including the $\beta$ penalty.

---

### Experiment 3: HAM Residual Learning (Exponential Reduction with K)

**System:** Nonlinear pendulum $\ddot{u} + \sin(u) = 0$, $u(0) = \pi/3$, $\dot{u}(0) = 0$.

**HAM setup:**
- Auxiliary operator: $\mathcal{L}[u] = \ddot{u} + u$
- Initial approximation: $u_0(t) = (\pi/3)\cos(t)$
- Convergence parameter: $\hbar = -1$ (simplest choice)
- Compute $u_0, u_1, \ldots, u_K$ analytically via the standard HAM recurrence (each is a linear ODE — use symbolic or explicit formulas)

**Reference solution:** Solve the pendulum with scipy `solve_ivp` (RK45, rtol=1e-12) on $[0, 10]$.

**For each truncation order $K \in \{0, 1, 2, 3, 4, 5, 6\}$:**
1. Compute HAM partial sum $S_K(t) = \sum_{k=0}^{K} u_k(t)$.
2. Define residual $h^*_K(t) = u^*(t) - S_K(t)$ (from reference solution).
3. Fix $N = 100$ training samples.
4. Explicit model: MLP learns $u^*(t)$ directly.
5. Implicit model: MLP learns $h^*_K(t)$, then reconstruct $\hat{u}(t) = S_K(t) + \text{MLP}(t;\theta)$.
6. Evaluate test MSE of both on dense grid.
7. Repeat 20 times.

**Expected result:** Test MSE of implicit model should decrease exponentially with $K$ (the more HAM terms you compute analytically, the simpler the residual). Explicit model MSE is constant (it doesn't benefit from $K$).

**Output figures:**
- `fig_exp3_mse_vs_K.pdf`: Semilog plot of test MSE vs $K$. Explicit model as horizontal line, implicit as decreasing curve. This is THE figure that validates Proposition 4 (HAM complexity reduction).
- `fig_exp3_residuals.pdf`: Plot $h^*_K(t)$ for $K = 0, 2, 4, 6$ showing how the residual shrinks.
- `fig_exp3_predictions.pdf`: For $K=3$, show reference solution, explicit prediction, implicit prediction.

---

### Experiment 4: Counterexample Verification

**System:** Same as Experiment 1 but with $F(t,y,z) = y - z$ (trivial, no structural content).

**Expected result:** Explicit and implicit models should have IDENTICAL performance. This validates Counterexample 1 ($\mathcal{S}(F) = 1$, no gain).

**Output:** `fig_exp4_trivial.pdf` showing overlapping learning curves.

---

## File Structure

```
/home/rodo/15paper/
├── CLAUDE.md              (this file)
├── src/
│   ├── models.py          (MLP class, training loop, evaluation)
│   ├── systems.py         (dynamical systems: damped oscillator, pendulum)
│   ├── ham.py             (HAM series computation for pendulum)
│   ├── experiment1.py     (exponential envelope)
│   ├── experiment2.py     (nonlinearity absorption)
│   ├── experiment3.py     (HAM residual)
│   ├── experiment4.py     (counterexample)
│   └── utils.py           (plotting, seed management, logging)
├── results/
│   ├── exp1/              (raw data: .npz files)
│   ├── exp2/
│   ├── exp3/
│   └── exp4/
├── figures/               (publication-quality PDFs)
├── docs/
│   └── paper_neurocomputing_v2.tex
└── requirements.txt
```

---

## Technical Constraints

### Dependencies
- Python 3.10+
- PyTorch (for MLP training — GPU not required, problems are 1D)
- NumPy, SciPy (solve_ivp for reference solutions)
- Matplotlib (publication-quality figures)
- SymPy (optional, for symbolic HAM terms in Experiment 3)

### Reproducibility
- Every experiment uses `torch.manual_seed(seed)` and `numpy.random.seed(seed)` with seeds 0–19.
- All hyperparameters are defined as constants at the top of each experiment file, NOT buried in code.
- Raw results saved as `.npz` files so figures can be regenerated without rerunning.

### Figure Standards (Neurocomputing/Elsevier)
- Format: PDF vector graphics
- Width: single column (~8.5 cm) or double column (~17 cm)
- Font size in figures: ≥ 8pt
- Use `plt.rcParams` to set: `font.size=10`, `axes.labelsize=11`, `legend.fontsize=9`
- Color scheme: use colorblind-safe palette (blue=#0072B2, orange=#D55E00, green=#009E73)
- Line styles: solid for explicit, dashed for implicit
- Always include axis labels with units
- Error bands: use `fill_between` with alpha=0.2
- No titles on figures (captions go in the paper)
- Grid: light gray, `alpha=0.3`

### Training Details
- All MLPs must be trained with IDENTICAL hyperparameters in explicit vs implicit comparison. The ONLY difference is the loss function.
- Early stopping is mandatory to avoid overfitting confounds.
- Log training loss and validation loss for every run (saved in results/).
- If training diverges (loss > 100 × initial loss), flag and exclude that run.

---

## Verification Checklist

Before declaring an experiment complete, verify:

1. [ ] Explicit and implicit models have IDENTICAL architecture and hyperparameters
2. [ ] Only the loss function differs between the two
3. [ ] For the implicit model, the reconstruction $\hat{f}(t) = F^{-1}(t, y, \text{MLP}(t;\theta))$ is correct
4. [ ] Test MSE is computed on held-out dense grid, NOT on training points
5. [ ] 20 random seeds, mean ± std reported
6. [ ] Learning curves show log-log scale with proper axis labels
7. [ ] Counterexample (Exp 4) confirms $\mathcal{S}=1$ gives no advantage
8. [ ] All figures saved as PDF in `figures/`
9. [ ] Raw data saved as `.npz` in `results/`
10. [ ] No hardcoded paths — use `os.path.join` relative to project root

---

## What Goes in the Paper

The experimental section should be ~2-3 pages and contain:

1. **Table 1:** Summary of experiments with columns: System, $F(x,y,z)$, $\beta$, $\mathcal{S}(F)$ (empirical), Theoretical ratio, Empirical ratio.

2. **Figure 1** (`fig_exp1_learning_curves.pdf`): The money figure. Log-log MSE vs N showing the horizontal shift between explicit and implicit curves. Caption should reference Corollary 1.

3. **Figure 2** (`fig_exp3_mse_vs_K.pdf`): Semilog MSE vs HAM order K. This validates Proposition 4 and is the strongest visual result.

4. **Figure 3** (`fig_exp1_predictions.pdf` or `fig_exp3_predictions.pdf`): One representative prediction comparison at a "hard" sample size.

5. **One paragraph** on Experiment 4 confirming the counterexample (no figure needed, just state the result).

---

## Known Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| MLP fails to converge for small $N$ | Use Xavier init + smaller lr (1e-4) + more epochs for $N < 30$ |
| Implicit loss landscape harder to optimize | This would actually strengthen the result: even with harder optimization, the implicit model generalizes better |
| HAM terms hard to compute symbolically | For the pendulum, $u_1, u_2, u_3$ have known closed forms. For higher $K$, compute numerically via the discrete HAM recurrence |
| $\beta$ effect in Exp 2 washes out the advantage | This is a valid outcome — report it honestly. The gain condition may not be satisfied for large $\beta$ |
| Overfitting masks generalization differences | Early stopping + validation split handles this. Also: report BOTH train and test MSE |

---

## Priority Order

1. **Experiment 1** — this is the cleanest and most important (β=1, clear prediction)
2. **Experiment 3** — HAM residual, strongest visual result (exponential decay with K)
3. **Experiment 4** — counterexample, quick sanity check
4. **Experiment 2** — β>1 case, important for completeness but less clean

Run Experiment 1 first. If it works, the paper is submittable with just Exp 1 + Exp 3 + Exp 4.
