# CLAUDE_upgrade.md — 15Paper: Three upgrades (8.0 → 8.5)

## Context

Paper: `/home/rodo/15Paper/docs/paper_neurocomputing_v3.tex`
Results: `/home/rodo/15Paper/results/exp3/experiment3.npz`
Figures: `/home/rodo/15Paper/figures/`

The paper proves that implicit neural models with structural inductive bias reduce sample complexity. It is at 8.0/10 for Neurocomputing. Three targeted upgrades will close the remaining reviewer-facing gaps.

**Execute in order: 1, 2, 3. Each produces concrete output files.**

---

## Upgrade 1: Barron norm decay verification (numerical, closes Prop. 4 gap)

### Problem

Proposition 4 (HAM complexity reduction) is CONDITIONAL on:
```
C_{h*_K} ≤ C' ρ^K
```
where C_f = ∫|ω||f̂(ω)|dω is the Barron norm (spectral variation). This is currently an unverified assumption. Experiment 3 already computed the HAM partial sums. We need to verify the Barron norm decay numerically via FFT.

### Implementation: `src/barron_analysis.py`

**Step 1: Load data and recover individual HAM terms.**

```python
import numpy as np

d = np.load('results/exp3/experiment3.npz')
t_test = d['t_test']          # shape (10000,)
u_ref = d['u_ref_test']       # shape (10000,)
S_K = d['S_K_on_test']        # shape (7, 10000) — partial sums S_0..S_6
K_values = d['K_values']      # [0, 1, 2, 3, 4, 5, 6]

# Recover individual terms: u_0 = S_0, u_k = S_k - S_{k-1}
terms = [S_K[0]]  # u_0
for k in range(1, len(K_values)):
    terms.append(S_K[k] - S_K[k-1])

# Residuals: h*_K = u* - S_K
residuals = [u_ref - S_K[k] for k in range(len(K_values))]
```

**Step 2: Compute Barron norm via FFT.**

For a function f(t) sampled on M points over [0, T]:
```python
def barron_norm(f_values, T):
    M = len(f_values)
    f_hat = np.fft.rfft(f_values)
    freqs = np.fft.rfftfreq(M, d=T/M)
    omega = 2 * np.pi * freqs
    d_omega = omega[1] - omega[0] if len(omega) > 1 else 1.0
    C_f = np.sum(np.abs(omega) * np.abs(f_hat)) * d_omega / M
    return C_f
```

Compute `C_uk` for each term k=0..6 and `C_hK` for each residual K=0..6.

**Step 3: Fit exponential decay and produce figures.**

Fit `log(C_uk) = a + b*k` via `np.polyfit(k_array, log_C_array, 1)`. Report `ρ_Barron = exp(b)`.

Do the same for the residuals `C_hK`.

**Step 4: Print summary table.**

```
k  | ||u_k||_inf  | C_{u_k} (Barron)
---|-------------|------------------
0  |  ...        | ...
1  |  ...        | ...
...

K  | ||h*_K||_inf | C_{h*_K} (Barron) 
---|-------------|-------------------
0  |  ...        | ...
...

Fitted decay: ρ_Barron(terms) = ...
Fitted decay: ρ_Barron(residuals) = ...
Compare with: ρ_sup (from ||u_k||_inf) ≈ 0.45 (Experiment 3)
```

### Output

- `src/barron_analysis.py` — self-contained script
- `figures/fig_barron_terms.pdf` — semilog C_{u_k} vs k with fitted line
- `figures/fig_barron_residuals.pdf` — semilog C_{h*_K} vs K with fitted line
- Print the summary table to stdout

### Figure style

Use the same style as other experiment figures (import from `utils.py`). Semilog y-axis. Markers with fitted dashed line. Colors: BLUE for data points, ORANGE for fit. Label axes: "$k$" or "$K$", "$C_{u_k}$" or "$C_{h^*_K}$". Report ρ in legend.

### Success criterion

If `ρ_Barron ≈ ρ_sup ≈ 0.45` (within a factor of 2), then the Barron norm decays geometrically and Proposition 4 is numerically verified. If ρ_Barron is much larger than ρ_sup, report that honestly — it means the spectral regularity assumption needs revision.

---

## Upgrade 2: Closed-form analytical example

### Problem

All experiments are numerical. A reviewer can ask: "show me ONE case where you can compute S(F), β, the theoretical ratio, AND the empirical ratio, and they all match." We need an example where everything is calculable analytically.

### The example

**Target function:**
```
f*(t) = (1 + 0.5*sin(3t)) * exp(-t),   t ∈ [0, 8]
```

**Known structure:** The exponential envelope exp(-t) is known. Define:
```
F(t, y, z) = y * exp(t) - z
```

**Implicit target:**
```
h*(t) = 1 + 0.5*sin(3t)
```

**Parameters (all calculable):**
- β = 1 (since ∂F/∂z = -1)
- B_exp ∝ C_{f*}: compute numerically via FFT
- B_imp ∝ C_{h*}: compute numerically via FFT
- S(F) = C_{f*} / C_{h*}: compute ratio
- Theoretical sample ratio: (β² * B_imp²) / B_exp² = 1/S(F)²
- Empirical sample ratio: from the actual MLP training (same protocol as Exp 1)

### Implementation: `src/experiment5_analytical.py`

1. Define f*(t) and h*(t) analytically.
2. Compute Barron norms C_{f*} and C_{h*} via FFT (using the `barron_norm` function from Upgrade 1).
3. Compute theoretical S(F) = C_{f*}/C_{h*} and theoretical ratio = 1/S(F)².
4. Run the same MLP training protocol as Experiment 1:
   - Same architecture (3×64, Tanh, Xavier, Adam)
   - Sample sizes N ∈ {10, 20, 50, 100, 200, 500, 1000}
   - 20 seeds, σ = 0.01
   - Explicit loss: (y - MLP(t))²
   - Implicit loss: (y*exp(t) - MLP(t))², reconstruct f̂(t) = MLP(t)*exp(-t)
5. Compute empirical sample ratio via MSE threshold interpolation (same as Exp 1).
6. Compare theoretical vs empirical ratio.

### Output

- `src/experiment5_analytical.py`
- `results/exp5/experiment5.npz` — raw data
- `figures/fig_exp5_analytical.pdf` — learning curves (same format as Exp 1)
- Print to stdout:
  ```
  Barron norm C_{f*} = ...
  Barron norm C_{h*} = ...
  Structural content S(F) = C_{f*}/C_{h*} = ...
  β = 1.0
  Theoretical ratio N_imp/N_exp = 1/S(F)² = ...
  Empirical ratio N_imp/N_exp = ...
  ```

### Success criterion

The empirical ratio should be in the same order of magnitude as the theoretical ratio. Exact match is NOT expected (Rademacher bounds are loose). A factor of 2-5× between theoretical and empirical is normal and publishable. Report both numbers honestly.

---

## Upgrade 3: Related work paragraph + fix phantom reference

### Problem

The paper cites Neyshabur et al. [8] in the bibliography but never in the text. It also lacks references to two important post-2018 works in generalization theory:

- Arora, Sanjeev, Rong Ge, Behnam Neyshabur, and Yi Zhang. "Stronger generalization bounds for deep nets via a compression approach." ICML 2018.
- E, Weinan and Stephan Wojtowytsch. "Representation formulas and pointwise properties for Barron functions." Calculus of Variations and PDE, 61(2), 2022.

### What to do

**Step 1:** Add the two new references to the bibliography in `docs/paper_neurocomputing_v3.tex`:

```latex
\bibitem{Arora2018}
S.~Arora, R.~Ge, B.~Neyshabur, Y.~Zhang,
Stronger generalization bounds for deep nets via a compression
approach,
in: \emph{Proc.\ ICML}, 2018, pp.~254--263.

\bibitem{EWojtowytsch2022}
W.~E, S.~Wojtowytsch,
Representation formulas and pointwise properties for Barron
functions,
\emph{Calc.\ Var.\ Partial Differ.\ Equ.} 61~(2) (2022) 46.
```

**Step 2:** Add a "Related work" subsection at the end of the Introduction (after the Contributions list, before Section 2). Insert this text:

```latex
\subsection{Related work}

The generalization bounds derived here build on the norm-based
capacity framework of Neyshabur et al.~\cite{Neyshabur2015} and the
size-independent Rademacher bounds of Golowich et
al.~\cite{Golowich2018}.
Arora et al.~\cite{Arora2018} obtained tighter bounds via a
compression approach, showing that networks that can be compressed
after training generalize better; the implicit formulation studied
here achieves an analogous effect by compressing the learning target
\emph{before} training.
The Barron norm used to characterize approximation complexity was
introduced by Barron~\cite{Barron1993} and has been given a
systematic treatment by E and Wojtowytsch~\cite{EWojtowytsch2022},
who established representation formulas and pointwise regularity
for Barron functions; their results provide the function-space
foundation for Assumption~\ref{hip:reduccion}.
The implicit model formulation has structural parallels with Deep
Equilibrium Models~\cite{Bai2019}, which define the output as a
fixed point of an implicit layer; however, DEQs operate on the
forward pass (the network still learns $f^*$), while our framework
operates on the learning target (the network learns a simpler
$h^*$).
```

**Step 3:** Verify that `\cite{Neyshabur2015}` now appears in the text (it was a phantom reference before).

### Output

Modified `docs/paper_neurocomputing_v3.tex` with the related work subsection and two new bibliography entries.

### Verification

Compile with `pdflatex` twice and check: zero undefined references, zero phantom bibliography entries, related work subsection appears between Contributions and Section 2.

---

## Execution order and time estimates

| Upgrade | Estimated time | Depends on |
|---------|---------------|------------|
| 1. Barron FFT | 30 min | Experiment 3 data |
| 2. Analytical example | 1-2 hours (MLP training) | Upgrade 1 (barron_norm function) |
| 3. Related work | 10 min | Nothing |

Run Upgrade 3 first (fastest), then 1, then 2.

## After all three are done

Print a final summary:
```
UPGRADE SUMMARY
===============
1. Barron decay: ρ_Barron = ..., ρ_sup = 0.45, match: YES/NO
2. Analytical: S(F) = ..., theoretical ratio = ..., empirical ratio = ...
3. Related work: added, Neyshabur cited, zero phantom refs
```

Then update the paper `docs/paper_neurocomputing_v3.tex`:
- In the Discussion of Experiment 3, add: "The Barron norm $C_{u_k}$ was computed via FFT and found to decay geometrically with rate $\rho_{\mathrm{Barron}} \approx \ldots$, consistent with the supremum-norm decay rate $\rho \approx 0.45$, providing numerical support for the spectral regularity condition of Proposition~4."
- Add Experiment 5 to Table 1 with the analytical example results.
- Add `fig_barron_residuals.pdf` as a new figure (Figure 3) with caption explaining the Barron norm decay.
