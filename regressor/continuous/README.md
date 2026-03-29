# HAM Continuo — Módulo de Series Analíticas

## Paradigma

Este módulo implementa el HAM clásico de Shijun Liao (SJTU, 1992–presente):
resolución de EDOs no lineales mediante series homotópicas **continuas**
con convergencia controlada por el parámetro ℏ.

## Diferencia con el regresor discreto

| | Discreto (`solver.py`) | Continuo (`continuous/`) |
|---|---|---|
| Dominio | t_k = k·T (discreto) | t ∈ ℝ (continuo) |
| Salida | Array NumPy | Expresión SymPy + evaluación |
| Derivadas | BDF 3/4 puntos | Simbólicas exactas |
| Convergencia | Paso T + orden p | Parámetro ℏ + orden M |
| Velocidad | O(n) por trayectoria | O(M) simbólico (lento) |
| Aplicación | Tiempo real, ESP32 | Análisis, validación |

## Uso rápido

```python
from sympy import symbols
from continuous import ham_solve, pade_from_ham, hbar_curve, optimal_hbar

y, yp, t = symbols('y yp t')

# Definir ecuación: y' + y² = sin(t)  →  N[u] = yp + y² - sin(t) = 0
N = yp + y**2 - sin(t)

# 1. Resolver con HAM (10 términos, ℏ = -1)
result = ham_solve(N, y, yp, t, ic=0.5, hbar=-1.0, M=10)

# 2. Evaluar numéricamente
import numpy as np
t_eval = np.linspace(0, 5, 100)
values = evaluate_series(result, t_eval)

# 3. Mejorar con Padé [5/5]
pade_res = pade_from_ham(result, m=5, n=5)
values_pade = pade_eval(pade_res, t_eval)

# 4. Encontrar ℏ óptimo
opt = optimal_hbar(N, y, yp, t, ic=0.5, M=8)
print(f"ℏ óptimo: {opt['hbar_optimal']}")
```

## Archivos

| Archivo | Contenido |
|---|---|
| `ham_series.py` | Motor principal: `ham_solve`, `ham_solve_system` |
| `operators.py` | Operadores auxiliares L: derivada, armónico, amortiguado |
| `convergence.py` | Curva-ℏ, ℏ óptimo, tabla de convergencia |
| `pade.py` | Aproximantes de Padé [m/n] |

## Referencia

- Liao, S.J. *Beyond Perturbation*, Chapman & Hall/CRC, 2003.
- Liao, S.J. *Homotopy Analysis Method in Nonlinear Differential Equations*, Springer, 2012.
- Liao, S.J. "Notes on the homotopy analysis method", CNSNS, 14:983-997, 2009.
