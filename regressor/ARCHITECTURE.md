# ARCHITECTURE.md — Homotopy Regressors Library
## Unified Discrete + Continuous HAM for Nonlinear ODE Systems

**Author:** Rodolfo H. Rodrigo — UNSJ / INAUT  
**Date:** Marzo 2026

---

## 1. Visión

Una librería Python unificada que implementa el **Homotopy Analysis Method (HAM)** 
en sus dos paradigmas fundamentales:

| Aspecto | Discreto (Rodrigo) | Continuo (Liao) |
|---|---|---|
| **Dominio** | Tiempo discreto, paso-a-paso | Tiempo continuo, series analíticas |
| **Derivadas** | BDF 3/4 puntos hacia atrás | Simbólicas exactas (SymPy) |
| **Convergencia** | Orden de truncamiento p + paso T | Parámetro ℏ + orden de truncamiento M |
| **Salida** | Arrays numéricos (NumPy) | Expresiones simbólicas (SymPy) + evaluación numérica |
| **Aplicación** | Control embebido, tiempo real, ESP32 | Análisis cualitativo, soluciones cerradas, BVP |
| **Ventaja** | O(1) por paso, sin matrices grandes | Convergencia ajustable, validez global |

**Puente entre ambos:** El regresor discreto puede verse como la evaluación 
punto-a-punto de la serie continua truncada a orden p, con ℏ = -1 y operador 
auxiliar L = derivada discreta BDF.

---

## 2. Estructura del paquete

```
/home/rodo/regressor/
│
├── __init__.py                     # API pública unificada
├── setup.py                        # Instalación pip
├── ARCHITECTURE.md                 # Este archivo
├── CLAUDE.md                       # Contrato original (discreto)
├── README.md                       # Documentación usuario
│
├── # ══════════════════════════════════════════════
├── # MÓDULOS DISCRETOS (paradigma Rodrigo)
├── # ══════════════════════════════════════════════
├── solver.py                       # Solver escalar: solve_order1, solve_order2
├── solver_system.py                # Solver vectorial: solve_system
├── regressor.py                    # Constructor simbólico escalar + inverso
├── regressor_system.py             # Constructor simbólico para sistemas
├── ode_solver.py                   # API step-by-step, variantes 3pt/4pt
│
├── # ══════════════════════════════════════════════
├── # MÓDULOS CONTINUOS (paradigma Liao)
├── # ══════════════════════════════════════════════
├── continuous/
│   ├── __init__.py                 # Exporta API continua
│   ├── ham_series.py               # Motor principal: series HAM orden M
│   ├── convergence.py              # Curva-ℏ, ℏ óptimo, región de convergencia
│   ├── pade.py                     # Aproximantes de Padé [m/n]
│   ├── operators.py                # Operadores auxiliares L predefinidos
│   └── README.md                   # Documentación del módulo continuo
│
├── # ══════════════════════════════════════════════
├── # MÓDULOS COMPARTIDOS
├── # ══════════════════════════════════════════════
├── derivatives.py                  # Fórmulas de derivadas discretas (Taylor)
├── parser.py                       # Parser texto → SymPy
├── identify_parameters.py          # Identificación LIP/No-LIP
├── shooting_jacobian.py            # Jacobiano analítico para LQR
├── verify_regressor.py             # Framework de verificación vs RK45
│
├── # ══════════════════════════════════════════════
├── # EJEMPLOS Y TESTS
├── # ══════════════════════════════════════════════
├── examples.py                     # Ejemplos escalares discretos (tesis)
├── examples_system.py              # Ejemplos de sistemas (contrato CLAUDE.md)
├── examples_continuous.py          # Ejemplos HAM continuo (Blasius, VdP, etc.)
│
├── tests/
│   ├── test_inverse_regressor.py
│   ├── test_regressor_vs_rk4.py
│   ├── test_step_size_effect.py
│   └── test_continuous_ham.py
│
└── UML/                            # Diagramas
```

---

## 3. HAM Continuo — Teoría implementada

### 3.1 Ecuación de deformación de orden cero

Dado el operador no lineal N[u(t)] = 0, la deformación homotópica es:

```
(1 - q) · L[φ(t;q) - u₀(t)] = q · ℏ · H(t) · N[φ(t;q)]
```

donde:
- q ∈ [0,1] es el parámetro de embedding
- L es el operador lineal auxiliar (elegido por el usuario)
- u₀(t) es la aproximación inicial (satisface condiciones iniciales/frontera)
- ℏ ≠ 0 es el parámetro de convergencia-control
- H(t) es la función auxiliar (usualmente H=1)

### 3.2 Ecuación de deformación de orden m

Expandiendo φ(t;q) = Σ uₘ(t)·qᵐ y diferenciando m veces respecto a q:

```
L[uₘ(t) - χₘ · uₘ₋₁(t)] = ℏ · H(t) · Rₘ(u₀, u₁, ..., uₘ₋₁)
```

donde:
- χₘ = 0 si m = 1, χₘ = 1 si m ≥ 2
- Rₘ = (1/(m-1)!) · ∂ᵐ⁻¹N[φ]/∂qᵐ⁻¹|_{q=0}

### 3.3 Algoritmo de resolución (ham_series.py)

```
Input: N[u] = 0, L, u₀(t), ℏ, M (orden máximo)
Output: u_approx(t) = Σₘ₌₀ᴹ uₘ(t)

1. Calcular u₀(t) satisfaciendo condiciones iniciales
2. Para m = 1, 2, ..., M:
   a. Calcular Rₘ a partir de {u₀, ..., uₘ₋₁}
   b. Resolver L[uₘ] = ℏ·Rₘ + χₘ·L[uₘ₋₁]  (ODE lineal)
   c. Aplicar condiciones homogéneas: uₘ(0) = 0 para m ≥ 1
3. Sumar: u_approx = u₀ + u₁ + ... + uₘ
4. (Opcional) Aplicar Padé [M/2, M/2] para acelerar convergencia
```

### 3.4 Operadores auxiliares predefinidos (operators.py)

| Operador | Forma | Uso típico |
|---|---|---|
| `L_derivative` | L[u] = u' | EDOs primer orden |
| `L_second` | L[u] = u'' | EDOs segundo orden |
| `L_damped` | L[u] = u'' + α·u' | Osciladores amortiguados |
| `L_harmonic` | L[u] = u'' + ω²·u | Osciladores |
| `L_custom` | L[u] = usuario | General |

### 3.5 Selección del parámetro ℏ (convergence.py)

1. **Curva-ℏ**: Evaluar u'(0) vs ℏ para detectar plateau de convergencia
2. **ℏ óptimo**: Minimizar residuo cuadrado ∫N[u_approx]² dt
3. **Región de convergencia**: Intervalo [ℏ_min, ℏ_max] donde la serie converge

### 3.6 Aproximantes de Padé (pade.py)

Dada la serie truncada S_M(t) = Σ aₖ·tᵏ, construir [m/n] con m+n = M:

```
[m/n](t) = P_m(t) / Q_n(t)
```

donde P_m, Q_n son polinomios de grado m, n que coinciden con S_M hasta O(t^{m+n+1}).

---

## 4. Conexión Discreto ↔ Continuo

### Teorema (informal): 
El regresor discreto de orden p con paso T es equivalente a:
- HAM continuo con ℏ = -1
- Operador auxiliar L = operador BDF de p puntos
- Un solo paso de deformación (q: 0→1) evaluado en cada tk

Esto significa que la serie continua HAM truncada a orden M, evaluada 
en los puntos t_k = k·T, debe converger a la misma solución que el 
regresor discreto cuando T → 0 y M → ∞.

### Verificación cruzada (en examples_continuous.py):
Para cada ejemplo, comparar:
1. Solución HAM continua (M términos, Padé)
2. Solución regresor discreto (p correcciones, n pasos)
3. Referencia RK45

---

## 5. API pública unificada (__init__.py)

```python
# === Discreto (paradigma Rodrigo) ===
from .solver import solve_order1, solve_order2, solve_order1_numeric
from .solver_system import solve_system, solve_system_numeric
from .regressor import build_regressor_order1, build_regressor_order2
from .regressor import build_inverse_regressor
from .regressor_system import build_system_regressor

# === Continuo (paradigma Liao) ===
from .continuous import ham_solve, ham_solve_system
from .continuous import hbar_curve, optimal_hbar
from .continuous import pade_approximant

# === Herramientas ===
from .parser import parse_ode, parse_and_build, show
from .derivatives import discrete_derivatives
from .identify_parameters import check_lip, identify_lip, identify_nonlip
from .verify_regressor import verify_regressor_vs_rk45, run_suite, print_report
```

---

## 6. Prioridades de implementación

### Fase 1 — Fijar lo existente (discreto)
1. ✅ Unificar interfaz solver_system ↔ regressor_system (10 vs 13 args)
2. ✅ Completar __init__.py
3. ✅ Crear examples_system.py consolidado

### Fase 2 — HAM continuo (nuevo)
4. ham_series.py — Motor de series HAM
5. operators.py — Operadores auxiliares L
6. convergence.py — Curva-ℏ y ℏ óptimo
7. pade.py — Aproximantes de Padé

### Fase 3 — Integración y verificación
8. examples_continuous.py — Ejemplos Blasius, VdP, Lorenz
9. Verificación cruzada discreto vs continuo
10. README.md completo con tutorial

---

*Fin del documento de arquitectura*
*Rodolfo H. Rodrigo / UNSJ-INAUT / Marzo 2026*
