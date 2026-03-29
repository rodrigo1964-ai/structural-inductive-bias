# CLAUDE.md — Contrato: `identify_parameters`
# Identificación de parámetros del regresor HAM con regularización Tikhonov
#
# Archivo destino: /home/rodo/regressor/identify_parameters.py
# Autor: Rodolfo H. Rodrigo — UNSJ
# Fecha: Marzo 2026
# =============================================================================

## 1. Motivación y distinción respecto a contratos anteriores

Los contratos anteriores asumen que la estructura F(y, y', u; θ) es **completamente
conocida** — incluyendo los parámetros θ. Este módulo aborda el caso opuesto:

    Dado:    mediciones y_meas[k], u_meas[k]  (posiblemente ruidosas)
             estructura simbólica F(y, yp, u; θ) con θ desconocido
    Hallar:  θ = [θ1, θ2, ..., θp]  que mejor explica los datos

La identificación usa exactamente las **mismas derivadas discretas del regresor**
(fórmula de 3 puntos hacia atrás) para construir la matriz de regresión Φ a partir
de los datos medidos, conectando directamente con solver.py y regressor.py.

Relación con tikhonov_identification.py (contrato anterior):
  - Ese módulo regulariza **señales** (suaviza u(t) o y(t))
  - Este módulo regulariza **parámetros** θ (evita sobreajuste, maneja
    ill-conditioning en el sistema Φ·θ = b)

---

## 2. Dos casos según linealidad en parámetros

### Caso LIP — Lineal en Parámetros (cerrado, prioritario)

F(y, yp, ypp, u; θ) = Φ(y, yp, ypp, u) · θ - r(y, yp, ypp, u) = 0

donde Φ es la matriz de regresión (no depende de θ) y r es el residuo
sin parámetros.

Al evaluar en los datos medidos, en cada paso k:
    Φ_k · θ = b_k    donde b_k = r(y_meas[k], yp_k, ypp_k, u_meas[k])

Apilando k = 2, ..., n-1 (derivadas discretas válidas desde k=2):
    Φ · θ = b    (sistema sobredeterminado (n-2) × p)

Tikhonov: min_θ ||Φ·θ - b||² + λ||L·θ||²
→ (Φ'Φ + λ L'L) · θ = Φ'·b    (sistema p × p, SPD)

### Caso No-LIP — No lineal en parámetros (iterativo, secundario)

F(y, yp, ypp, u; θ) no factoriza como Φ·θ.

Problema: min_θ Σ_k F(y_k, yp_k, ypp_k, u_k; θ)² + λ||θ - θ_prior||²

Resolver con Levenberg-Marquardt usando gradiente analítico:
    J_k = ∂F/∂θ  (Jacobiano respecto a θ, calculado por SymPy diff)

---

## 3. Interfaz pública — cinco funciones

### 3.1  `check_lip(F_expr, theta_syms)`

```python
def check_lip(F_expr, theta_syms):
    """
    Verifica si F_expr es lineal en los parámetros theta_syms.

    Parameters
    ----------
    F_expr : sympy.Expr
        Expresión residuo F(y, yp, ypp, u, t; θ) = 0.
    theta_syms : list of sympy.Symbol
        Símbolos de los parámetros a identificar.

    Returns
    -------
    is_lip : bool
        True si F es lineal en todos los θ_i.
    Phi_expr : list of sympy.Expr or None
        Si is_lip=True: lista de p expresiones Φ_i = ∂F/∂θ_i (coeficientes).
        Si is_lip=False: None.
    r_expr : sympy.Expr or None
        Si is_lip=True: residuo sin parámetros r = F|_{θ=0}.
        Si is_lip=False: None.

    Implementación
    --------------
    Para cada θ_i:
        1. coeff_i = diff(F_expr, theta_i)
        2. Si coeff_i.has(theta_i) → NO es LIP en θ_i → retornar False, None, None
        3. Si todos pasan: Phi_expr = [coeff_0, ..., coeff_{p-1}]
                           r_expr = F_expr.subs([(th, 0) for th in theta_syms])
    """
```

### 3.2  `build_phi_matrix(Phi_exprs, r_expr, state_syms, y_data, u_data, T)`

```python
def build_phi_matrix(Phi_exprs, r_expr, state_syms, y_data, u_data, T):
    """
    Construye la matriz de regresión Φ y el vector b a partir de datos medidos.

    Usa exactamente la misma fórmula de 3 puntos del solver:
        yp[k]  = (3*y[k] - 4*y[k-1] + y[k-2]) / (2T)
        ypp[k] = (y[k] - 2*y[k-1] + y[k-2]) / T^2

    Parameters
    ----------
    Phi_exprs : list of sympy.Expr, longitud p
        Coeficientes ∂F/∂θ_i obtenidos de check_lip().
    r_expr : sympy.Expr
        Residuo sin parámetros: r = F|_{θ=0}.
    state_syms : tuple of sympy.Symbol
        Símbolos en el orden: (y, yp, ypp, u, t)
        Para sistemas N-dimensional: (y1, y2, ..., yN, yp1, ..., ypN, ypp1, ..., yppN, u1, ..., uN, t)
        Orden exacto debe coincidir con lambdify.
    y_data : np.ndarray or list of np.ndarray
        Scalar: shape (n,).
        Multivariable: lista de N arrays shape (n,).
    u_data : np.ndarray or list of np.ndarray
        Scalar: shape (n,).
        Multivariable: lista de M arrays shape (n,).
    T : float
        Período de muestreo.

    Returns
    -------
    Phi : np.ndarray, shape (n-2, p)
        Matriz de regresión (columnas = coeficientes de cada parámetro).
    b : np.ndarray, shape (n-2,)
        Vector del lado derecho: b[k] = -r(y[k+2], yp[k+2], ...).
        (índice desplazado: fila 0 de Phi corresponde a paso k=2)
    t_valid : np.ndarray, shape (n-2,)
        Instantes de tiempo correspondientes.

    Implementación
    --------------
    Para k = 2, ..., n-1:
        yp_k  = (3*y[k] - 4*y[k-1] + y[k-2]) / (2T)
        ypp_k = (y[k] - 2*y[k-1] + y[k-2]) / T^2
        args  = (y[k], yp_k, ypp_k, u[k], k*T)
        para j = 0, ..., p-1:
            Phi[k-2, j] = Phi_j_num(*args)
        b[k-2] = -r_num(*args)
    """
```

### 3.3  `identify_lip(F_expr, theta_syms, state_syms, y_data, u_data, T, lam='auto', L=None)`

```python
def identify_lip(F_expr, theta_syms, state_syms, y_data, u_data, T,
                 lam='auto', L=None):
    """
    Identificación de parámetros para sistemas LIP con Tikhonov.

    Resuelve: (Φ'Φ + λ L'L) θ = Φ'b

    Parameters
    ----------
    F_expr : sympy.Expr
        Residuo F(y, yp, ypp, u, t; θ) = 0.
    theta_syms : list of sympy.Symbol
        Parámetros a identificar, longitud p.
    state_syms : tuple of sympy.Symbol
        Ver build_phi_matrix().
    y_data : np.ndarray or list
        Datos medidos de salida.
    u_data : np.ndarray or list
        Datos medidos de entrada.
    T : float
        Período de muestreo.
    lam : float or 'auto'
        Parámetro de regularización.
        'auto': seleccionar por GCV sobre la grilla np.logspace(-6, 3, 20).
    L : np.ndarray or None
        Matriz de regularización (p × p). Default: np.eye(p) (Tikhonov estándar).
        Alternativa: np.diag(theta_scale) para regularización por escala.

    Returns
    -------
    theta_hat : np.ndarray, shape (p,)
        Parámetros identificados.
    info : dict
        'Phi':          np.ndarray shape (n-2, p)
        'b':            np.ndarray shape (n-2,)
        'lam':          float, λ usado
        'lam_auto':     bool
        'residual':     float, ||Φ·θ_hat - b||² / (n-2)  (RMSE de ajuste)
        'condition':    float, cond(Φ'Φ)  (condición sin regularizar)
        'theta_syms':   list of Symbol
        'theta_hat':    dict {sym: value}  (más legible que el array)

    Implementación
    --------------
    1. check_lip(F_expr, theta_syms) → Phi_exprs, r_expr
       Si not is_lip: lanzar ValueError con mensaje claro.
    2. build_phi_matrix(...) → Phi, b
    3. Construir A = Phi.T @ Phi + lam * L.T @ L
    4. rhs = Phi.T @ b
    5. theta_hat = np.linalg.solve(A, rhs)
       (A es p × p, típicamente p << n, np.linalg.solve es aceptable aquí)
    6. Si lam='auto': iterar sobre lam_grid, calcular GCV para cada λ:
          x_lam = solve(Phi'Phi + lam*L'L, Phi'b)
          res = ||Phi @ x_lam - b||^2
          df = trace((Phi'Phi + lam*L'L)^{-1} Phi'Phi)   ← ver Sección 6
          GCV(lam) = (n-2) * res / ((n-2) - df)^2
       Seleccionar lam* = argmin GCV(lam).
    """
```

### 3.4  `identify_nonlip(F_expr, theta_syms, state_syms, y_data, u_data, T, theta0, lam=0.0, max_iter=50, tol=1e-8)`

```python
def identify_nonlip(F_expr, theta_syms, state_syms, y_data, u_data, T,
                    theta0, lam=0.0, max_iter=50, tol=1e-8):
    """
    Identificación de parámetros para sistemas No-LIP con LM regularizado.

    Minimiza: J(θ) = Σ_k F_k(θ)² + λ||θ - θ_prior||²

    Parameters
    ----------
    F_expr : sympy.Expr
    theta_syms : list of sympy.Symbol, longitud p
    state_syms : tuple of sympy.Symbol
    y_data, u_data : np.ndarray or list
    T : float
    theta0 : np.ndarray, shape (p,)
        Punto inicial para la iteración.
    lam : float
        Regularización Tikhonov sobre θ: añade λ*I al sistema LM.
        0.0 = sin regularización (LM puro).
    max_iter : int
    tol : float
        Criterio de convergencia: ||∆θ|| < tol * ||θ||.

    Returns
    -------
    theta_hat : np.ndarray, shape (p,)
    info : dict
        'converged':    bool
        'iterations':   int
        'residuals':    list of float (historia de ||F||² por iteración)
        'theta_history': list of np.ndarray
        'theta_syms':   list
        'theta_hat':    dict

    Algoritmo LM
    ------------
    Construir el Jacobiano simbólico J_F = [∂F/∂θ_1, ..., ∂F/∂θ_p] via SymPy diff.
    Lambdificar J_F con los mismos state_syms.

    En cada iteración:
        1. Evaluar F_k(θ) para k=2,...,n-1 → vector r de longitud (n-2)
        2. Evaluar J_k(θ) para k=2,...,n-1 → matriz J de shape (n-2, p)
        3. Sistema LM: (J'J + (mu + lam)*I) Δθ = -J'r
           donde mu = max(diag(J'J)) * 1e-3 al inicio, adaptado por trust-region
        4. θ = θ + Δθ
        5. Si ||Δθ|| < tol * max(||θ||, 1): convergido

    Estrategia mu (trust-region estándar):
        Si ||F(θ+Δθ)||² < ||F(θ)||²: aceptar, mu /= 3
        Si no: rechazar Δθ, mu *= 2
    """
```

### 3.5  `build_parametric_regressor(F_expr, state_syms, theta_syms, theta_values, order=1)`

```python
def build_parametric_regressor(F_expr, state_syms, theta_syms, theta_values, order=1):
    """
    Sustituye los parámetros identificados en F y construye el regresor HAM.

    Parameters
    ----------
    F_expr : sympy.Expr
        Residuo simbólico con parámetros θ.
    state_syms : list of sympy.Symbol
        Los 10 símbolos estándar del paquete:
        [x, y, z, xp, yp, zp, xpp, ypp, zpp, t]
    theta_syms : list of sympy.Symbol
        Parámetros identificados.
    theta_values : np.ndarray or dict
        Valores identificados. Si array: en el mismo orden que theta_syms.
        Si dict: {sym: valor}.
    order : int
        Orden de la ODE (1 o 2).

    Returns
    -------
    regressor : callable
        Regresor HAM listo para simular. Misma firma que build_system_regressor.
    F_substituted : sympy.Expr
        F con θ sustituidos (para inspección).

    Implementación
    --------------
    1. Construir dict de sustitución: {theta_syms[i]: theta_values[i]}
    2. F_sub = F_expr.subs(subs_dict)
    3. Llamar build_system_regressor([F_sub], state_syms, order)
    4. Retornar (regressor, F_sub)
    """
```

---

## 4. Estructura del archivo

```
identify_parameters.py
├── imports
├── check_lip(F_expr, theta_syms)
├── build_phi_matrix(Phi_exprs, r_expr, state_syms, y_data, u_data, T)
├── identify_lip(...)                    # Caso LIP — cerrado
├── identify_nonlip(...)                 # Caso No-LIP — iterativo
├── build_parametric_regressor(...)      # post-identificación
├── _gcv_score(Phi, b, lam, L)          # auxiliar GCV
└── if __name__ == "__main__":           # suite de demostración (Sección 5)
```

---

## 5. Suite de demostración (`__main__`)

### Test P1 — check_lip: verificar detección automática

```python
from sympy import symbols

y, yp, ypp, u, t = symbols('y yp ypp u t')
a, b, c = symbols('a b c')   # parámetros

# Caso LIP: F = yp + a*y + b*y^2 - u
F_lip = yp + a*y + b*y**2 - u
is_lip, Phi_e, r_e = check_lip(F_lip, [a, b])
assert is_lip == True
assert len(Phi_e) == 2  # [y, y^2]
print(f"  Phi_exprs = {Phi_e}")    # debe ser [y, y^2]
print(f"  r_expr    = {r_e}")      # debe ser yp - u

# Caso No-LIP: F = yp + a*sin(b*y) - u
F_nonlip = yp + a*sin(b*y) - u
is_lip2, _, _ = check_lip(F_nonlip, [a, b])
assert is_lip2 == False
print("  ✓ check_lip detecta LIP y No-LIP correctamente")
```

### Test P2 — LIP scalar: Lotka-Volterra, recuperar [α, β, γ, δ]

Sistema verdadero:
```
x' = 1.0*x - 0.1*x*y       θ_true = [α=1.0, β=0.1]
y' = 0.075*x*y - 1.5*y     θ_true = [δ=0.075, γ=1.5]
```

Procedimiento:
1. Generar trayectorias de referencia con solve_ivp (RK45, tol=1e-9).
2. Agregar ruido gaussiano σ=0.01 a x_meas, y_meas.
3. Plantear F1 = xp - α*x + β*x*y con theta_syms=[α, β].
4. Llamar identify_lip → θ̂_x = [α̂, β̂].
5. Plantear F2 = yp - δ*x*y + γ*y con theta_syms=[δ, γ].
6. Llamar identify_lip → θ̂_y = [δ̂, γ̂].

```python
theta_true_x = [1.0, 0.1]
theta_true_y = [0.075, 1.5]
tol_rel = 0.05   # 5% de error relativo aceptable
```

Criterio:
```
|α̂ - 1.0|  / 1.0   < 5%
|β̂ - 0.1|  / 0.1   < 5%
|δ̂ - 0.075| / 0.075 < 5%
|γ̂ - 1.5|  / 1.5   < 5%
```

Imprimir tabla:
```
Parámetro   Verdadero   Identificado   Error (%)
α           1.0000      X.XXXX         X.XX%
β           0.1000      X.XXXX         X.XX%
δ           0.0750      X.XXXX         X.XX%
γ           1.5000      X.XXXX         X.XX%
```

### Test P3 — LIP 2do orden: Duffing, recuperar [α, β, γ]

Sistema verdadero:
```
y'' + 0.1*y' + 1.0*y + 0.2*y³ = u(t)    →   θ_true = [α=0.1, β=1.0, γ=0.2]
```

Forma F:
```
F = ypp + α*yp + β*y + γ*y³ - u
Phi_exprs = [yp, y, y^3]
r_expr = ypp - u
```

1. Generar datos con u(t) = 0.5*cos(1.2*t), t ∈ [0,30], n=3000.
2. Agregar ruido σ=0.05 a y_meas.
3. identify_lip con lam='auto'.

Criterio: error relativo < 10% en cada parámetro.

### Test P4 — No-LIP scalar: recuperar parámetro en argumento de sin

Sistema verdadero:
```
y' + sin(ω*y) = u(t)    →   θ_true = [ω = 1.5]
```

Forma F:
```python
F = yp + sin(omega*y) - u    # omega es el parámetro
```

No es LIP en ω (ω está dentro del seno).
Usar identify_nonlip con theta0=[1.0], max_iter=100.

Criterio: |ω̂ - 1.5| / 1.5 < 5%.

### Test P5 — build_parametric_regressor: integración post-identificación

Usando los parámetros identificados en Test P2:
1. Construir regresor HAM con build_parametric_regressor.
2. Simular trayectorias con los parámetros identificados.
3. Verificar error contra referencia: max|x_sim - x_ref| < 1e-2.

```python
from regressor_system import build_system_regressor
from sympy import symbols

x, y, z = symbols('x y z')
xp, yp, zp = symbols('xp yp zp')
xpp, ypp, zpp = symbols('xpp ypp zpp')
ts = Symbol('t')
alpha, beta = symbols('alpha beta')

F_lv = xp - alpha*x + beta*x*y
state_syms = [x, y, z, xp, yp, zp, xpp, ypp, zpp, ts]
theta_syms = [alpha, beta]

reg, F_sub = build_parametric_regressor(
    F_lv, state_syms, theta_syms, theta_values=[alpha_hat, beta_hat], order=1
)
# Simular y verificar
```

---

## 6. GCV para selección de λ en identificación LIP

El sistema LIP es: (Φ'Φ + λL'L)θ = Φ'b

La matriz de influencia (hat matrix) es:
    H_λ = Φ (Φ'Φ + λL'L)^{-1} Φ'    shape (n-2) × (n-2)

GCV(λ) = ||( I - H_λ ) b||² / [trace(I - H_λ)]²
        = ||(b - Φ θ̂)||² * (n-2) / (n-2 - trace(H_λ))²

Calcular trace(H_λ) eficientemente:
    trace(H_λ) = trace(Φ (Φ'Φ + λL'L)^{-1} Φ')
               = trace((Φ'Φ + λL'L)^{-1} Φ'Φ)   ← ciclicidad de traza
               = Σ_i σ_i² / (σ_i² + λ)           ← con SVD de Φ: Φ=UΣV'

Implementación eficiente via SVD (O(n·p²)):
```python
def _gcv_score(Phi, b, lam, L=None):
    """GCV via SVD de Phi (L=I asumido si None)."""
    if L is not None:
        # Transformar: Phi_tilde = Phi @ inv(L), b_tilde = b (si L es square)
        # Para L=I: sin cambio
        pass
    U, s, Vt = np.linalg.svd(Phi, full_matrices=False)  # s: singulares
    # theta_hat = V @ diag(s/(s²+lam)) @ U' @ b
    d = s**2 / (s**2 + lam)           # factor de encogimiento
    df = np.sum(d)                      # grados de libertad efectivos
    theta_hat = Vt.T @ np.diag(s / (s**2 + lam)) @ U.T @ b
    residual = np.sum((b - Phi @ theta_hat)**2)
    n = len(b)
    gcv = n * residual / (n - df)**2
    return gcv, theta_hat, df
```

**Usar SVD para todo lam='auto'**. La factorización se hace una sola vez y se
reutiliza para toda la grilla.

---

## 7. Convenciones de nombres en el código

| Concepto                  | Nombre                  |
|---------------------------|-------------------------|
| Parámetros a identificar  | `theta_hat`             |
| Matriz de regresión       | `Phi`                   |
| Vector lado derecho       | `b`                     |
| Coeficientes simbólicos   | `Phi_exprs`             |
| Residuo sin parámetros    | `r_expr`                |
| Error relativo por param  | `rel_err`               |
| Historia de iteraciones   | `theta_history`         |

---

## 8. Actualizar `__init__.py`

Agregar al final de `/home/rodo/regressor/__init__.py`:

```python
from .identify_parameters import (
    check_lip,
    identify_lip,
    identify_nonlip,
    build_parametric_regressor,
)
```

---

## 9. Restricciones

- NO modificar ningún archivo existente.
- `check_lip` usa solo SymPy `diff` — NO evaluación numérica.
- `build_phi_matrix` usa EXACTAMENTE `(3*y[k] - 4*y[k-1] + y[k-2])/(2T)`
  para yp y `(y[k] - 2*y[k-1] + y[k-2])/T**2` para ypp — mismas fórmulas
  que solver.py. NO usar diferencias de otro orden.
- `identify_lip` usa `np.linalg.solve` solo para el sistema p×p final (p << n).
  NO para el sistema n×n.
- La SVD en `_gcv_score` es sobre Φ (forma (n-2)×p) — economical SVD con
  `full_matrices=False`. NO sobre Φ'Φ.
- `identify_nonlip` lanza `RuntimeError` si no converge en `max_iter` iteraciones
  con mensaje: "LM no convergió en {max_iter} iter. Último ||F||² = {val:.3e}".
- NO importar `build_system_regressor` dentro de `build_phi_matrix` ni
  `identify_lip`. Solo importar en `build_parametric_regressor`.

---

## 10. Criterio de aceptación

```bash
cd /home/rodo/regressor
python identify_parameters.py
```

Debe mostrar:

```
Test P1 — check_lip:
  ✓ LIP detectado correctamente
  ✓ No-LIP detectado correctamente

Test P2 — Lotka-Volterra (LIP, σ=0.01):
  Parámetro   Verdadero   Identificado   Error (%)
  α           1.0000      X.XXXX         X.XX%    ✓
  β           0.1000      X.XXXX         X.XX%    ✓
  δ           0.0750      X.XXXX         X.XX%    ✓
  γ           1.5000      X.XXXX         X.XX%    ✓
  λ seleccionado (GCV): X.XXe+XX

Test P3 — Duffing 2do orden (LIP, σ=0.05):
  α, β, γ todos dentro del 10%  ✓

Test P4 — sin(ω·y) No-LIP:
  ω̂ = X.XXXX  (verdadero: 1.5)  error: X.XX%  ✓
  Convergió en XX iteraciones.

Test P5 — Simulación post-identificación (Lotka-Volterra):
  max|x_sim - x_ref| = X.XXe-XX  ✓
  max|y_sim - y_ref| = X.XXe-XX  ✓
```

---

## 11. Flujo de uso completo (ejemplo guía para el desarrollador)

```
Datos medidos (y_meas, u_meas)
        ↓
check_lip(F_expr, theta_syms)
        ↓ LIP?
   Sí ──────────────────────────────────── No
   ↓                                       ↓
build_phi_matrix(...)              identify_nonlip(...)
        ↓                                  ↓
identify_lip(...)              theta_hat (iterativo)
        ↓                                  ↓
   theta_hat                          theta_hat
        ↓                                  ↓
        └──────────┬────────────────────────┘
                   ↓
     build_parametric_regressor(F_expr, state_syms, theta_syms, theta_hat)
                   ↓
         regressor callable (HAM, listo para simular)
                   ↓
         verify_regressor_vs_rk45(rhs, ic, ...)
```

---

*Fin del contrato — Rodolfo H. Rodrigo / UNSJ / Marzo 2026*
