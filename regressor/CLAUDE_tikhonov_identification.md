# CLAUDE.md — Contrato: `tikhonov_identification`
# Identificación robusta de entrada/salida con ruido en u o en y
#
# Archivo destino: /home/rodo/regressor/tikhonov_identification.py
# Autor: Rodolfo H. Rodrigo — UNSJ
# Fecha: Marzo 2026
# =============================================================================

## 1. Motivación

El TII actual en `/home/rodo/10Paper/src/methods.py` está acoplado al motor DC
(usa `motor_model`, `Phi` hardcodeada) y solo resuelve el caso de ruido en y
para reconstruir u.

Este módulo implementa la versión **genérica y desacoplada**:

  Caso A — Ruido en y, reconstruir u:
      Se conoce la forma funcional F(y, y', u) = 0.
      y[k] está contaminada. HAM inverso da b_k = û_k punto a punto.
      Tikhonov suaviza la secuencia {b_k} → {u_k} regularizada.

  Caso B — Ruido en u, reconstruir y:
      Se conoce la forma funcional F(y, y', u) = 0.
      u[k] está contaminada. HAM directo da b_k = ŷ_k punto a punto.
      Tikhonov suaviza la secuencia {b_k} → {y_k} regularizada.

  Caso multivariable:
      Aplicar independientemente a cada componente yi o ui.
      El módulo acepta listas de arrays.

El núcleo matemático es idéntico en ambos casos:
    min_x ||x - b||² + λ ||D x||²
    →  (I + λ D'D) x = b
donde D es la matriz de diferencias de orden p (p=1 por defecto).
La matriz I + λ D'D es tridiagonal simétrica definida positiva → O(n).

---

## 2. Interfaz pública — cuatro funciones

### 2.1  `tikhonov_smooth(b, lam, order=1)`

Función núcleo. Genérica, no sabe si b viene de u o de y.

```python
def tikhonov_smooth(b, lam, order=1):
    """
    Resuelve  (I + λ·D'D) x = b  con D de diferencias de orden `order`.

    Parameters
    ----------
    b : np.ndarray, shape (n,)
        Estimaciones punto a punto (lado derecho).
        NaN en posiciones de inicialización se reemplazan por 0 antes de resolver.
    lam : float
        Parámetro de regularización λ ≥ 0.
        lam = 0 devuelve x = b sin modificar.
    order : int
        Orden de la diferencia finita en la penalización (1 o 2).
        order=1: penaliza ||Δx||² = ||x_{k+1} - x_k||²  (suavidad de 1er orden)
        order=2: penaliza ||Δ²x||² (suavidad de curvatura)

    Returns
    -------
    x : np.ndarray, shape (n,)
        Secuencia regularizada.

    Notas de implementación
    -----------------------
    Para order=1:
        D'D es tridiagonal: diag=[1,2,...,2,1], off-diag=[-1,...,-1]
        I + λ D'D: diag=[1+λ, 1+2λ,...,1+2λ, 1+λ], off=[-λ,...,-λ]
    Para order=2:
        D'D es pentadiagonal → usar scipy.linalg.solve_banded con (2,2).
    Resolver con scipy.linalg.solve_banded en formato banded.
    NO usar np.linalg.solve (coste O(n³)).
    """
```

### 2.2  `select_lambda(b, lam_grid=None, method='gcv')`

Selección automática de λ.

```python
def select_lambda(b, lam_grid=None, method='gcv'):
    """
    Selecciona λ óptimo por GCV o L-curve.

    Parameters
    ----------
    b : np.ndarray, shape (n,)
        Estimaciones punto a punto (lado derecho del sistema).
    lam_grid : array_like or None
        Grilla de candidatos. Default: np.logspace(-4, 4, 25).
    method : str
        'gcv'    — Generalized Cross-Validation (recomendado)
        'lcurve' — L-curve: selecciona el codo por máxima curvatura
        'grid'   — mínimo RMSE en grilla (requiere verdad conocida, solo tests)

    Returns
    -------
    lam_opt : float
        λ óptimo seleccionado.
    info : dict
        'method': str
        'lam_grid': array
        'scores': array  (GCV o curvatura)
        'lam_opt': float

    Implementación de GCV
    ---------------------
    Para cada λ en la grilla:
        x_lam = tikhonov_smooth(b, lam)
        residual = ||x_lam - b||²
        df = trace(S_lam)  donde S_lam = (I + λ D'D)^{-1}
              (traza de la matriz suavizadora)
        GCV(λ) = n · residual / (n - df)²
    Seleccionar λ* = argmin GCV(λ).

    Para la traza de S_lam en O(n):
        Como S_lam es tridiagonal inversa, su traza no es trivial.
        Aproximar df ≈ n / (1 + λ · ||d||²_2 / n) donde d es la diagonal
        de D'D, o usar la fórmula exacta para tridiagonales via recursión de
        Thomas inversa (ver Sección 6 para algoritmo).
    """
```

### 2.3  `tii_reconstruct_u(y_meas, point_estimator, T, lam='auto', lam_grid=None)`

Caso A: ruido en y, reconstruir u.

```python
def tii_reconstruct_u(y_meas, point_estimator, T, lam='auto', lam_grid=None):
    """
    Tikhonov Integral Inversion genérica — Caso A (ruido en y).

    Parameters
    ----------
    y_meas : np.ndarray or list of np.ndarray
        Scalar: shape (n,).
        Multivariable: lista de N arrays shape (n,), uno por variable de estado.

    point_estimator : callable
        Estimador punto a punto que produce b_k = û_k (sin regularizar).
        Firma: point_estimator(y_meas) -> np.ndarray shape (n,)
        Construido con build_inverse_regressor o inverse_integral.
        El llamador construye este closure con T, parámetros del modelo, etc.

    T : float
        Período de muestreo (usado solo si lam='auto' para escalar lam_grid).

    lam : float or 'auto'
        Si float: usar directamente.
        Si 'auto': llamar select_lambda(b, lam_grid, method='gcv').

    lam_grid : array_like or None
        Solo si lam='auto'. Default interno: np.logspace(-3, 3, 13).

    Returns
    -------
    u_reg : np.ndarray, shape (n,)
        Entrada reconstruida y regularizada.
    info : dict
        'b': array,          estimaciones sin regularizar
        'lam': float,        λ usado
        'lam_selected': bool True si fue auto-seleccionado
        'gcv_info': dict or None
    """
```

### 2.4  `tii_smooth_y(u_known, point_estimator, T, lam='auto', lam_grid=None)`

Caso B: ruido en u, suavizar y.

```python
def tii_smooth_y(u_known, point_estimator, T, lam='auto', lam_grid=None):
    """
    Tikhonov sobre estimaciones directas — Caso B (ruido en u).

    Parameters
    ----------
    u_known : np.ndarray or list of np.ndarray
        Scalar: shape (n,).   Multivariable: lista de N arrays.
        Señal de entrada contaminada por ruido.

    point_estimator : callable
        Estimador punto a punto que produce b_k = ŷ_k (sin regularizar).
        Firma: point_estimator(u_known) -> np.ndarray shape (n,)
        Construido con build_regressor_order1 o solve_system.

    T : float
        Período de muestreo.

    lam : float or 'auto'
        Igual que tii_reconstruct_u.

    Returns
    -------
    y_reg : np.ndarray, shape (n,)
        Salida suavizada y regularizada.
    info : dict   (mismas claves que tii_reconstruct_u)
    """
```

---

## 3. Caso multivariable

Para sistemas con N variables, tanto `tii_reconstruct_u` como `tii_smooth_y`
aceptan y devuelven **listas** de arrays.

```python
# Entrada: lista de N arrays
y_meas = [omega_array, i_array]   # N=2

# El point_estimator recibe la lista y devuelve lista
def estimator(y_meas_list):
    omega, i = y_meas_list
    b_u = inverse_integral_generic(omega, i, T, params)  # shape (n,)
    return b_u   # solo u es escalar aquí

# Para sistemas donde u también es vectorial:
def estimator_multi(y_meas_list):
    return [b_u1, b_u2]   # lista de N_u arrays

# tii_reconstruct_u detecta el tipo de retorno:
# - Si callable devuelve ndarray → caso escalar
# - Si callable devuelve list → aplica tikhonov_smooth a cada componente
#   con el mismo λ (o lista de λ si se pasa lam como lista)
```

---

## 4. Estructura del archivo

```
tikhonov_identification.py
├── imports
├── _build_DtD_banded(n, order)     # auxiliar: construye D'D en formato banded
├── _trace_smoother(n, lam, order)  # auxiliar: traza de (I + λD'D)^{-1} en O(n)
├── tikhonov_smooth(b, lam, order)  # núcleo
├── select_lambda(b, ...)           # selección automática de λ
├── tii_reconstruct_u(...)          # Caso A
├── tii_smooth_y(...)               # Caso B
└── if __name__ == "__main__":      # suite de demostración (Sección 5)
```

No importar nada de `motor_model.py`, `methods.py`, `ekf.py`.
Ese código pertenece a 10Paper. Este módulo es parte de `/home/rodo/regressor/`.

---

## 5. Suite de demostración (`__main__`)

Cuatro tests que no dependen del motor DC. Usar sistemas propios del paquete.

### Test T1 — Caso A scalar: reconstrucción de u con ruido en y

Sistema:  `y' + 0.5*y = u(t)`  con  `u(t) = sin(2t)`

```python
# Ground truth
t = np.linspace(0, 10, 2000);  T = t[1]-t[0]
u_true = np.sin(2*t)
# solve_ivp para obtener y_true
# Agregar ruido gaussiano sigma=0.05 a y_true → y_noisy

# Construir point_estimator con build_inverse_regressor:
#   F = yp + 0.5*y - u   →   b_k = û_k = yp_k + 0.5*y_k (u despejada analíticamente)
# Pasar y_noisy al estimador para obtener b ruidoso
# Aplicar tii_reconstruct_u con lam='auto'

# Criterio: RMSE(u_reg, u_true) < RMSE(b, u_true)
# Imprimir: RMSE sin regularizar, RMSE regularizado, λ seleccionado
```

### Test T2 — Caso B scalar: suavizado de y con ruido en u

Sistema:  `y' + y^2 = u(t)`  con  `u_true(t) = 1 + 0.3*cos(3t)`

```python
# Ground truth: solve_ivp con u_true
# Agregar ruido sigma=0.1 a u_true → u_noisy

# Construir point_estimator con build_regressor_order1:
#   f(y) = y^2,  u = u_noisy
# El estimador resuelve el regresor directo con u_noisy → ŷ_k ruidosa
# Aplicar tii_smooth_y con lam='auto'

# Criterio: RMSE(y_reg, y_true) < RMSE(y_noisy_hat, y_true)
```

### Test T3 — Caso A multivariable: Lotka-Volterra con ruido en x e y

Sistema:
```
x' = α*x - β*x*y  + u1(t)   ← u1 desconocida, a reconstruir
y' = δ*x*y - γ*y  + u2(t)   ← u2 desconocida, a reconstruir
```
Usar u1=u2=0 (autónomo). Agregar ruido sigma=0.02 a las trayectorias.
Reconstruir [u1_reg, u2_reg] con `tii_reconstruct_u` multivariable.

Criterio: max(RMSE u1_reg, RMSE u2_reg) < 0.05

### Test T4 — Comparar métodos de selección de λ

Usar Test T1. Ejecutar con:
- `lam='auto', method='gcv'`
- `lam='auto', method='lcurve'`
- `lam=1.0` (fijo, referencia)
- `lam=100.0` (sobre-regularizado)
- `lam=0.001` (sub-regularizado)

Imprimir tabla:

```
λ selección    λ usado    RMSE u_reg
-----------    -------    ----------
GCV            X.XXe+YY   X.XXe-ZZ   ← debe ser el mejor
L-curve        X.XXe+YY   X.XXe-ZZ
Fijo 1.0       1.00e+00   X.XXe-ZZ
Fijo 100       1.00e+02   X.XXe-ZZ
Fijo 0.001     1.00e-03   X.XXe-ZZ
```

---

## 6. Algoritmo de traza para GCV en O(n)

Para la traza de `S_λ = (I + λ D'D)^{-1}` (necesaria en GCV):

La matrix `A = I + λ D'D` es tridiagonal simétrica SPD.
La traza de `A^{-1}` se obtiene de la factorización LDL' de Thomas:

```python
def _trace_smoother(n, lam, order=1):
    """
    Calcula trace((I + lam*D'D)^{-1}) en O(n) para order=1.

    Algoritmo: factorizar A = LDL' con el algoritmo de Thomas,
    luego usar la fórmula de la traza vía los cofactores diagonales.

    Para order=1, A es tridiagonal:
        diag = [1+lam, 1+2*lam, ..., 1+2*lam, 1+lam]
        off  = [-lam, ..., -lam]

    La traza de A^{-1} = sum_i (A^{-1})_{ii}.
    Usando la fórmula de Cramer para tridiagonales:
        (A^{-1})_{ii} = (producto de cofactores) / det(A)
    Se calcula eficientemente con la recursión de determinantes:
        d_0 = 1
        d_k = a_k * d_{k-1} - b_{k-1}^2 * d_{k-2}
    donde a_k = diag[k], b_k = off[k].
    """
```

Si la implementación exacta resulta compleja, usar la aproximación:

```python
# Aproximación O(n) suficiente para GCV:
# df ≈ sum_k 1/(1 + lam * eigenvalue_approx_k)
# Para D'D tridiagonal, eigenvalues ≈ 4*sin^2(k*pi/(2n)), k=1,...,n-1
eigenvalues = 4 * np.sin(np.pi * np.arange(1, n) / (2*n))**2
df = 1 + np.sum(1 / (1 + lam * eigenvalues))   # +1 por el modo cero
```

**Usar la aproximación si la exacta es más de 20 líneas de código.**

---

## 7. Actualizar `__init__.py`

Agregar al final de `/home/rodo/regressor/__init__.py`:

```python
from .tikhonov_identification import (
    tikhonov_smooth,
    select_lambda,
    tii_reconstruct_u,
    tii_smooth_y,
)
```

---

## 8. Restricciones

- NO importar `motor_model`, `methods`, `ekf` — este módulo es independiente.
- NO usar `np.linalg.solve` ni `np.linalg.inv` para el sistema tridiagonal.
  Usar `scipy.linalg.solve_banded` con formato `(1,1)` para order=1
  y `(2,2)` para order=2.
- `lam=0` debe devolver `x = b.copy()` sin llamar al solver.
- NaN en `b` (posiciones de inicialización del regresor) → reemplazar por 0
  antes de resolver, restaurar NaN en la salida para las mismas posiciones.
- Para multivariable, aplicar `tikhonov_smooth` independientemente a cada
  componente. NO construir el sistema conjunto de tamaño N*n.
- `select_lambda` con `method='gcv'` es el default y el más importante.
  `method='lcurve'` y `method='grid'` son opcionales — si no están
  implementados, lanzar `NotImplementedError` con mensaje claro.

---

## 9. Relación con el código existente

Este módulo **generaliza** `tii()` de `10Paper/src/methods.py`:

| `methods.py`                         | `tikhonov_identification.py`              |
|--------------------------------------|-------------------------------------------|
| `Phi(i)` hardcoded para motor DC     | `point_estimator` es un closure genérico  |
| `inverse_integral` acoplada al motor | cualquier estimador HAM directo o inverso |
| `lam` fijo por el llamador           | `lam='auto'` con GCV                     |
| solo scalar                          | scalar y multivariable                    |
| solo Caso A (ruido en y)             | Caso A y Caso B                           |
| `solve_banded` directo               | encapsulado en `tikhonov_smooth`          |

Para reproducir exactamente `tii()` del motor DC con este módulo:

```python
from tikhonov_identification import tii_reconstruct_u

# point_estimator wrappea inverse_integral del 10Paper
def estimator(y_meas):
    omega, i_arr = y_meas
    return inverse_integral_motor(omega, i_arr, T)   # del 10Paper

u_reg, info = tii_reconstruct_u(
    y_meas=[omega_noisy, i_noisy],
    point_estimator=estimator,
    T=T,
    lam=100.0    # λ óptimo del paper
)
```

---

## 10. Criterio de aceptación

```bash
cd /home/rodo/regressor
python tikhonov_identification.py
```

Debe mostrar:

```
Test T1 — Caso A scalar (ruido en y, reconstruir u):
  RMSE sin regularizar : X.XXe-XX
  RMSE regularizado    : X.XXe-XX   ← menor que sin regularizar
  λ seleccionado (GCV) : X.XXe+XX
  ✓ PASS

Test T2 — Caso B scalar (ruido en u, suavizar y):
  RMSE sin regularizar : X.XXe-XX
  RMSE regularizado    : X.XXe-XX   ← menor que sin regularizar
  λ seleccionado (GCV) : X.XXe+XX
  ✓ PASS

Test T3 — Caso A multivariable (Lotka-Volterra):
  RMSE u1 : X.XXe-XX
  RMSE u2 : X.XXe-XX
  ✓ PASS

Test T4 — Comparación selección de λ:
  [tabla]
  ✓ GCV seleccionó el mejor λ (o dentro de factor 2 del óptimo)
```

---

*Fin del contrato — Rodolfo H. Rodrigo / UNSJ / Marzo 2026*
