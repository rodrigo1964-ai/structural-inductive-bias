# CLAUDE.md — Contrato: `verify_regressor_vs_rk45`
# Verificación unificada del regresor homotópico contra RK45
#
# Archivo destino: /home/rodo/regressor/verify_regressor.py
# Autor: Rodolfo H. Rodrigo — UNSJ
# Fecha: Marzo 2026
# =============================================================================

## 1. Objetivo

Implementar la función `verify_regressor_vs_rk45` que ejecuta el regresor
homotópico (scalar o multivariable) sobre un sistema de ODEs definido por el
usuario y lo compara contra `scipy.integrate.solve_ivp` con método RK45.

La función es el punto de entrada único para verificación.
Soporta los dos modos según los módulos del paquete:

  Modo A — Scalar:    usa `solver.py` / `regressor.py`   (N = 1, cualquier orden)
  Modo B — Sistema:   usa `solver_system.py` / `regressor_system.py`  (N ≥ 2)

---

## 2. Interfaz pública

### 2.1 Función principal

```python
def verify_regressor_vs_rk45(
    rhs,
    ic,
    t_span,
    n,
    regressor_callable,
    var_names=None,
    threshold=None,
    label="",
    plot=False,
    plot_path=None,
):
    """
    Compara el regresor homotópico contra RK45 en un sistema de ODEs.

    Parameters
    ----------
    rhs : callable
        Lado derecho del sistema en forma estándar para solve_ivp:
            rhs(t, y) -> array_like de longitud N
        donde y = [y1, y2, ..., yN] es el vector de estado.
        Para N=1: rhs(t, y) -> array de longitud 1, NO un escalar.

    ic : array_like, longitud N
        Condiciones iniciales [y1_0, y2_0, ..., yN_0].

    t_span : tuple (t0, tf)
        Intervalo de integración.

    n : int
        Número de puntos de muestreo (incluyendo t0).
        T = (tf - t0) / (n - 1)

    regressor_callable : callable
        Función que ejecuta el regresor. Firma según modo:

        Modo A (scalar):
            reg(u_array, y0, y1, T, n) -> np.ndarray shape (n,)

        Modo B (sistema):
            reg(excitations, initial_conditions, T, n) -> list of np.ndarray

        El llamador es responsable de construir reg con los builders
        correctos (build_regressor_order1, build_system_regressor, etc.)
        ANTES de llamar a verify_regressor_vs_rk45.

    var_names : list of str or None
        Nombres de las variables, ej: ["x", "y"] o ["theta1", "theta2"].
        Si None: ["y0", "y1", ...] o ["y"] para N=1.

    threshold : float or None
        Si no es None, la función lanza AssertionError si
        max_error[i] >= threshold para cualquier i.
        Usar None para solo reportar sin afirmar.

    label : str
        Etiqueta del test para el reporte impreso.

    plot : bool
        Si True, genera figura con subplots (solución + error por variable).
        Requiere matplotlib. Si matplotlib no está disponible, ignora silenciosamente.

    plot_path : str or None
        Ruta donde guardar la figura. Si None y plot=True, muestra en pantalla.

    Returns
    -------
    result : dict con las siguientes claves:
        "label"        : str,  etiqueta del test
        "t"            : np.ndarray shape (n,),  vector de tiempos
        "rk45"         : np.ndarray shape (N, n),  solución RK45
        "ham"          : np.ndarray shape (N, n),  solución regresor
        "error_abs"    : np.ndarray shape (N, n),  |ham - rk45| por variable
        "max_error"    : np.ndarray shape (N,),    max del error absoluto por variable
        "rms_error"    : np.ndarray shape (N,),    RMS del error por variable
        "passed"       : bool,  True si max_error[i] < threshold para todo i
                                (True si threshold=None)
        "T"            : float, período de muestreo efectivo
        "n"            : int,   número de puntos
        "var_names"    : list of str
    """
```

### 2.2 Función de reporte

```python
def print_report(result):
    """
    Imprime tabla de resultados formateada a stdout.

    Formato:
    ========================================
    TEST: <label>
    T = <T>s,  n = <n>,  t ∈ [t0, tf]
    ----------------------------------------
    Variable   max|error|    RMS error   Status
    --------   ----------    ---------   ------
    x          1.23e-04      4.56e-05    ✓ PASS
    y          7.89e-03      2.34e-03    ✗ FAIL   (threshold: 1e-2)
    ----------------------------------------
    OVERALL: PASS / FAIL
    ========================================
    """
```

### 2.3 Función batch

```python
def run_suite(test_list, stop_on_failure=False):
    """
    Ejecuta una lista de tests y devuelve resumen.

    Parameters
    ----------
    test_list : list of dict, cada dict con claves:
        "label"               : str
        "rhs"                 : callable
        "ic"                  : array_like
        "t_span"              : tuple
        "n"                   : int
        "regressor_callable"  : callable
        "var_names"           : list of str (opcional)
        "threshold"           : float (opcional)
        "plot"                : bool (opcional, default False)
        "plot_path"           : str  (opcional)

    stop_on_failure : bool
        Si True, detiene la suite al primer test que falla.

    Returns
    -------
    summary : dict con claves:
        "results"   : list de result dicts (uno por test)
        "passed"    : int, cantidad de tests que pasaron
        "failed"    : int, cantidad de tests que fallaron
        "all_passed": bool
    """
```

---

## 3. Convenciones internas

### 3.1 Construcción del vector RK45

```python
t_arr = np.linspace(t_span[0], t_span[1], n)
T = t_arr[1] - t_arr[0]

sol = solve_ivp(
    rhs,
    t_span,
    ic,
    method='RK45',
    t_eval=t_arr,
    rtol=1e-9,
    atol=1e-9,
)
# sol.y tiene shape (N, n)
```

### 3.2 Construcción de condiciones iniciales para el regresor

Las primeras dos muestras del regresor deben coincidir exactamente con RK45:

```python
# Modo A (scalar, N=1):
y0 = sol.y[0, 0]
y1 = sol.y[0, 1]
# llamar: regressor_callable(u_array, y0, y1, T, n)

# Modo B (sistema, N≥2):
initial_conditions = [[sol.y[i, 0], sol.y[i, 1]] for i in range(N)]
# llamar: regressor_callable(excitations, initial_conditions, T, n)
```

`verify_regressor_vs_rk45` NO construye las excitaciones `u_array` ni
`excitations` — eso ya está encapsulado en `regressor_callable` por el
llamador (usando closures). Ver Sección 5 para ejemplos.

### 3.3 Normalización de la salida del regresor

```python
# Modo A devuelve np.ndarray shape (n,)
ham_array = regressor_callable(...)           # shape (n,)
ham = ham_array.reshape(1, n)                 # shape (1, n)

# Modo B devuelve list of np.ndarray
ham_list = regressor_callable(...)            # list de N arrays shape (n,)
ham = np.array(ham_list)                      # shape (N, n)
```

La función detecta el modo por el tipo de retorno: si es `list`, es Modo B.

### 3.4 Cálculo de errores

```python
error_abs = np.abs(ham - sol.y)              # shape (N, n)
max_error = np.max(error_abs, axis=1)        # shape (N,)
rms_error = np.sqrt(np.mean(error_abs**2, axis=1))  # shape (N,)
```

### 3.5 Evaluación del threshold

```python
if threshold is not None:
    passed = bool(np.all(max_error < threshold))
else:
    passed = True
```

---

## 4. Estructura del archivo

El archivo `verify_regressor.py` contiene:

```
verify_regressor.py
├── imports
├── verify_regressor_vs_rk45(...)     # función principal
├── print_report(result)              # impresión formateada
├── run_suite(test_list, ...)         # ejecución batch
├── _detect_N(ic)                     # auxiliar: detecta N
├── _maybe_plot(result, plot_path)    # auxiliar: genera figura
└── if __name__ == "__main__":        # suite de demostración (Sección 5)
```

NO importar nada de `examples.py` ni `test_regressor_vs_rk4.py`.
NO duplicar los tests existentes — la suite de demostración usa sistemas nuevos.

---

## 5. Suite de demostración (`__main__`)

La suite de demostración en `if __name__ == "__main__":` debe ejecutar
exactamente los siguientes 6 tests usando `run_suite`.

### Test D1 — Scalar 1er orden: Van der Pol (μ=0.5)

```
y' + μ(y²-1)*y - y = A·sin(ωt)    →    y' = -μ(y²-1)*y + y + A·sin(ωt)
μ=0.5, A=0.5, ω=1.5
IC: y(0) = 0.5
t ∈ [0, 20],  n = 2000
threshold = 5e-2
```

Construir con `build_regressor_order1` (SymPy) o derivadas manuales.
La `regressor_callable` es un closure que usa `u = A*sin(ω*t_arr)`.

### Test D2 — Scalar 2do orden: Duffing forzado

```
y'' + 0.1*y' + y + 0.2*y³ = 0.3·cos(1.2t)
IC: y(0)=0.1, y'(0)=0
t ∈ [0, 50],  n = 5000
threshold = 1e-2
```

### Test D3 — Sistema 2D 1er orden: Lotka-Volterra

```
x' = α*x - β*x*y           α=1.0, β=0.1
y' = δ*x*y - γ*y            γ=1.5, δ=0.075
IC: x(0)=10, y(0)=5
t ∈ [0, 30],  n = 5000
threshold = 1e-2
var_names = ["x (prey)", "y (pred)"]
```

### Test D4 — Sistema 3D 1er orden: Lorenz

```
x' = σ(y-x)                 σ=10, ρ=28, β=8/3
y' = x(ρ-z) - y
z' = xy - βz
IC: (1, 1, 1)
t ∈ [0, 2],   n = 20000      ← solo t≤2 donde RK45 es comparable
threshold = 1e-1
var_names = ["x", "y", "z"]
```

### Test D5 — Sistema 2D 2do orden: Duffing acoplado

```
x'' + 0.1*x' + x + 0.2*x³ + 0.5*(x-y) = 0.5·cos(1.2t)
y'' + 0.1*y' + y + 0.2*y³ - 0.5*(x-y) = 0
IC: x(0)=0, x'(0)=0.5, y(0)=0.2, y'(0)=0
t ∈ [0, 50],  n = 50000
threshold = 1e-2
```

### Test D6 — Sistema 3D 1er orden: Euler cuerpo rígido

```
I1*w1' = (I2-I3)*w2*w3      I1=2, I2=1, I3=0.5
I2*w2' = (I3-I1)*w3*w1
I3*w3' = (I1-I2)*w1*w2
IC: w(0)=(1.0, 0.1, 0.5)
t ∈ [0, 20],  n = 2000
threshold = 1e-3
var_names = ["w1", "w2", "w3"]
```

Para D6, agregar verificación de conservación en `__main__` (no en la función
principal):
```python
I1, I2, I3 = 2.0, 1.0, 0.5
E0 = 0.5*(I1*w1[0]**2 + I2*w2[0]**2 + I3*w3[0]**2)
E  = 0.5*(I1*w1**2    + I2*w2**2    + I3*w3**2)
E_drift = np.max(np.abs(E - E0)) / E0
print(f"  Energía: deriva máx = {E_drift:.2e}  {'✓' if E_drift < 1e-4 else '✗'}")
```

---

## 6. Construcción de los regressors para la suite

Cada `regressor_callable` de la suite se construye **fuera** de
`verify_regressor_vs_rk45` y se pasa como closure.

Ejemplo para D3 (Lotka-Volterra) con `build_system_regressor`:

```python
from sympy import symbols, Symbol
from regressor_system import build_system_regressor

x, y, z    = symbols('x y z')
xp, yp, zp = symbols('xp yp zp')
xpp, ypp, zpp = symbols('xpp ypp zpp')
t = Symbol('t')

a, b, g, d = 1.0, 0.1, 1.5, 0.075
F = xp - a*x + b*x*y
G = yp - d*x*y + g*y

state_syms = [x, y, z, xp, yp, zp, xpp, ypp, zpp, t]
reg_lv, _ = build_system_regressor([F, G], state_syms, order=1)

n_lv = 5000
t_lv = np.linspace(0, 30, n_lv)
T_lv = t_lv[1] - t_lv[0]

def lv_regressor_callable(initial_conditions):
    """Closure: ya sabe T, n, excitaciones."""
    u = np.zeros(n_lv)
    v = np.zeros(n_lv)
    return reg_lv([u, v], initial_conditions, T_lv, n_lv)

# Pero verify_regressor_vs_rk45 necesita IC desde RK45.
# Entonces la closure recibe IC como argumento en runtime:
# regressor_callable = lambda ic: lv_regressor_callable(
#     [[ic[0], sol_lv.y[0,1]], [ic[1], sol_lv.y[1,1]]]
# )
```

**Patrón recomendado para todos los tests del sistema:**
La `regressor_callable` recibe el resultado de `solve_ivp` como referencia
para construir las condiciones iniciales de dos puntos. El llamador lo hace
en `__main__` antes de armar el `test_list`.

Alternativamente, `verify_regressor_vs_rk45` puede aceptar una firma
extendida para la callable:

```python
# Firma alternativa que recibe sol directamente:
regressor_callable(sol_rk45)  ->  ham_array or ham_list
# donde sol_rk45 = resultado de solve_ivp (tiene .y, .t)
```

**Usar esta firma alternativa** (más limpia). El contrato la adopta:

```python
# FIRMA OFICIAL de regressor_callable:
regressor_callable(sol) -> np.ndarray shape (n,)    [Modo A]
                        -> list of np.ndarray         [Modo B]
# donde sol = resultado de solve_ivp con .y shape (N, n) y .t shape (n,)
```

Esto permite que el closure acceda a `sol.y[i, 1]` para la segunda
condición inicial del regresor.

---

## 7. Generación de figura (si plot=True)

Layout: `N` filas × 2 columnas.
- Columna izquierda: trayectorias RK45 (azul sólido) y HAM (rojo punteado).
- Columna derecha: error absoluto en escala logarítmica.
- Cada subplot con título `var_names[i]`, eje x = "Tiempo [s]".
- Título general: `f"HAM vs RK45 — {label}"`.

Si `plot_path` es None y `plot=True`, usar `plt.show()`.
Si matplotlib no está instalado, imprimir advertencia y continuar sin error.

---

## 8. Actualizar `__init__.py`

Agregar al final de `/home/rodo/regressor/__init__.py`:

```python
from .verify_regressor import verify_regressor_vs_rk45, run_suite, print_report
```

---

## 9. Restricciones

- NO modificar `solver.py`, `regressor.py`, `examples.py`,
  `test_regressor_vs_rk4.py` ni ningún archivo existente.
- NO usar `odeint` — solo `solve_ivp` con `method='RK45'`, `rtol=1e-9`, `atol=1e-9`.
- Para N=1 la función acepta tanto `ic = [y0]` (lista longitud 1) como `ic = y0`
  (escalar) — normalizar internamente con `ic = np.atleast_1d(ic)`.
- NO asumir que `regressor_callable` conoce T o n — el closure los captura.
- `threshold=None` nunca lanza AssertionError, solo reporta.
- Si `solve_ivp` no converge (`sol.success == False`), lanzar
  `RuntimeError("RK45 no convergió: " + sol.message)`.

---

## 10. Criterio de aceptación

```bash
cd /home/rodo/regressor
python verify_regressor.py
```

Debe mostrar:

```
========================================
TEST: D1 — Van der Pol scalar
...  ✓ PASS
TEST: D2 — Duffing scalar 2do orden
...  ✓ PASS
TEST: D3 — Lotka-Volterra 2D
...  ✓ PASS
TEST: D4 — Lorenz 3D
...  ✓ PASS
TEST: D5 — Duffing acoplado 2D 2do orden
...  ✓ PASS
TEST: D6 — Euler cuerpo rígido 3D
...  ✓ PASS
  Energía: deriva máx = X.XXe-XX  ✓
========================================
Suite: 6/6 passed
```

Si algún test falla, reducir T (aumentar n) antes de diagnosticar fallo del método.

---

*Fin del contrato — Rodolfo H. Rodrigo / UNSJ / Marzo 2026*
