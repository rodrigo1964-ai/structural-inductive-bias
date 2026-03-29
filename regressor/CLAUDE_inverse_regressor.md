# CLAUDE.md — Contrato: `build_inverse_regressor`

**Archivo destino:** `regressor/regressor.py` — agregar al final, sin modificar nada existente  
**Autor:** Rodolfo H. Rodrigo — UNSJ  
**Fecha:** Marzo 2026

---

## 1. Motivación (Teorema 1 — Paper 10)

El regresor directo resuelve para `y[k]` dado `u[k]`:

```
F(y[k], y'[k], y''[k], u[k]) = 0   →   g(y[k]) = 0   →   z1/z2/z3 en y[k]
```

El regresor inverso resuelve para `u[k]` dado `y[k]` (medido):

```
F(y[k], y'[k], y''[k], u[k]) = 0   →   g(u[k]) = 0   →   z1/z2/z3 en u[k]
```

**La única diferencia es el símbolo respecto al cual se derivan g', g'', g'''.**
`y'[k]` y `y''[k]` son constantes en el loop inverso (calculadas del array `y` conocido).
`u` no aparece en las fórmulas discretas de derivadas → no hay regla de la cadena para `u`.

---

## 2. Forma de entrada obligatoria

`F_expr` debe estar en **forma residuo**:

```
F(y, yp, ypp, u, t) = 0
```

Ejemplos válidos:
```python
# Motor DC eléctrico:  L(i)*di/dt + R*i + Ke*w = u
# y=i, u=u_voltaje, yp=i', los otros símbolos son parámetros numéricos
F = L_sym * yp + R_val * y + Ke_val * w_sym - u

# Sistema genérico no lineal:
F = yp + a*y**3 + b*sin(y) - u**2 - c*u
```

Restricción: `u_sym` debe aparecer en `F_expr` con `∂F/∂u ≠ 0` en el punto de expansión.
Si `u` es lineal en `F`, z2 y z3 son cero — el contrato los calcula igual y SymPy los simplifica a 0.

---

## 3. Interfaz pública a implementar

```python
def build_inverse_regressor(F_expr, all_syms, u_sym):
    """
    Genera el regresor homotópico inverso: resuelve u[k] dado y[k].

    Dado F(y, yp, ypp, u, t) = 0 con y[k] conocido, encuentra u[k] tal
    que el residuo sea cero, usando la serie homotópica de 3 términos
    con derivadas respecto a u_sym.

    Parameters
    ----------
    F_expr : sympy.Expr
        Residuo completo F(y, yp, ypp, u, t) en forma F = 0.
        Puede depender de parámetros adicionales siempre que estén
        incluidos en all_syms o sean numéricos literales.

    all_syms : tuple of sympy.Symbol
        Todos los símbolos que aparecen en F_expr, en el orden en que
        se pasarán a lambdify. Obligatorio incluir: y_sym, yp_sym,
        ypp_sym (o dummy), u_sym, t_sym. Parámetros extras van al final.
        Ejemplo: (y, yp, ypp, u, t)
        Ejemplo con parámetros: (y, yp, ypp, u, t, w, R, Ke, L0, Isat)

    u_sym : sympy.Symbol
        Símbolo de la variable de entrada a resolver. Debe estar en all_syms.

    Returns
    -------
    inverse_regressor : callable
        Firma: inverse_regressor(y, u0, u1, T, n, *params) -> np.ndarray
        Ver Sección 5.
    info : dict
        Ver Sección 6.
    """
```

---

## 4. Derivadas simbólicas a calcular

Derivar `F_expr` respecto a `u_sym` únicamente (sin regla de la cadena discreta):

```python
dF_u   = diff(F_expr, u_sym)          # dF/du
d2F_u2 = diff(dF_u,   u_sym)          # d²F/du²
d3F_u3 = diff(d2F_u2, u_sym)          # d³F/du³
```

Simplificar con `simplify()` antes de `lambdify`.

Imprimir al construir:
```
F(...)    = <expr>
dF/du     = <expr>
d²F/du²   = <expr>
d³F/du³   = <expr>
```

Si `d2F_u2 == 0` simbólicamente → imprimir `"u es lineal en F: z2 = z3 = 0 (se calculan igual)"`.

---

## 5. Loop del regresor inverso

### Firma de la función devuelta

```python
def inverse_regressor(y, u0, u1, T, n, *params):
    """
    Parameters
    ----------
    y      : np.ndarray, shape (n,)
        Trayectoria de salida conocida (medida).
    u0     : float   u en t_0 (valor inicial, no se recomputa)
    u1     : float   u en t_1 (valor inicial, no se recomputa)
    T      : float   período de muestreo
    n      : int     número total de puntos
    *params: floats  parámetros adicionales que aparecen en F_expr
                     (ej: w_array si w es señal externa, o R, Ke, L0...)
                     En el mismo orden que en all_syms después de t_sym.

    Returns
    -------
    u : np.ndarray, shape (n,)
    """
```

### Estructura del loop

```python
u = np.zeros(n)
u[0] = u0
u[1] = u1

for k in range(2, n):
    # Paso 1: calcular derivadas discretas de y (conocidas, constantes en este paso)
    yp_k  = (3*y[k] - 4*y[k-1] + y[k-2]) / (2*T)
    ypp_k = (y[k] - 2*y[k-1] + y[k-2]) / T**2
    tk    = k * T

    # Construir args para lambdify (en el orden de all_syms):
    # args = (y[k], yp_k, ypp_k, u_current, tk, *params_k)
    # donde params_k son los valores numéricos de parámetros adicionales
    # en el paso k (escalares o y_extra[k] si son arrays)

    # Estimación inicial: u_current = u[k-1]
    u_curr = u[k-1]

    # --- z1: Newton ---
    args = _build_args(y[k], yp_k, ypp_k, u_curr, tk, params, k)
    g  = F_num(*args)
    gp = dF_u_num(*args)
    u_curr = u_curr - g / gp

    # --- z2: curvatura ---
    args = _build_args(y[k], yp_k, ypp_k, u_curr, tk, params, k)
    g   = F_num(*args)
    gp  = dF_u_num(*args)
    gpp = d2F_u2_num(*args)
    u_curr = u_curr - (1/2) * g**2 * gpp / gp**3

    # --- z3: tercer orden ---
    args = _build_args(y[k], yp_k, ypp_k, u_curr, tk, params, k)
    g    = F_num(*args)
    gp   = dF_u_num(*args)
    gpp  = d2F_u2_num(*args)
    gppp = d3F_u3_num(*args)
    u_curr = u_curr - (1/6) * g**3 * (-gppp * gp + 3 * gpp**2) / gp**5

    u[k] = u_curr

return u
```

**Nota sobre `_build_args`:** es una función interna auxiliar (no pública) que
ensambla la tupla de argumentos en el orden exacto de `all_syms`, sustituyendo
`u_sym` por `u_curr` y manejando parámetros escalares o arrays.

---

## 6. `info` dict a devolver

```python
info = {
    'F':        F_expr,          # expresión original
    'dF_u':     dF_u,            # sympy: dF/du
    'd2F_u2':   d2F_u2,          # sympy: d²F/du²
    'd3F_u3':   d3F_u3,          # sympy: d³F/du³
    'F_num':    F_num,           # callable lambdify
    'dF_u_num': dF_u_num,        # callable lambdify
    'd2F_u2_num': d2F_u2_num,    # callable lambdify
    'd3F_u3_num': d3F_u3_num,    # callable lambdify
    'u_sym':    u_sym,
    'all_syms': all_syms,
    'u_is_linear': bool(d2F_u2 == 0),   # True si u lineal en F
}
```

---

## 7. Cuatro casos de validación

Implementar en el bloque `if __name__ == "__main__":` de `regressor.py`
(agregar al final de los tests existentes), **o** en un archivo separado
`test_inverse_regressor.py` en `/home/rodo/regressor/`.

### Test 1 — u lineal (motor DC simplificado, sin saturación)

Sistema:   `L0 * yp + R * y + Ke * w - u = 0`
con `L0=0.01, R=1.0, Ke=0.1`, `w(t)=10*sin(t)`, `T=1e-4`, `n=2000`.

Ground truth: generar `y_true[k]` con `solve_ivp` dado `u_true(t) = 12*(t>0.005)`.
Pasar `y_true` al inverso y verificar que `u_hat ≈ u_true`.

Criterio: `max|u_hat[10:] - u_true[10:]| < 0.1` (excluir transiente inicial).

```python
from sympy import symbols, Symbol
y, yp, ypp, u, t_s, w_s = symbols('y yp ypp u t w')
L0_val, R_val, Ke_val = 0.01, 1.0, 0.1

F = L0_val * yp + R_val * y + Ke_val * w_s - u
all_syms = (y, yp, ypp, u, t_s, w_s)

inv_reg, info = build_inverse_regressor(F, all_syms, u)
# info['u_is_linear'] debe ser True
```

### Test 2 — u no lineal: `F = yp + a*y - u**2 = 0`

Sistema: `y' = u² - a*y` con `a=0.5`.
Ground truth: `solve_ivp` con `u_true(t) = 1 + 0.5*sin(2*t)`.

Criterio: `max|u_hat[5:] - u_true[5:]| < 1e-2`.

```python
F2 = yp + 0.5*y - u**2
all_syms2 = (y, yp, ypp, u, t_s)
inv_reg2, info2 = build_inverse_regressor(F2, all_syms2, u)
# info2['u_is_linear'] debe ser False
```

### Test 3 — Verificar simetría directa/inversa

Usando `build_regressor_order1` con `f = 0.5*y`:

```
F = yp + 0.5*y - u = 0
```

1. Generar `y_ref` con el regresor directo dado `u_known`.
2. Reconstruir `u_hat` con el regresor inverso dado `y_ref`.
3. Verificar: `max|u_hat[5:] - u_known[5:]| < 1e-4`.

Este test verifica la simetría del Teorema 1: mismo F, misma precisión en ambas direcciones.

### Test 4 — u lineal: verificar que z2 y z3 son exactamente cero

Para `F = yp + y - u`:
```python
assert info['u_is_linear'] == True
assert info['d2F_u2'] == 0      # sympy S.Zero
assert info['d3F_u3'] == 0
```

---

## 8. Actualizar `__init__.py`

Agregar al final de `/home/rodo/regressor/__init__.py`:

```python
from .regressor import build_inverse_regressor
```

---

## 9. Restricciones

- NO modificar ninguna función existente en `regressor.py` ni `solver.py`.
- NO usar `np.linalg.inv` ni `np.linalg.solve` — el inverso es escalar, división directa.
- NO asumir que `ypp_sym` aparece en `F_expr`; puede ser cero para sistemas de 1er orden.
  Manejar con un símbolo `ypp` dummy que lambdify evalúa pero que puede no estar en `F_expr`.
- Los parámetros adicionales en `*params` pueden ser:
  - **escalares** (R, Ke, L0): pasar como literales numéricos en la llamada.
  - **arrays** (w(t), señal externa): en ese caso `params_k = (w[k],)` dentro del loop.
  El contrato no distingue — el llamador es responsable de la indexación.
- División por cero: si `gp == 0` en z1, lanzar `RuntimeError` con mensaje claro indicando
  `k`, `u_curr` y `gp`. No silenciar con `np.errstate`.

---

## 10. Criterio de aceptación final

Ejecutar:
```bash
cd /home/rodo/regressor
python test_inverse_regressor.py
```

Debe mostrar:
```
Test 1 - Motor lineal:    ✓ PASS  max_err = X.XXe-0X
Test 2 - u cuadrático:    ✓ PASS  max_err = X.XXe-0X
Test 3 - Simetría T1:     ✓ PASS  max_err = X.XXe-0X
Test 4 - z2=z3=0 lineal:  ✓ PASS
```

Si algún test falla, reducir `T` a la mitad antes de diagnosticar fallo del método.

---

*Fin del contrato — Rodolfo H. Rodrigo / UNSJ / Marzo 2026*
