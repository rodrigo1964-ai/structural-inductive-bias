"""
convergence.py — Control de convergencia para el HAM continuo

Implementa:
1. Curva-hbar: evalua u'(0) vs hbar para detectar plateau de convergencia
2. hbar optimo: minimiza residuo cuadrado integral
3. Region de convergencia: intervalo [hbar_min, hbar_max]

Referencia:
    Liao, S.J. "An optimal homotopy-analysis approach for strongly
    nonlinear differential equations", CNSNS, 15:2003-2016 (2010).

Author: Rodolfo H. Rodrigo — UNSJ / INAUT
"""

import numpy as np
from sympy import diff, lambdify, simplify, expand, S, symbols, integrate


def hbar_curve(N_expr, y_sym, yp_sym, t_sym, ic,
               hbar_range=(-2.0, 0.5), n_points=50, M=10,
               eval_point=None, ypp_sym=None, ic_prime=None):
    """
    Calcula la curva-hbar: evalua una cantidad derivada de la solucion
    como funcion de hbar, para detectar la region de convergencia.

    La region de convergencia es el intervalo de hbar donde la curva
    forma un "plateau" horizontal.

    Parameters
    ----------
    N_expr : sympy.Expr
        Operador no lineal N[u] = 0.
    y_sym, yp_sym : Symbol
        Simbolos de y, y'.
    t_sym : Symbol
        Variable independiente.
    ic : float
        Condicion inicial u(0).
    hbar_range : tuple (hbar_min, hbar_max)
        Rango de hbar a explorar.
    n_points : int
        Numero de puntos en el rango.
    M : int
        Numero de terminos HAM.
    eval_point : float or None
        Punto t donde evaluar u(t; hbar). Si None, usa u'(0).
    ypp_sym : Symbol or None
        Para EDOs de 2do orden.
    ic_prime : float or None
        y'(0) para 2do orden.

    Returns
    -------
    result : dict
        'hbar_values' : np.ndarray
        'curve_values' : np.ndarray (u(eval_point; hbar) o u'(0; hbar))
        'plateau_range' : tuple (hbar_lo, hbar_hi) estimado
    """
    from .ham_series import ham_solve

    hbar_values = np.linspace(hbar_range[0], hbar_range[1], n_points)
    curve_values = np.zeros(n_points)

    order = 2 if ypp_sym is not None else 1

    print(f"Calculando curva-hbar: {n_points} puntos, M={M}")

    for i, h in enumerate(hbar_values):
        try:
            # Resolver HAM con este hbar (silencioso)
            import io, sys
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()

            res = ham_solve(N_expr, y_sym, yp_sym, t_sym, ic,
                            hbar=float(h), M=M,
                            ypp_sym=ypp_sym, ic_prime=ic_prime,
                            simplify_terms=False)

            sys.stdout = old_stdout

            series_expr = res['series']

            if eval_point is not None:
                # Evaluar u(eval_point; hbar)
                f = lambdify(t_sym, series_expr, modules='numpy')
                curve_values[i] = float(f(eval_point))
            else:
                # Evaluar u'(0; hbar)
                u_prime = diff(series_expr, t_sym)
                f = lambdify(t_sym, u_prime, modules='numpy')
                curve_values[i] = float(f(0.0))

        except Exception:
            curve_values[i] = np.nan

    # Estimar plateau: region donde la derivada de la curva es minima
    plateau_range = _estimate_plateau(hbar_values, curve_values)

    print(f"  Plateau estimado: hbar in [{plateau_range[0]:.3f}, {plateau_range[1]:.3f}]")

    return {
        'hbar_values': hbar_values,
        'curve_values': curve_values,
        'plateau_range': plateau_range,
    }


def optimal_hbar(N_expr, y_sym, yp_sym, t_sym, ic,
                 hbar_range=(-2.0, -0.1), M=10, n_search=20,
                 t_domain=(0, 1), n_quad=50,
                 ypp_sym=None, ic_prime=None):
    """
    Encuentra el hbar optimo minimizando el residuo cuadrado:

        E(hbar) = integral_0^T [ N[u_approx(t; hbar)] ]^2 dt

    Usa busqueda sobre grilla + refinamiento.

    Parameters
    ----------
    N_expr : sympy.Expr
    y_sym, yp_sym, t_sym : Symbol
    ic : float
    hbar_range : tuple
    M : int
    n_search : int
        Puntos en la grilla de busqueda.
    t_domain : tuple (t0, tf)
        Dominio para la integral del residuo.
    n_quad : int
        Puntos de cuadratura.
    ypp_sym : Symbol or None
    ic_prime : float or None

    Returns
    -------
    result : dict
        'hbar_optimal' : float
        'E_optimal'    : float (residuo cuadrado minimo)
        'hbar_grid'    : np.ndarray
        'E_grid'       : np.ndarray
    """
    from .ham_series import ham_solve, _eval_N_at

    order = 2 if ypp_sym is not None else 1
    hbar_grid = np.linspace(hbar_range[0], hbar_range[1], n_search)
    E_grid = np.zeros(n_search)
    t_quad = np.linspace(t_domain[0], t_domain[1], n_quad)

    print(f"Buscando hbar optimo: {n_search} candidatos, M={M}")

    for i, h in enumerate(hbar_grid):
        try:
            import io, sys
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()

            res = ham_solve(N_expr, y_sym, yp_sym, t_sym, ic,
                            hbar=float(h), M=M,
                            ypp_sym=ypp_sym, ic_prime=ic_prime,
                            simplify_terms=False)

            sys.stdout = old_stdout

            # Evaluar residuo N[u_approx] en puntos de cuadratura
            residual_expr = _eval_N_at(N_expr, y_sym, yp_sym, ypp_sym,
                                       t_sym, res['series'], order)
            residual_expr = expand(residual_expr)
            f_res = lambdify(t_sym, residual_expr, modules='numpy')

            res_vals = np.array([float(f_res(tv)) for tv in t_quad])
            E_grid[i] = np.trapz(res_vals**2, t_quad)

        except Exception:
            E_grid[i] = np.inf

    # Encontrar minimo
    idx_min = np.argmin(E_grid)
    hbar_opt = hbar_grid[idx_min]
    E_opt = E_grid[idx_min]

    print(f"  hbar optimo: {hbar_opt:.4f}")
    print(f"  E(hbar_opt): {E_opt:.4e}")

    return {
        'hbar_optimal': hbar_opt,
        'E_optimal': E_opt,
        'hbar_grid': hbar_grid,
        'E_grid': E_grid,
    }


def convergence_table(result, t_point=None, reference=None):
    """
    Genera tabla de convergencia: muestra como las sumas parciales
    S_m convergen al valor real.

    Parameters
    ----------
    result : dict
        Resultado de ham_solve().
    t_point : float or None
        Punto de evaluacion. Si None, usa t=1.
    reference : float or None
        Valor de referencia (solucion exacta). Si None, usa S_M.

    Returns
    -------
    table : list of dict
        [{'m': m, 'S_m': valor, 'error': error_relativo}]
    """
    from sympy import lambdify

    t_sym = result['t_sym']
    terms = result['terms']
    M = len(terms) - 1

    if t_point is None:
        t_point = 1.0

    table = []
    partial = S.Zero

    for m in range(M + 1):
        partial = partial + terms[m]
        f = lambdify(t_sym, partial, modules='numpy')
        val = float(f(t_point))

        entry = {'m': m, 'S_m': val}
        table.append(entry)

    # Calcular errores
    if reference is None:
        reference = table[-1]['S_m']

    for entry in table:
        if abs(reference) > 1e-15:
            entry['error'] = abs(entry['S_m'] - reference) / abs(reference)
        else:
            entry['error'] = abs(entry['S_m'] - reference)

    return table


def print_convergence_table(table, label=""):
    """Imprime tabla de convergencia formateada."""
    print(f"{'='*60}")
    if label:
        print(f"Convergencia: {label}")
    print(f"{'m':<5} {'S_m':<20} {'Error relativo':<15}")
    print(f"{'-'*60}")
    for entry in table:
        print(f"{entry['m']:<5} {entry['S_m']:<20.10f} {entry['error']:<15.4e}")
    print(f"{'='*60}\n")


# ======================================================================
# Funciones internas
# ======================================================================

def _estimate_plateau(hbar_values, curve_values):
    """
    Estima la region de plateau en la curva-hbar.

    Busca donde la derivada discreta |dcurve/dhbar| es minima.
    """
    # Filtrar NaN
    valid = ~np.isnan(curve_values) & ~np.isinf(curve_values)
    if np.sum(valid) < 5:
        return (hbar_values[0], hbar_values[-1])

    h_valid = hbar_values[valid]
    c_valid = curve_values[valid]

    # Derivada discreta
    dc = np.abs(np.diff(c_valid) / np.diff(h_valid))

    # Suavizar con ventana movil
    if len(dc) >= 5:
        kernel = np.ones(5) / 5
        dc_smooth = np.convolve(dc, kernel, mode='same')
    else:
        dc_smooth = dc

    # Encontrar region de minima variacion
    threshold = np.median(dc_smooth) * 0.5
    flat_mask = dc_smooth < threshold

    if np.any(flat_mask):
        flat_indices = np.where(flat_mask)[0]
        lo = h_valid[flat_indices[0]]
        hi = h_valid[min(flat_indices[-1] + 1, len(h_valid) - 1)]
        return (float(lo), float(hi))
    else:
        # Fallback: centro del rango
        mid = (hbar_values[0] + hbar_values[-1]) / 2
        return (float(mid - 0.3), float(mid + 0.3))


# ======================================================================
# Test
# ======================================================================

if __name__ == "__main__":
    from sympy import Symbol, symbols

    print("=" * 70)
    print("TEST: CONVERGENCIA HAM")
    print("=" * 70)

    y, yp, t = symbols('y yp t')

    # Test: y' + y^2 = 0, y(0) = 1 => y = 1/(1+t)
    print("\n--- Curva-hbar para y' + y² = 0 ---")
    N = yp + y**2

    hbar_result = hbar_curve(N, y, yp, t, ic=1.0,
                              hbar_range=(-2.5, 0.5), n_points=30, M=8,
                              eval_point=1.0)

    print(f"  Plateau: {hbar_result['plateau_range']}")

    # Test: hbar optimo
    print("\n--- hbar optimo para y' + y² = 0 ---")
    opt_result = optimal_hbar(N, y, yp, t, ic=1.0,
                               hbar_range=(-2.0, -0.1), M=8,
                               n_search=15, t_domain=(0, 2))

    print(f"  hbar optimo: {opt_result['hbar_optimal']:.4f}")

    print("\n" + "=" * 70)
