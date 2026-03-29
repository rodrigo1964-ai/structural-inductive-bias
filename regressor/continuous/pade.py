"""
pade.py — Aproximantes de Padé para acelerar convergencia de series HAM

Dado S_M(t) = sum(a_k * t^k, k=0..M), construye [m/n] con m+n <= M:

    [m/n](t) = P_m(t) / Q_n(t)

donde P_m, Q_n son polinomios que coinciden con S_M hasta O(t^{m+n+1}).

Los aproximantes de Padé extienden dramaticamente el radio de convergencia
de la serie de Taylor, lo que combinado con el HAM produce soluciones
validas en intervalos mucho mas largos.

Referencia:
    Baker, G.A. & Graves-Morris, P. "Padé Approximants", Cambridge, 1996.
    Liao, S.J. "Homotopy Analysis Method", Springer, 2012 (Cap. 3).

Author: Rodolfo H. Rodrigo — UNSJ / INAUT
"""

import numpy as np
from sympy import (
    Symbol, Poly, series as sym_series, Rational, S,
    lambdify, simplify, expand, cancel, collect, factor
)


def pade_approximant(series_expr, t_sym, m, n):
    """
    Construye el aproximante de Padé [m/n] a partir de una expresion simbolica.

    Parameters
    ----------
    series_expr : sympy.Expr
        Expresion simbolica en t_sym (normalmente la suma de la serie HAM).
    t_sym : Symbol
        Variable independiente.
    m : int
        Grado del numerador.
    n : int
        Grado del denominador.

    Returns
    -------
    result : dict
        'pade'       : sympy.Expr (P_m/Q_n simplificado)
        'numerator'  : sympy.Expr (P_m)
        'denominator': sympy.Expr (Q_n)
        'coeffs_P'   : list (coeficientes de P_m)
        'coeffs_Q'   : list (coeficientes de Q_n)
        'm'          : int
        'n'          : int
    """
    # Extraer coeficientes de Taylor de la serie
    coeffs = _extract_taylor_coeffs(series_expr, t_sym, m + n)

    if len(coeffs) < m + n + 1:
        raise ValueError(
            f"Serie insuficiente: necesito {m+n+1} coeficientes, "
            f"tengo {len(coeffs)}. Aumente M en ham_solve."
        )

    # Calcular coeficientes de Padé
    P_coeffs, Q_coeffs = _compute_pade_coefficients(coeffs, m, n)

    # Construir polinomios
    P_expr = sum(S(P_coeffs[k]) * t_sym**k for k in range(len(P_coeffs)))
    Q_expr = sum(S(Q_coeffs[k]) * t_sym**k for k in range(len(Q_coeffs)))

    # Simplificar
    pade_expr = cancel(P_expr / Q_expr)

    return {
        'pade': pade_expr,
        'numerator': P_expr,
        'denominator': Q_expr,
        'coeffs_P': P_coeffs,
        'coeffs_Q': Q_coeffs,
        'm': m,
        'n': n,
    }


def pade_from_ham(result, m=None, n=None):
    """
    Construye el aproximante de Padé directamente desde un resultado HAM.

    Parameters
    ----------
    result : dict
        Resultado de ham_solve().
    m : int or None
        Grado del numerador. Si None, usa M//2.
    n : int or None
        Grado del denominador. Si None, usa M//2.

    Returns
    -------
    pade_result : dict (ver pade_approximant)
    """
    M = result['M']
    t_sym = result['t_sym']
    series_expr = result['series']

    if m is None:
        m = M // 2
    if n is None:
        n = M // 2

    if m + n > M:
        print(f"  Advertencia: m+n={m+n} > M={M}, reduciendo a [{M//2}/{M//2}]")
        m = M // 2
        n = M // 2

    return pade_approximant(series_expr, t_sym, m, n)


def pade_eval(pade_result, t_eval):
    """
    Evalua un aproximante de Padé en puntos numericos.

    Parameters
    ----------
    pade_result : dict
        Resultado de pade_approximant().
    t_eval : np.ndarray
        Puntos de evaluacion.

    Returns
    -------
    values : np.ndarray
    """
    # Extraer t_sym del pade_result
    pade_expr = pade_result['pade']
    free = pade_expr.free_symbols
    if len(free) == 1:
        t_sym = free.pop()
    else:
        raise ValueError("Expresion Padé tiene mas de un simbolo libre")

    f = lambdify(t_sym, pade_expr, modules='numpy')
    return np.array([float(f(tv)) for tv in t_eval])


def pade_diagonal_sequence(series_expr, t_sym, max_order, t_eval=None):
    """
    Calcula la secuencia diagonal de Padé [1/1], [2/2], [3/3], ...
    que es la secuencia con convergencia mas rapida.

    Parameters
    ----------
    series_expr : sympy.Expr
    t_sym : Symbol
    max_order : int
        Orden maximo: calcula hasta [max_order/max_order].
    t_eval : np.ndarray or None

    Returns
    -------
    sequence : list of dict
        Cada dict tiene 'order', 'pade_expr', 'values' (si t_eval dado)
    """
    sequence = []

    for k in range(1, max_order + 1):
        try:
            res = pade_approximant(series_expr, t_sym, k, k)
            entry = {
                'order': k,
                'pade_expr': res['pade'],
            }

            if t_eval is not None:
                f = lambdify(t_sym, res['pade'], modules='numpy')
                entry['values'] = np.array([float(f(tv)) for tv in t_eval])

            sequence.append(entry)
        except Exception as e:
            print(f"  Padé [{k}/{k}] fallo: {e}")
            break

    return sequence


# ======================================================================
# Funciones internas
# ======================================================================

def _extract_taylor_coeffs(expr, t_sym, max_order):
    """
    Extrae coeficientes de Taylor a_0, a_1, ..., a_max_order
    de una expresion simbolica.
    """
    # Expandir como serie de Taylor alrededor de t=0
    taylor = sym_series(expr, t_sym, 0, max_order + 1)
    taylor = expand(taylor.removeO())

    coeffs = []
    for k in range(max_order + 1):
        try:
            p = Poly(taylor, t_sym)
            c = p.nth(k)
        except Exception:
            # Fallback: coeficiente directo
            c = taylor.coeff(t_sym, k)
        coeffs.append(float(c))

    return coeffs


def _compute_pade_coefficients(coeffs, m, n):
    """
    Calcula los coeficientes del aproximante de Padé [m/n].

    Dado c_0, c_1, ..., c_{m+n}, calcula P y Q tales que:
        P(t)/Q(t) = c_0 + c_1*t + ... + c_{m+n}*t^{m+n} + O(t^{m+n+1})

    con Q(0) = 1 (normalizacion).

    Metodo: resolver el sistema lineal para los coeficientes de Q,
    luego calcular P por multiplicacion.
    """
    c = coeffs  # c[k] = a_k

    # Si n == 0, el aproximante es simplemente el polinomio truncado
    if n == 0:
        P_coeffs = [c[k] for k in range(m + 1)]
        Q_coeffs = [1.0]
        return P_coeffs, Q_coeffs

    # Construir sistema lineal para q_1, ..., q_n
    # Ecuacion: sum_{j=1}^{n} q_j * c_{m+k-j} = -c_{m+k}  para k = 1, ..., n
    #
    # Es decir: A * q = -b  donde
    #   A[k-1, j-1] = c_{m+k-j}  (k=1..n, j=1..n)
    #   b[k-1] = c_{m+k}

    A = np.zeros((n, n))
    b = np.zeros(n)

    for k in range(1, n + 1):
        for j in range(1, n + 1):
            idx = m + k - j
            if 0 <= idx < len(c):
                A[k-1, j-1] = c[idx]
            else:
                A[k-1, j-1] = 0.0
        idx_b = m + k
        if idx_b < len(c):
            b[k-1] = c[idx_b]
        else:
            b[k-1] = 0.0

    # Resolver para q
    try:
        q = np.linalg.solve(A, -b)
    except np.linalg.LinAlgError:
        raise ValueError(
            f"Sistema singular al calcular Padé [{m}/{n}]. "
            f"Pruebe con m, n diferentes."
        )

    # Q(t) = 1 + q_1*t + q_2*t^2 + ... + q_n*t^n
    Q_coeffs = [1.0] + list(q)

    # P(t) = sum_{k=0}^{m} p_k * t^k
    # donde p_k = c_k + sum_{j=1}^{min(k,n)} q_j * c_{k-j}
    P_coeffs = []
    for k in range(m + 1):
        p_k = c[k]
        for j in range(1, min(k, n) + 1):
            p_k += q[j-1] * c[k-j]
        P_coeffs.append(p_k)

    return P_coeffs, Q_coeffs


# ======================================================================
# Test
# ======================================================================

if __name__ == "__main__":
    from sympy import symbols, exp as sym_exp

    print("=" * 70)
    print("TEST: APROXIMANTES DE PADÉ")
    print("=" * 70)

    t = Symbol('t')

    # Test 1: exp(-t) ≈ serie de Taylor + Padé
    # La serie de Taylor de exp(-t) truncada a orden 6 diverge para t > ~3
    # Padé [3/3] mantiene precision mucho mas alla
    print("\n--- Test 1: exp(-t), comparar Taylor vs Padé ---")

    # Serie de Taylor orden 6
    e_series = 1 - t + t**2/2 - t**3/6 + t**4/24 - t**5/120 + t**6/720

    # Padé [3/3]
    pade_res = pade_approximant(e_series, t, 3, 3)
    print(f"  Padé [3/3] = {pade_res['pade']}")

    t_test = np.array([0, 1, 2, 3, 4, 5])
    exact = np.exp(-t_test)

    f_taylor = lambdify(t, e_series, modules='numpy')
    taylor_vals = np.array([float(f_taylor(tv)) for tv in t_test])

    pade_vals = pade_eval(pade_res, t_test)

    print(f"\n  {'t':<6} {'Exacto':<15} {'Taylor[6]':<15} {'Padé[3/3]':<15} {'Err Taylor':<12} {'Err Padé':<12}")
    for i, tv in enumerate(t_test):
        print(f"  {tv:<6.1f} {exact[i]:<15.8f} {taylor_vals[i]:<15.8f} "
              f"{pade_vals[i]:<15.8f} {abs(taylor_vals[i]-exact[i]):<12.4e} "
              f"{abs(pade_vals[i]-exact[i]):<12.4e}")

    # Test 2: 1/(1+t) desde serie truncada
    print("\n--- Test 2: 1/(1+t) desde serie geometrica truncada ---")
    geo_series = sum((-t)**k for k in range(8))  # 1 - t + t² - t³ + ...

    pade2 = pade_approximant(geo_series, t, 4, 4)
    print(f"  Padé [4/4] = {simplify(pade2['pade'])}")
    # Deberia dar exactamente 1/(1+t) para esta serie

    print("\n" + "=" * 70)
    print("PADÉ: Tests completados")
    print("=" * 70)
