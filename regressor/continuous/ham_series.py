"""
ham_series.py — Motor principal del HAM continuo

Resuelve N[u(t)] = 0 construyendo la serie homotopica:
    u(t) = u_0(t) + u_1(t) + u_2(t) + ... + u_M(t)

donde cada u_m(t) se obtiene resolviendo la ecuacion de deformacion
de orden m:
    L[u_m - chi_m * u_{m-1}] = hbar * R_m(u_0, ..., u_{m-1})

Implementacion basada en:
    Liao, S.J. "Homotopy Analysis Method in Nonlinear Differential
    Equations", Springer & Higher Education Press, 2012.

Author: Rodolfo H. Rodrigo — UNSJ / INAUT
Fecha: Marzo 2026
"""

import numpy as np
from sympy import (
    Symbol, Function, symbols, diff, integrate, simplify,
    series as sym_series, lambdify, Rational, S, oo,
    Poly, Expr, Add, Mul, Pow, collect, expand, factor,
    sin, cos, exp, sqrt, Derivative, dsolve, Eq,
    Piecewise, zoo, nan, Float
)


def ham_solve(N_expr, y_sym, yp_sym, t_sym, ic, hbar=-1.0, M=10,
              L_operator=None, ypp_sym=None, ic_prime=None,
              t_eval=None, simplify_terms=True):
    """
    Resuelve N[u(t)] = 0 usando el Homotopy Analysis Method.

    Construye M terminos de la serie u(t) = u_0 + u_1 + ... + u_M
    usando la ecuacion de deformacion de orden m con parametro hbar.

    Parameters
    ----------
    N_expr : sympy.Expr
        Operador no lineal N[u] en terminos de y_sym, yp_sym, t_sym.
        Convencion: N[u] = 0 es la ecuacion a resolver.
        Ejemplo 1er orden: yp + y**2 - sin(t)  (para y' + y² = sin(t))
        Ejemplo 2do orden: ypp + 0.1*yp + sin(y)  (para y'' + 0.1y' + sin(y) = 0)

    y_sym : sympy.Symbol
        Simbolo de la variable dependiente (posicion).

    yp_sym : sympy.Symbol
        Simbolo de la primera derivada y'.

    t_sym : sympy.Symbol
        Simbolo de la variable independiente (tiempo).

    ic : float or sympy.Expr
        Condicion inicial u(0) = ic.

    hbar : float
        Parametro de convergencia-control (default: -1.0).
        Valores tipicos: -1.0 <= hbar < 0.

    M : int
        Numero de terminos HAM a calcular (default: 10).

    L_operator : callable or None
        Operador lineal auxiliar L.
        Firma: L_operator(expr, t_sym) -> sympy.Expr
        L opera sobre una expresion simbolica en t.
        Si None, usa L[u] = u' (derivada primera).

    ypp_sym : sympy.Symbol or None
        Simbolo de la segunda derivada y'' (solo para EDOs de 2do orden).

    ic_prime : float or None
        Condicion inicial y'(0) = ic_prime (solo para 2do orden).

    t_eval : np.ndarray or None
        Si se proporciona, evalua la serie en estos puntos.

    simplify_terms : bool
        Si True, simplifica cada termino u_m (mas lento pero mas limpio).

    Returns
    -------
    result : dict
        'terms'    : list of sympy.Expr [u_0, u_1, ..., u_M]
        'series'   : sympy.Expr (suma truncada)
        'hbar'     : float (parametro usado)
        'M'        : int (numero de terminos)
        'R_terms'  : list of sympy.Expr [R_1, R_2, ..., R_M]
        'residual' : sympy.Expr (N[serie] evaluado simbolicamente)
        'values'   : np.ndarray or None (si t_eval proporcionado)
        'order'    : int (1 o 2)
    """
    # Detectar orden
    order = 2 if ypp_sym is not None else 1

    if order == 2 and ic_prime is None:
        raise ValueError("Para EDOs de 2do orden, ic_prime es obligatorio.")

    # Operador L por defecto
    if L_operator is None:
        if order == 1:
            # L[u] = u'  =>  resolver L[u_m] = RHS  =>  u_m = integral de RHS
            L_operator = lambda expr, t: diff(expr, t)
        else:
            # L[u] = u''  =>  resolver L[u_m] = RHS  =>  doble integracion
            L_operator = lambda expr, t: diff(expr, t, 2)

    # Simbolo auxiliar para el parametro de embedding
    q = Symbol('_q_ham')

    # ======================================================================
    # Paso 1: Aproximacion inicial u_0(t)
    # ======================================================================
    # Para L[u] = u': u_0(t) = ic (constante satisface u_0(0) = ic)
    # Para L[u] = u'': u_0(t) = ic + ic_prime * t
    if order == 1:
        u0 = S(ic)
    else:
        u0 = S(ic) + S(ic_prime) * t_sym

    u_terms = [u0]
    R_terms = []

    print(f"HAM continuo: orden {order}, M={M}, hbar={hbar}")
    print(f"  u_0(t) = {u0}")

    # ======================================================================
    # Paso 2: Calcular terminos u_1, u_2, ..., u_M
    # ======================================================================
    for m in range(1, M + 1):
        chi_m = 0 if m == 1 else 1

        # --- Calcular R_m ---
        R_m = _compute_R_m(N_expr, y_sym, yp_sym, ypp_sym,
                           t_sym, q, u_terms, m, order)

        if simplify_terms:
            R_m = simplify(expand(R_m))

        R_terms.append(R_m)

        # --- Resolver ecuacion de deformacion ---
        # L[u_m - chi_m * u_{m-1}] = hbar * R_m
        #
        # Para L[u] = u' (1er orden):
        #   u'_m - chi_m * u'_{m-1} = hbar * R_m
        #   Integrando: u_m = chi_m * u_{m-1} + hbar * ∫_0^t R_m(s) ds + C
        #   Con u_m(0) = 0: C = -chi_m * u_{m-1}(0) = 0
        #                   (porque u_{m-1}(0) = 0 para m >= 2, y chi_1 = 0)
        #
        # Para L[u] = u'' (2do orden):
        #   u''_m - chi_m * u''_{m-1} = hbar * R_m
        #   Doble integracion con u_m(0) = 0, u'_m(0) = 0

        u_m = _solve_deformation(L_operator, u_terms, R_m, hbar, chi_m,
                                 m, t_sym, order)

        if simplify_terms:
            u_m = simplify(expand(u_m))

        u_terms.append(u_m)

        if m <= 5 or m == M:
            # Solo mostrar los primeros y el ultimo para no saturar
            u_m_short = str(u_m)
            if len(u_m_short) > 80:
                u_m_short = u_m_short[:77] + "..."
            print(f"  u_{m}(t) = {u_m_short}")

    # ======================================================================
    # Paso 3: Sumar serie
    # ======================================================================
    series_sum = sum(u_terms)
    if simplify_terms:
        series_sum = simplify(expand(series_sum))

    # ======================================================================
    # Paso 4: Calcular residuo (opcional, puede ser lento para M grande)
    # ======================================================================
    residual = None
    if M <= 15:
        try:
            residual = _eval_N_at(N_expr, y_sym, yp_sym, ypp_sym,
                                  t_sym, series_sum, order)
            residual = simplify(expand(residual))
        except Exception:
            residual = None

    # ======================================================================
    # Paso 5: Evaluar numericamente si se pide
    # ======================================================================
    values = None
    if t_eval is not None:
        try:
            f_eval = lambdify(t_sym, series_sum, modules='numpy')
            values = np.array([float(f_eval(tv)) for tv in t_eval])
        except Exception as e:
            print(f"  Advertencia: evaluacion numerica fallo: {e}")
            values = None

    result = {
        'terms': u_terms,
        'series': series_sum,
        'hbar': hbar,
        'M': M,
        'R_terms': R_terms,
        'residual': residual,
        'values': values,
        'order': order,
        't_sym': t_sym,
        'y_sym': y_sym,
    }

    print(f"  Serie construida: {M} terminos.")
    if residual is not None:
        res_str = str(residual)
        if len(res_str) > 80:
            res_str = res_str[:77] + "..."
        print(f"  Residuo: {res_str}")
    print()

    return result


def ham_solve_system(N_exprs, var_syms, varp_syms, t_sym, ics,
                     hbar=-1.0, M=10, t_eval=None):
    """
    Resuelve un sistema de EDOs N_i[x, y, z, ...] = 0 usando HAM continuo.

    Parameters
    ----------
    N_exprs : list of sympy.Expr
        [N_1, N_2, ...] operadores no lineales. N_i = 0.

    var_syms : list of sympy.Symbol
        [x, y, z, ...] variables dependientes.

    varp_syms : list of sympy.Symbol
        [xp, yp, zp, ...] primeras derivadas.

    t_sym : sympy.Symbol
        Variable independiente.

    ics : list of float
        [x(0), y(0), z(0), ...] condiciones iniciales.

    hbar : float
        Parametro de convergencia-control.

    M : int
        Numero de terminos.

    t_eval : np.ndarray or None
        Puntos de evaluacion.

    Returns
    -------
    result : dict
        'terms'   : list of list (terms[i] = [u0_i, u1_i, ...])
        'series'  : list of sympy.Expr (serie para cada variable)
        'values'  : list of np.ndarray or None
    """
    n_vars = len(var_syms)
    q = Symbol('_q_ham')

    # Condiciones iniciales: u0_i = ics[i] (constantes)
    all_terms = [[S(ics[i])] for i in range(n_vars)]

    print(f"HAM sistema: {n_vars} variables, M={M}, hbar={hbar}")
    for i in range(n_vars):
        print(f"  {var_syms[i]}_0 = {ics[i]}")

    for m in range(1, M + 1):
        chi_m = 0 if m == 1 else 1

        # Construir phi_i = sum(u_k_i * q^k, k=0..m-1) para cada variable
        phi = {}
        phi_p = {}
        for i in range(n_vars):
            phi[var_syms[i]] = sum(
                all_terms[i][k] * q**k for k in range(len(all_terms[i]))
            )
            phi_p[varp_syms[i]] = sum(
                diff(all_terms[i][k], t_sym) * q**k
                for k in range(len(all_terms[i]))
            )

        # Calcular R_m para cada ecuacion
        new_terms = []
        for i in range(n_vars):
            # Sustituir en N_i
            N_sub = N_exprs[i]
            for j in range(n_vars):
                N_sub = N_sub.subs(var_syms[j], phi[var_syms[j]])
                N_sub = N_sub.subs(varp_syms[j], phi_p[varp_syms[j]])

            # Expandir en q y extraer coeficiente de q^{m-1}
            N_expanded = expand(N_sub)
            # Usar Poly para extraer coeficientes eficientemente
            try:
                R_m_i = Poly(N_expanded, q).nth(m - 1)
            except Exception:
                # Fallback: series expansion
                N_series = sym_series(N_sub, q, 0, m).removeO()
                R_m_i = N_series.coeff(q, m - 1)

            R_m_i = simplify(expand(R_m_i))

            # Resolver: u'_{m,i} = chi_m * u'_{m-1,i} + hbar * R_m_i
            # => u_{m,i} = chi_m * u_{m-1,i} + hbar * integral(R_m_i)
            # con u_{m,i}(0) = 0

            prev = all_terms[i][-1] if chi_m == 1 else S.Zero
            integral_Rm = integrate(R_m_i, (t_sym, 0, t_sym))
            u_m_i = chi_m * prev + S(hbar) * integral_Rm
            u_m_i = simplify(expand(u_m_i))

            new_terms.append(u_m_i)

        for i in range(n_vars):
            all_terms[i].append(new_terms[i])

        if m <= 3 or m == M:
            for i in range(n_vars):
                s = str(new_terms[i])
                if len(s) > 60:
                    s = s[:57] + "..."
                print(f"  {var_syms[i]}_{m} = {s}")

    # Sumar series
    series_list = [sum(all_terms[i]) for i in range(n_vars)]
    series_list = [simplify(expand(s)) for s in series_list]

    # Evaluar numericamente
    values_list = None
    if t_eval is not None:
        values_list = []
        for i in range(n_vars):
            try:
                f_eval = lambdify(t_sym, series_list[i], modules='numpy')
                vals = np.array([float(f_eval(tv)) for tv in t_eval])
                values_list.append(vals)
            except Exception:
                values_list.append(None)

    result = {
        'terms': all_terms,
        'series': series_list,
        'hbar': hbar,
        'M': M,
        'values': values_list,
        't_sym': t_sym,
        'var_syms': var_syms,
    }

    print(f"  Sistema resuelto: {M} terminos.\n")
    return result


# ======================================================================
# Funciones internas
# ======================================================================

def _compute_R_m(N_expr, y_sym, yp_sym, ypp_sym, t_sym, q, u_terms, m, order):
    """
    Calcula R_m = (1/(m-1)!) * d^{m-1} N[phi]/dq^{m-1} |_{q=0}

    Implementacion: sustituye phi en N, expande en serie de q,
    extrae coeficiente de q^{m-1}.
    """
    # Construir phi = u_0 + u_1*q + u_2*q^2 + ...
    phi = sum(u_terms[k] * q**k for k in range(m))
    phi_p = sum(diff(u_terms[k], t_sym) * q**k for k in range(m))

    if order == 2:
        phi_pp = sum(diff(u_terms[k], t_sym, 2) * q**k for k in range(m))

    # Sustituir en N_expr
    N_sub = N_expr.subs(y_sym, phi).subs(yp_sym, phi_p)
    if order == 2 and ypp_sym is not None:
        N_sub = N_sub.subs(ypp_sym, phi_pp)

    # Expandir como polinomio en q
    N_expanded = expand(N_sub)

    # Extraer coeficiente de q^{m-1}
    try:
        R_m = Poly(N_expanded, q).nth(m - 1)
    except Exception:
        # Fallback para expresiones no polinomicas en q
        # (por ejemplo, sin(phi) produce terminos transcendentes)
        # Usar Taylor series en q
        N_taylor = sym_series(N_sub, q, 0, m).removeO()
        R_m = N_taylor.coeff(q, m - 1)

    return R_m


def _solve_deformation(L_operator, u_terms, R_m, hbar, chi_m, m, t_sym, order):
    """
    Resuelve la ecuacion de deformacion de orden m:
        L[u_m - chi_m * u_{m-1}] = hbar * R_m
    con u_m(0) = 0 (y u'_m(0) = 0 para 2do orden).

    Para L = d/dt (1er orden):
        u_m = chi_m * u_{m-1} + hbar * integral_0^t R_m ds

    Para L = d²/dt² (2do orden):
        u_m = chi_m * u_{m-1} + hbar * integral_0^t integral_0^s R_m(r) dr ds
    """
    hbar_sym = S(hbar)

    if order == 1:
        # L[u] = u' => integracion simple
        prev = u_terms[m - 1] if chi_m == 1 else S.Zero
        integral_Rm = integrate(R_m, (t_sym, 0, t_sym))
        u_m = chi_m * prev + hbar_sym * integral_Rm

    elif order == 2:
        # L[u] = u'' => doble integracion
        prev = u_terms[m - 1] if chi_m == 1 else S.Zero
        # Primera integracion
        inner = integrate(R_m, (t_sym, 0, t_sym))
        # Segunda integracion
        outer = integrate(inner, (t_sym, 0, t_sym))
        u_m = chi_m * prev + hbar_sym * outer

    else:
        raise ValueError(f"Orden {order} no soportado")

    return u_m


def _eval_N_at(N_expr, y_sym, yp_sym, ypp_sym, t_sym, u_approx, order):
    """
    Evalua N[u_approx] sustituyendo la serie truncada.
    """
    u_p = diff(u_approx, t_sym)
    N_val = N_expr.subs(y_sym, u_approx).subs(yp_sym, u_p)
    if order == 2 and ypp_sym is not None:
        u_pp = diff(u_approx, t_sym, 2)
        N_val = N_val.subs(ypp_sym, u_pp)
    return N_val


def evaluate_series(result, t_eval):
    """
    Evalua una serie HAM ya calculada en puntos arbitrarios.

    Parameters
    ----------
    result : dict
        Resultado de ham_solve().
    t_eval : np.ndarray
        Puntos de evaluacion.

    Returns
    -------
    values : np.ndarray
    """
    t_sym = result['t_sym']
    series_expr = result['series']
    f_eval = lambdify(t_sym, series_expr, modules='numpy')
    return np.array([float(f_eval(tv)) for tv in t_eval])


def partial_sums(result, t_eval):
    """
    Calcula las sumas parciales S_0, S_1, ..., S_M en t_eval.
    Util para visualizar convergencia.

    Returns
    -------
    sums : np.ndarray, shape (M+1, len(t_eval))
    """
    t_sym = result['t_sym']
    terms = result['terms']
    M = len(terms) - 1

    sums = np.zeros((M + 1, len(t_eval)))
    partial = S.Zero

    for m in range(M + 1):
        partial = partial + terms[m]
        f_eval = lambdify(t_sym, partial, modules='numpy')
        sums[m, :] = np.array([float(f_eval(tv)) for tv in t_eval])

    return sums


# ======================================================================
# Test
# ======================================================================

if __name__ == "__main__":
    from scipy.integrate import solve_ivp

    print("=" * 70)
    print("TEST HAM CONTINUO")
    print("=" * 70)

    # --- Test 1: y' + y = 0, y(0) = 1 ---
    # Solucion exacta: y = exp(-t)
    print("\n--- Test 1: y' + y = 0, y(0) = 1 ---")
    y, yp, t = symbols('y yp t')

    N1 = yp + y  # N[u] = u' + u = 0
    result1 = ham_solve(N1, y, yp, t, ic=1.0, hbar=-1.0, M=10)

    t_test = np.linspace(0, 3, 50)
    vals = evaluate_series(result1, t_test)
    exact = np.exp(-t_test)
    err1 = np.max(np.abs(vals - exact))
    print(f"  Error max vs exp(-t): {err1:.4e}")
    print(f"  {'PASS' if err1 < 1e-4 else 'FAIL'}")

    # --- Test 2: y' + y² = 0, y(0) = 1 ---
    # Solucion exacta: y = 1/(1+t)
    print("\n--- Test 2: y' + y² = 0, y(0) = 1 ---")
    N2 = yp + y**2
    result2 = ham_solve(N2, y, yp, t, ic=1.0, hbar=-1.0, M=15)

    vals2 = evaluate_series(result2, t_test)
    exact2 = 1.0 / (1.0 + t_test)
    err2 = np.max(np.abs(vals2 - exact2))
    print(f"  Error max vs 1/(1+t): {err2:.4e}")
    print(f"  {'PASS' if err2 < 0.1 else 'FAIL'}")

    # --- Test 3: y' = y - y², y(0) = 0.5 (ecuacion logistica) ---
    # Solucion exacta: y = 1/(1 + exp(-t))
    print("\n--- Test 3: Logistica y' = y - y², y(0) = 0.5 ---")
    N3 = yp - y + y**2
    result3 = ham_solve(N3, y, yp, t, ic=0.5, hbar=-1.0, M=12)

    vals3 = evaluate_series(result3, t_test)
    exact3 = 1.0 / (1.0 + np.exp(-t_test))
    err3 = np.max(np.abs(vals3 - exact3))
    print(f"  Error max vs logistica: {err3:.4e}")
    print(f"  {'PASS' if err3 < 0.1 else 'FAIL'}")

    # --- Test 4: Sistema Lotka-Volterra ---
    print("\n--- Test 4: Lotka-Volterra (sistema 2D) ---")
    x, y_lv, xp_s, yp_s = symbols('x y_lv xp yp')

    alpha_p, beta_p, gamma_p, delta_p = 1.0, 0.1, 1.5, 0.075
    N_x = xp_s - alpha_p*x + beta_p*x*y_lv
    N_y = yp_s - delta_p*x*y_lv + gamma_p*y_lv

    result4 = ham_solve_system(
        [N_x, N_y], [x, y_lv], [xp_s, yp_s], t,
        ics=[10.0, 5.0], hbar=-1.0, M=8,
        t_eval=np.linspace(0, 1, 20)
    )

    # Comparar con RK45
    def rhs_lv(t, z):
        return [alpha_p*z[0] - beta_p*z[0]*z[1],
                delta_p*z[0]*z[1] - gamma_p*z[1]]

    t_short = np.linspace(0, 1, 20)
    sol_ref = solve_ivp(rhs_lv, (0, 1), [10.0, 5.0],
                         t_eval=t_short, rtol=1e-9)

    if result4['values'] is not None:
        err4_x = np.max(np.abs(result4['values'][0] - sol_ref.y[0]))
        err4_y = np.max(np.abs(result4['values'][1] - sol_ref.y[1]))
        print(f"  Error x: {err4_x:.4e}")
        print(f"  Error y: {err4_y:.4e}")
        print(f"  {'PASS' if max(err4_x, err4_y) < 1.0 else 'FAIL'}")
    else:
        print("  No se pudo evaluar numericamente")

    print("\n" + "=" * 70)
    print("HAM CONTINUO: Tests completados")
    print("=" * 70)
