"""
examples_continuous.py — Ejemplos del HAM continuo con verificacion cruzada

Cada ejemplo:
1. Resuelve con HAM continuo (series + Padé)
2. Resuelve con regresor discreto (si aplica)
3. Compara con RK45

Demuestra la equivalencia entre ambos paradigmas y las ventajas de cada uno.

Author: Rodolfo H. Rodrigo — UNSJ / INAUT
Fecha: Marzo 2026
"""

import numpy as np
from sympy import symbols, Symbol, sin, cos, exp, Rational, S, simplify
from scipy.integrate import solve_ivp

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from continuous.ham_series import ham_solve, ham_solve_system, evaluate_series, partial_sums
from continuous.convergence import convergence_table, print_convergence_table
from continuous.pade import pade_from_ham, pade_eval, pade_diagonal_sequence


def ejemplo_1_lineal():
    """
    Ejemplo 1: y' + y = 0, y(0) = 1
    Solucion exacta: y(t) = exp(-t)

    Caso trivial: la serie HAM con hbar=-1 reproduce la serie de Taylor
    de exp(-t), que converge para todo t.
    """
    print("=" * 70)
    print("EJEMPLO 1: y' + y = 0, y(0) = 1")
    print("Solucion exacta: exp(-t)")
    print("=" * 70)

    y, yp, t = symbols('y yp t')
    N = yp + y

    result = ham_solve(N, y, yp, t, ic=1.0, hbar=-1.0, M=12)

    # Tabla de convergencia en t=1 (exacto: 1/e = 0.367879...)
    table = convergence_table(result, t_point=1.0, reference=float(exp(-1)))
    print_convergence_table(table, "exp(-t) en t=1")

    # Comparar con Padé
    pade_res = pade_from_ham(result, m=6, n=6)

    t_eval = np.linspace(0, 5, 100)
    vals_series = evaluate_series(result, t_eval)
    vals_pade = pade_eval(pade_res, t_eval)
    exact = np.exp(-t_eval)

    err_series = np.max(np.abs(vals_series - exact))
    err_pade = np.max(np.abs(vals_pade - exact))

    print(f"  Error serie M=12 en [0,5]:  {err_series:.4e}")
    print(f"  Error Padé [6/6] en [0,5]:  {err_pade:.4e}")
    print(f"  Mejora Padé: {err_series/max(err_pade, 1e-16):.1f}x\n")

    return result


def ejemplo_2_cuadratica():
    """
    Ejemplo 2: y' + y² = 0, y(0) = 1
    Solucion exacta: y(t) = 1/(1+t)

    La serie de Taylor tiene radio de convergencia = 1.
    Padé extiende la validez a todo t > 0.
    """
    print("=" * 70)
    print("EJEMPLO 2: y' + y² = 0, y(0) = 1")
    print("Solucion exacta: 1/(1+t)")
    print("=" * 70)

    y, yp, t = symbols('y yp t')
    N = yp + y**2

    result = ham_solve(N, y, yp, t, ic=1.0, hbar=-1.0, M=15)

    # Tabla de convergencia
    table = convergence_table(result, t_point=0.5, reference=1.0/1.5)
    print_convergence_table(table, "1/(1+t) en t=0.5")

    # Serie vs Padé en [0, 5]
    pade_res = pade_from_ham(result, m=7, n=7)

    t_eval = np.linspace(0, 5, 100)
    vals_series = evaluate_series(result, t_eval)
    vals_pade = pade_eval(pade_res, t_eval)
    exact = 1.0 / (1.0 + t_eval)

    # La serie diverge para t > 1, Padé no
    err_series_short = np.max(np.abs(vals_series[:20] - exact[:20]))  # t in [0, 1]
    err_series_long = np.max(np.abs(vals_series[50:] - exact[50:]))   # t in [2.5, 5]
    err_pade = np.max(np.abs(vals_pade - exact))

    print(f"  Error serie en [0, 1]:    {err_series_short:.4e}")
    print(f"  Error serie en [2.5, 5]:  {err_series_long:.4e}  (¡diverge!)")
    print(f"  Error Padé [7/7] en [0,5]: {err_pade:.4e}")
    print(f"  Padé extiende convergencia mas alla del radio de Taylor.\n")

    return result


def ejemplo_3_logistica():
    """
    Ejemplo 3: y' = y - y², y(0) = 0.5
    Solucion exacta: y(t) = 1/(1 + exp(-t))

    Ecuacion logistica clasica.
    """
    print("=" * 70)
    print("EJEMPLO 3: Logistica y' = y - y², y(0) = 0.5")
    print("Solucion exacta: 1/(1 + exp(-t))")
    print("=" * 70)

    y, yp, t = symbols('y yp t')
    N = yp - y + y**2

    result = ham_solve(N, y, yp, t, ic=0.5, hbar=-1.0, M=12)

    t_eval = np.linspace(0, 4, 80)
    vals = evaluate_series(result, t_eval)
    exact = 1.0 / (1.0 + np.exp(-t_eval))

    err = np.max(np.abs(vals - exact))
    print(f"  Error max serie M=12 en [0,4]: {err:.4e}")

    # Padé
    pade_res = pade_from_ham(result, m=6, n=6)
    vals_pade = pade_eval(pade_res, t_eval)
    err_pade = np.max(np.abs(vals_pade - exact))
    print(f"  Error max Padé [6/6] en [0,4]: {err_pade:.4e}\n")

    return result


def ejemplo_4_pendulo():
    """
    Ejemplo 4: y'' + sin(y) = 0, y(0) = pi/4, y'(0) = 0
    Pendulo simple (no lineal, 2do orden)

    La serie de Taylor de sin(y) = y - y³/6 + ... permite
    calcular terminos HAM hasta el orden deseado.

    Nota: usamos sin(y) ≈ y - y³/6 para que SymPy pueda
    expandir en serie de q. Para sin exacto, se necesita
    expansion previa.
    """
    print("=" * 70)
    print("EJEMPLO 4: Pendulo y'' + sin(y) = 0")
    print("y(0) = pi/4, y'(0) = 0")
    print("=" * 70)

    y, yp, ypp, t = symbols('y yp ypp t')

    # Aproximacion de sin(y) por Taylor: sin(y) ≈ y - y³/6 + y⁵/120
    # Esto permite la expansion en serie de q
    N_approx = ypp + y - y**3/6 + y**5/120

    from sympy import pi
    ic_val = float(pi / 4)

    result = ham_solve(N_approx, y, yp, t, ic=ic_val, hbar=-1.0, M=8,
                       ypp_sym=ypp, ic_prime=0.0)

    # Comparar con RK45
    t_eval = np.linspace(0, 6, 100)

    def rhs_pendulum(t, z):
        return [z[1], -np.sin(z[0])]

    sol_ref = solve_ivp(rhs_pendulum, (0, 6), [ic_val, 0.0],
                         t_eval=t_eval, rtol=1e-10)

    vals = evaluate_series(result, t_eval)
    err = np.max(np.abs(vals - sol_ref.y[0]))
    print(f"  Error max vs RK45 en [0,6]: {err:.4e}")

    # Padé
    pade_res = pade_from_ham(result, m=4, n=4)
    vals_pade = pade_eval(pade_res, t_eval)
    err_pade = np.max(np.abs(vals_pade - sol_ref.y[0]))
    print(f"  Error max Padé [4/4] en [0,6]: {err_pade:.4e}\n")

    return result


def ejemplo_5_lotka_volterra_continuo():
    """
    Ejemplo 5: Sistema Lotka-Volterra (HAM continuo)
    x' = α·x - β·x·y     (presas)
    y' = δ·x·y - γ·y      (depredadores)

    Compara HAM continuo vs RK45.
    """
    print("=" * 70)
    print("EJEMPLO 5: Lotka-Volterra (HAM continuo, sistema 2D)")
    print("=" * 70)

    x, y_lv, xp, yp, t = symbols('x y_lv xp yp t')

    alpha, beta, gamma, delta = 1.0, 0.1, 1.5, 0.075
    N_x = xp - alpha*x + beta*x*y_lv
    N_y = yp - delta*x*y_lv + gamma*y_lv

    t_eval = np.linspace(0, 2, 50)

    result = ham_solve_system(
        [N_x, N_y], [x, y_lv], [xp, yp], t,
        ics=[10.0, 5.0], hbar=-1.0, M=10, t_eval=t_eval
    )

    # RK45 referencia
    def rhs(t, z):
        return [alpha*z[0] - beta*z[0]*z[1],
                delta*z[0]*z[1] - gamma*z[1]]

    sol_ref = solve_ivp(rhs, (0, 2), [10.0, 5.0],
                         t_eval=t_eval, rtol=1e-10)

    if result['values'] is not None:
        err_x = np.max(np.abs(result['values'][0] - sol_ref.y[0]))
        err_y = np.max(np.abs(result['values'][1] - sol_ref.y[1]))
        print(f"  Error x (presas) en [0,2]:      {err_x:.4e}")
        print(f"  Error y (depredadores) en [0,2]: {err_y:.4e}")
    else:
        print("  Evaluacion numerica no disponible")

    print()
    return result


def ejemplo_6_verificacion_cruzada():
    """
    Ejemplo 6: Verificacion cruzada Discreto vs Continuo

    Ecuacion: y' + y² = sin(5t), y(0) = -0.2

    Resuelve con:
    1. HAM continuo (M=12)
    2. Regresor discreto (solver.py)
    3. RK45

    Verifica que ambos paradigmas convergen a la misma solucion.
    """
    print("=" * 70)
    print("EJEMPLO 6: Verificacion cruzada Discreto vs Continuo")
    print("y' + y² = sin(5t), y(0) = -0.2")
    print("=" * 70)

    y, yp, t = symbols('y yp t')

    # --- HAM Continuo ---
    N = yp + y**2 - sin(5*t)
    result_cont = ham_solve(N, y, yp, t, ic=-0.2, hbar=-1.0, M=12)

    # --- Regresor Discreto ---
    from solver import solve_order1
    from scipy.integrate import odeint

    n_disc = 500
    t_disc = np.linspace(0, 2, n_disc)
    T_disc = t_disc[1] - t_disc[0]

    # Referencia RK
    sol_odeint = odeint(lambda y, t: -y**2 + np.sin(5*t), -0.2, t_disc).ravel()

    # Regresor discreto
    f   = lambda y: y**2
    df  = lambda y: 2*y
    d2f = lambda y: 2.0
    d3f = lambda y: 0.0
    u_disc = np.sin(5*t_disc)

    y_discrete = solve_order1(f, df, d2f, d3f, u_disc,
                               sol_odeint[0], sol_odeint[1], T_disc, n_disc)

    # --- Comparar en t in [0, 2] ---
    t_eval = np.linspace(0, 2, 50)
    vals_cont = evaluate_series(result_cont, t_eval)

    # RK45 referencia fina
    sol_ref = solve_ivp(lambda t, y: [-y[0]**2 + np.sin(5*t)],
                         (0, 2), [-0.2], t_eval=t_eval, rtol=1e-10)

    # Interpolar discreto en t_eval
    y_disc_interp = np.interp(t_eval, t_disc, y_discrete)

    err_cont = np.max(np.abs(vals_cont - sol_ref.y[0]))
    err_disc = np.max(np.abs(y_disc_interp - sol_ref.y[0]))

    print(f"\n  Error HAM continuo (M=12) en [0,2]: {err_cont:.4e}")
    print(f"  Error regresor discreto (n=500) en [0,2]: {err_disc:.4e}")

    # Verificar que Padé mejora el continuo
    pade_res = pade_from_ham(result_cont, m=6, n=6)
    vals_pade = pade_eval(pade_res, t_eval)
    err_pade = np.max(np.abs(vals_pade - sol_ref.y[0]))
    print(f"  Error Padé [6/6] en [0,2]:           {err_pade:.4e}")

    print()
    return result_cont


# ======================================================================
# Main
# ======================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print(" EJEMPLOS HAM CONTINUO + VERIFICACION CRUZADA")
    print(" Libreria homotopy_regressors — paradigma Liao")
    print("=" * 70 + "\n")

    results = {}

    try:
        results['Ej1'] = ejemplo_1_lineal()
    except Exception as e:
        print(f"  ERROR en Ejemplo 1: {e}\n")

    try:
        results['Ej2'] = ejemplo_2_cuadratica()
    except Exception as e:
        print(f"  ERROR en Ejemplo 2: {e}\n")

    try:
        results['Ej3'] = ejemplo_3_logistica()
    except Exception as e:
        print(f"  ERROR en Ejemplo 3: {e}\n")

    try:
        results['Ej4'] = ejemplo_4_pendulo()
    except Exception as e:
        print(f"  ERROR en Ejemplo 4: {e}\n")

    try:
        results['Ej5'] = ejemplo_5_lotka_volterra_continuo()
    except Exception as e:
        print(f"  ERROR en Ejemplo 5: {e}\n")

    try:
        results['Ej6'] = ejemplo_6_verificacion_cruzada()
    except Exception as e:
        print(f"  ERROR en Ejemplo 6: {e}\n")

    print("=" * 70)
    print(" RESUMEN")
    print("=" * 70)
    print(f"  Ejemplos ejecutados: {len(results)}/6")
    print(f"  Paradigma continuo + discreto integrados en una libreria.")
    print("=" * 70 + "\n")
