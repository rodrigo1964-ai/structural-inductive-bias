#!/usr/bin/env python3
"""
test_all.py — Test rápido de toda la librería homotopy_regressors

Ejecutar:  cd /home/rodo/regressor && python3 test_all.py

Author: Rodolfo H. Rodrigo — UNSJ / INAUT
"""

import sys
import numpy as np

print("=" * 70)
print(" TEST COMPLETO: homotopy_regressors v0.2.0")
print(" Discreto + Continuo")
print("=" * 70)

passed = 0
failed = 0
errors = []

def run_test(name, func):
    global passed, failed, errors
    try:
        func()
        passed += 1
        print(f"  ✓ {name}")
    except Exception as e:
        failed += 1
        errors.append((name, str(e)))
        print(f"  ✗ {name}: {e}")


# ======================================================================
# 1. IMPORTS
# ======================================================================
print("\n--- 1. Imports ---")

def test_imports_discrete():
    from solver import solve_order1, solve_order2, solve_order1_numeric
    from solver_system import solve_system, solve_system_numeric
    from regressor import build_regressor_order1, build_regressor_order2, build_inverse_regressor
    from regressor_system import build_system_regressor

run_test("Imports discreto", test_imports_discrete)

def test_imports_continuous():
    from continuous.ham_series import ham_solve, ham_solve_system
    from continuous.convergence import hbar_curve, optimal_hbar
    from continuous.pade import pade_approximant, pade_eval
    from continuous.operators import L_derivative, L_second, L_harmonic

run_test("Imports continuo", test_imports_continuous)

def test_imports_tools():
    from parser import parse_ode
    from derivatives import discrete_derivatives
    from identify_parameters import check_lip, identify_lip

run_test("Imports herramientas", test_imports_tools)


# ======================================================================
# 2. DISCRETO — Solver escalar
# ======================================================================
print("\n--- 2. Discreto: solver escalar ---")

def test_discrete_order1():
    from solver import solve_order1
    # y' + 2y = 0, y(0) = 1 => y = exp(-2t)
    f = lambda y: 2*y
    df = lambda y: 2.0
    d2f = lambda y: 0.0
    d3f = lambda y: 0.0
    n = 500
    t = np.linspace(0, 3, n)
    T = t[1] - t[0]
    u = np.zeros(n)
    exact = np.exp(-2*t)
    y = solve_order1(f, df, d2f, d3f, u, exact[0], exact[1], T, n)
    err = np.max(np.abs(y - exact))
    assert err < 1e-3, f"Error demasiado grande: {err:.4e}"

run_test("solve_order1 (lineal)", test_discrete_order1)

def test_discrete_order1_nonlinear():
    from solver import solve_order1
    from scipy.integrate import odeint
    # y' + y² = sin(5t)
    f = lambda y: y**2
    df = lambda y: 2*y
    d2f = lambda y: 2.0
    d3f = lambda y: 0.0
    n = 500
    t = np.linspace(0, 10, n)
    T = t[1] - t[0]
    u = np.sin(5*t)
    ref = odeint(lambda y, t: -y**2 + np.sin(5*t), -0.2, t).ravel()
    y = solve_order1(f, df, d2f, d3f, u, ref[0], ref[1], T, n)
    err = np.max(np.abs(y - ref))
    assert err < 1e-2, f"Error: {err:.4e}"

run_test("solve_order1 (no lineal)", test_discrete_order1_nonlinear)


# ======================================================================
# 3. DISCRETO — Solver de sistemas
# ======================================================================
print("\n--- 3. Discreto: solver de sistemas ---")

def test_discrete_system_lotka():
    from solver_system import solve_system_numeric
    from scipy.integrate import solve_ivp
    alpha, beta, gamma, delta = 1.0, 0.1, 1.5, 0.075
    def F(x, y, z, w, xp, yp, zp, wp, xpp, ypp, zpp, wpp, t):
        return xp - alpha*x + beta*x*y
    def G(x, y, z, w, xp, yp, zp, wp, xpp, ypp, zpp, wpp, t):
        return yp - delta*x*y + gamma*y
    n = 3000
    t_arr = np.linspace(0, 30, n)
    T = t_arr[1] - t_arr[0]
    sol = solve_ivp(lambda t, z: [alpha*z[0]-beta*z[0]*z[1],
                                   delta*z[0]*z[1]-gamma*z[1]],
                     (0, 30), [10, 5], t_eval=t_arr, rtol=1e-9)
    ic = [[10.0, sol.y[0, 1]], [5.0, sol.y[1, 1]]]
    u = [np.zeros(n), np.zeros(n)]
    results = solve_system_numeric([F, G], u, ic, T, n)
    err_x = np.max(np.abs(results[0] - sol.y[0]))
    err_y = np.max(np.abs(results[1] - sol.y[1]))
    assert err_x < 0.1, f"Error x: {err_x:.4e}"
    assert err_y < 0.1, f"Error y: {err_y:.4e}"

run_test("solve_system_numeric (Lotka-Volterra)", test_discrete_system_lotka)


# ======================================================================
# 4. CONTINUO — HAM series
# ======================================================================
print("\n--- 4. Continuo: HAM series ---")

def test_ham_lineal():
    from continuous.ham_series import ham_solve, evaluate_series
    from sympy import symbols
    y, yp, t = symbols('y yp t')
    # Resolver silenciosamente
    import io
    old = sys.stdout; sys.stdout = io.StringIO()
    result = ham_solve(yp + y, y, yp, t, ic=1.0, hbar=-1.0, M=10)
    sys.stdout = old
    t_eval = np.linspace(0, 3, 50)
    vals = evaluate_series(result, t_eval)
    exact = np.exp(-t_eval)
    err = np.max(np.abs(vals - exact))
    assert err < 1e-3, f"Error: {err:.4e}"

run_test("ham_solve (y'+y=0, lineal)", test_ham_lineal)

def test_ham_cuadratica():
    from continuous.ham_series import ham_solve, evaluate_series
    from sympy import symbols
    y, yp, t = symbols('y yp t')
    import io
    old = sys.stdout; sys.stdout = io.StringIO()
    result = ham_solve(yp + y**2, y, yp, t, ic=1.0, hbar=-1.0, M=12)
    sys.stdout = old
    t_eval = np.linspace(0, 0.8, 30)
    vals = evaluate_series(result, t_eval)
    exact = 1.0 / (1.0 + t_eval)
    err = np.max(np.abs(vals - exact))
    assert err < 0.01, f"Error: {err:.4e}"

run_test("ham_solve (y'+y²=0, no lineal)", test_ham_cuadratica)

def test_ham_sistema():
    from continuous.ham_series import ham_solve_system
    from sympy import symbols
    from scipy.integrate import solve_ivp
    x, y_lv, xp, yp, t = symbols('x y_lv xp yp t')
    N_x = xp - 1.0*x + 0.1*x*y_lv
    N_y = yp - 0.075*x*y_lv + 1.5*y_lv
    t_eval = np.linspace(0, 0.5, 20)
    import io
    old = sys.stdout; sys.stdout = io.StringIO()
    result = ham_solve_system([N_x, N_y], [x, y_lv], [xp, yp], t,
                               ics=[10.0, 5.0], hbar=-1.0, M=6, t_eval=t_eval)
    sys.stdout = old
    sol = solve_ivp(lambda t, z: [z[0]-0.1*z[0]*z[1], 0.075*z[0]*z[1]-1.5*z[1]],
                     (0, 0.5), [10, 5], t_eval=t_eval, rtol=1e-10)
    assert result['values'] is not None
    err_x = np.max(np.abs(result['values'][0] - sol.y[0]))
    assert err_x < 2.0, f"Error x: {err_x:.4e}"

run_test("ham_solve_system (Lotka-Volterra)", test_ham_sistema)


# ======================================================================
# 5. PADÉ
# ======================================================================
print("\n--- 5. Aproximantes de Padé ---")

def test_pade_exp():
    from continuous.pade import pade_approximant, pade_eval
    from sympy import Symbol
    t = Symbol('t')
    # Taylor de exp(-t) hasta orden 6
    e_series = 1 - t + t**2/2 - t**3/6 + t**4/24 - t**5/120 + t**6/720
    pade_res = pade_approximant(e_series, t, 3, 3)
    t_test = np.array([0, 1, 2, 3])
    vals = pade_eval(pade_res, t_test)
    exact = np.exp(-t_test)
    err = np.max(np.abs(vals - exact))
    assert err < 0.01, f"Error: {err:.4e}"

run_test("Padé [3/3] de exp(-t)", test_pade_exp)

def test_pade_geometric():
    from continuous.pade import pade_approximant
    from sympy import Symbol, simplify
    t = Symbol('t')
    # Serie geometrica: 1 - t + t² - t³ + ... (convergencia para |t|<1)
    geo = sum((-t)**k for k in range(8))
    pade_res = pade_approximant(geo, t, 4, 4)
    # Debe dar exactamente 1/(1+t)
    exact = 1/(1+t)
    diff_expr = simplify(pade_res['pade'] - exact)
    assert diff_expr == 0, f"No es exacto: {pade_res['pade']}"

run_test("Padé [4/4] de serie geométrica", test_pade_geometric)


# ======================================================================
# 6. HERRAMIENTAS
# ======================================================================
print("\n--- 6. Herramientas ---")

def test_parser():
    from parser import parse_ode
    f, u, order, info = parse_ode("y' + y**2 = sin(5*t)")
    assert order == 1
    f2, u2, order2, info2 = parse_ode("y'' + 0.1*y' + sin(y) = sin(3*t)")
    assert order2 == 2

run_test("Parser de ODEs", test_parser)

def test_derivatives():
    from derivatives import discrete_derivatives
    d3 = discrete_derivatives(3)
    assert 1 in d3  # primera derivada
    assert 2 in d3  # segunda derivada

run_test("Derivadas discretas", test_derivatives)

def test_check_lip():
    from identify_parameters import check_lip
    from sympy import symbols
    y, yp, ypp, u, t = symbols('y yp ypp u t')
    a, b = symbols('a b')
    F_lip = yp + a*y + b*y**2 - u
    is_lip, Phi, r = check_lip(F_lip, [a, b])
    assert is_lip == True

run_test("check_lip (detección LIP)", test_check_lip)


# ======================================================================
# RESUMEN
# ======================================================================
print("\n" + "=" * 70)
print(f" RESULTADOS: {passed} pasaron, {failed} fallaron")
print("=" * 70)

if errors:
    print("\nErrores:")
    for name, err in errors:
        print(f"  ✗ {name}: {err}")

if failed == 0:
    print("\n✓ TODOS LOS TESTS PASARON\n")
    sys.exit(0)
else:
    print(f"\n✗ {failed} TESTS FALLARON\n")
    sys.exit(1)
