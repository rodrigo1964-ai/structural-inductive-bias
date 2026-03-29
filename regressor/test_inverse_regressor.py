"""
test_inverse_regressor.py - Validación del regresor homotópico inverso

Tests basados en CLAUDE_inverse_regressor.md (Teorema 1, Paper 10)

Author: Rodolfo H. Rodrigo - UNSJ
Fecha: Marzo 2026
"""

import numpy as np
from sympy import symbols, Symbol, sin as sym_sin, cos as sym_cos
from scipy.integrate import solve_ivp
from regressor import build_inverse_regressor, build_regressor_order1


def test1_motor_lineal():
    """
    Test 1 - Motor DC simplificado (u lineal en F)

    Sistema: L0*yp + R*y + Ke*w - u = 0
    con L0=0.01, R=1.0, Ke=0.1, w(t)=10*sin(t)

    Criterio: max|u_hat[10:] - u_true[10:]| < 0.1
    """
    print("=" * 70)
    print("Test 1 - Motor DC lineal (u aparece linealmente)")
    print("=" * 70)

    # Definir símbolos
    y, yp, ypp, u, t_s, w_s = symbols('y yp ypp u t w')

    # Parámetros del motor
    L0_val, R_val, Ke_val = 0.01, 1.0, 0.1

    # Residuo: L0*yp + R*y + Ke*w - u = 0
    F = L0_val * yp + R_val * y + Ke_val * w_s - u
    all_syms = (y, yp, ypp, u, t_s, w_s)

    # Construir regresor inverso
    inv_reg, info = build_inverse_regressor(F, all_syms, u)

    # Verificar linealidad
    assert info['u_is_linear'] == True, "u debe ser lineal en F"
    print("\n✓ Verificado: u es lineal en F (z2 = z3 = 0)\n")

    # Generar trayectoria de referencia con solve_ivp
    # Reducir T para mejorar precisión (sistema stiff con L0 pequeño)
    T = 5e-5
    n = 4000
    t = np.linspace(0, T*(n-1), n)

    # Señal externa w(t)
    w_array = 10 * np.sin(t)

    # Entrada verdadera: rampa suave en lugar de escalón
    # Usar transición suave: u(t) = 6*(1 + tanh((t-0.005)/0.001))
    u_true = 6.0 * (1.0 + np.tanh((t - 0.005) / 0.001))

    # Resolver y(t) dado u(t) usando solve_ivp
    def rhs(t_val, y_val):
        w_val = 10 * np.sin(t_val)
        u_val = 6.0 * (1.0 + np.tanh((t_val - 0.005) / 0.001))
        dydt = (u_val - R_val * y_val - Ke_val * w_val) / L0_val
        return [dydt]

    sol = solve_ivp(rhs, [t[0], t[-1]], [0.0], t_eval=t,
                     method='RK45', rtol=1e-9, atol=1e-12)
    y_true = sol.y[0]

    # Reconstruir u usando el regresor inverso
    u_hat = inv_reg(y_true, u_true[0], u_true[1], T, n, w_array)

    # Error (excluir transiente inicial)
    error = np.abs(u_hat[10:] - u_true[10:])
    max_error = np.max(error)

    print(f"Error máximo (excluyendo transiente): {max_error:.4e}")

    if max_error < 0.1:
        print("✓ PASS: Test 1 - Motor lineal\n")
        return True
    else:
        print("✗ FAIL: Test 1 - Motor lineal\n")
        return False


def test2_u_cuadratico():
    """
    Test 2 - u no lineal: F = yp + a*y - u² = 0

    Sistema: y' = u² - a*y con a=0.5

    Criterio: max|u_hat[5:] - u_true[5:]| < 1e-2
    """
    print("=" * 70)
    print("Test 2 - u cuadrático (u aparece como u²)")
    print("=" * 70)

    # Definir símbolos
    y, yp, ypp, u, t_s = symbols('y yp ypp u t')

    # Residuo: yp + 0.5*y - u² = 0
    a_val = 0.5
    F = yp + a_val * y - u**2
    all_syms = (y, yp, ypp, u, t_s)

    # Construir regresor inverso
    inv_reg, info = build_inverse_regressor(F, all_syms, u)

    # Verificar no linealidad
    assert info['u_is_linear'] == False, "u debe ser NO lineal en F"
    print("\n✓ Verificado: u es NO lineal en F (z2, z3 no nulos)\n")

    # Generar trayectoria de referencia
    T = 1e-3
    n = 1000
    t = np.linspace(0, T*(n-1), n)

    # Entrada verdadera
    u_true = 1.0 + 0.5 * np.sin(2*t)

    # Resolver y(t) dado u(t)
    def rhs(t_val, y_val):
        u_val = 1.0 + 0.5 * np.sin(2*t_val)
        dydt = u_val**2 - a_val * y_val
        return [dydt]

    sol = solve_ivp(rhs, [t[0], t[-1]], [0.0], t_eval=t,
                     method='RK45', rtol=1e-9, atol=1e-12)
    y_true = sol.y[0]

    # Reconstruir u (estimación inicial positiva para la raíz correcta)
    u_hat = inv_reg(y_true, u_true[0], u_true[1], T, n)

    # Error (excluir transiente inicial)
    error = np.abs(u_hat[5:] - u_true[5:])
    max_error = np.max(error)

    print(f"Error máximo (excluyendo transiente): {max_error:.4e}")

    if max_error < 1e-2:
        print("✓ PASS: Test 2 - u cuadrático\n")
        return True
    else:
        print("✗ FAIL: Test 2 - u cuadrático\n")
        return False


def test3_simetria_directa_inversa():
    """
    Test 3 - Simetría del Teorema 1

    Usando F = yp + 0.5*y - u = 0:
    1. Generar y_ref con regresor directo dado u_known
    2. Reconstruir u_hat con regresor inverso dado y_ref
    3. Verificar: max|u_hat[5:] - u_known[5:]| < 1e-4

    Criterio: Mismo F, misma precisión en ambas direcciones.
    """
    print("=" * 70)
    print("Test 3 - Simetría Teorema 1 (directo ↔ inverso)")
    print("=" * 70)

    # Definir símbolos
    y_sym = Symbol('y')
    y, yp, ypp, u, t_s = symbols('y yp ypp u t')

    # Sistema: yp + 0.5*y - u = 0  →  y' = -0.5*y + u
    a_val = 0.5
    f_expr = a_val * y_sym  # f(y) en y' + f(y) = u

    # Regresor directo: resuelve y dado u
    print("\n--- Construyendo regresor directo ---")
    reg_direct, info_direct = build_regressor_order1(f_expr, y_sym)

    # Regresor inverso: resuelve u dado y
    print("\n--- Construyendo regresor inverso ---")
    F = yp + a_val * y - u
    all_syms = (y, yp, ypp, u, t_s)
    reg_inverse, info_inverse = build_inverse_regressor(F, all_syms, u)

    # Entrada conocida
    T = 1e-3
    n = 1000
    t = np.linspace(0, T*(n-1), n)
    u_known = 1.0 + 0.3 * np.sin(3*t)

    # Paso 1: Generar y_ref con regresor directo
    y0 = 0.5
    # Calcular y1 con RK para tener condición inicial consistente
    from scipy.integrate import solve_ivp
    sol_ic = solve_ivp(lambda t_val, y_val: -a_val*y_val + (1.0 + 0.3*np.sin(3*t_val)),
                        [0, T], [y0], method='RK45', rtol=1e-12)
    y1 = sol_ic.y[0, -1]

    y_ref = reg_direct(u_known, y0, y1, T, n)

    # Paso 2: Reconstruir u_hat con regresor inverso
    u_hat = reg_inverse(y_ref, u_known[0], u_known[1], T, n)

    # Paso 3: Verificar simetría
    error = np.abs(u_hat[5:] - u_known[5:])
    max_error = np.max(error)

    print(f"\nError máximo (reconstrucción u): {max_error:.4e}")

    if max_error < 1e-4:
        print("✓ PASS: Test 3 - Simetría T1 verificada\n")
        return True
    else:
        print("✗ FAIL: Test 3 - Simetría T1\n")
        return False


def test4_verificar_z2_z3_cero():
    """
    Test 4 - Verificar que z2 y z3 son exactamente cero cuando u es lineal

    Para F = yp + y - u, verificar simbólicamente:
    - d²F/du² == 0
    - d³F/du³ == 0
    """
    print("=" * 70)
    print("Test 4 - Verificar z2 = z3 = 0 para u lineal")
    print("=" * 70)

    # Definir símbolos
    y, yp, ypp, u, t_s = symbols('y yp ypp u t')

    # Residuo lineal en u: yp + y - u = 0
    F = yp + y - u
    all_syms = (y, yp, ypp, u, t_s)

    # Construir regresor inverso
    inv_reg, info = build_inverse_regressor(F, all_syms, u)

    # Verificar simbólicamente
    from sympy import S

    print("\nVerificando expresiones simbólicas:")
    print(f"  d²F/du² = {info['d2F_u2']}")
    print(f"  d³F/du³ = {info['d3F_u3']}")

    is_zero_d2 = info['d2F_u2'] == S.Zero or info['d2F_u2'] == 0
    is_zero_d3 = info['d3F_u3'] == S.Zero or info['d3F_u3'] == 0
    is_linear  = info['u_is_linear']

    if is_zero_d2 and is_zero_d3 and is_linear:
        print("\n✓ PASS: Test 4 - z2 = z3 = 0 verificado simbólicamente\n")
        return True
    else:
        print("\n✗ FAIL: Test 4 - z2 y z3 no son cero\n")
        return False


if __name__ == "__main__":
    print("\n" + "="*70)
    print(" VALIDACIÓN DEL REGRESOR HOMOTÓPICO INVERSO")
    print(" Teorema 1 - Paper 10")
    print("="*70 + "\n")

    results = []

    # Ejecutar tests
    results.append(("Test 1 - Motor lineal", test1_motor_lineal()))
    results.append(("Test 2 - u cuadrático", test2_u_cuadratico()))
    results.append(("Test 3 - Simetría T1", test3_simetria_directa_inversa()))
    results.append(("Test 4 - z2=z3=0 lineal", test4_verificar_z2_z3_cero()))

    # Resumen final
    print("\n" + "="*70)
    print(" RESUMEN FINAL")
    print("="*70)

    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:10s} - {name}")

    all_passed = all(r[1] for r in results)

    print("="*70)
    if all_passed:
        print("\n✓ TODOS LOS TESTS PASARON\n")
    else:
        print("\n✗ ALGUNOS TESTS FALLARON\n")

    print("="*70 + "\n")
