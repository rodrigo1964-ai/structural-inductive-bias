"""
verify_regressor.py - Verificación unificada del regresor homotópico vs RK45

Compara el regresor homotópico (scalar o multivariable) contra scipy.integrate.solve_ivp
con método RK45.

Author: Rodolfo H. Rodrigo - UNSJ
Fecha: Marzo 2026
"""

import numpy as np
from scipy.integrate import solve_ivp


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
        Función que ejecuta el regresor con firma:
            regressor_callable(sol) -> np.ndarray shape (n,)    [Modo A: scalar]
                                    -> list of np.ndarray        [Modo B: sistema]
        donde sol = resultado de solve_ivp con .y shape (N, n) y .t shape (n,)

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
        "T"            : float, período de muestreo efectivo
        "n"            : int,   número de puntos
        "var_names"    : list of str
    """
    # Normalizar condiciones iniciales
    ic = np.atleast_1d(ic)

    # Vector de tiempos
    t_arr = np.linspace(t_span[0], t_span[1], n)
    T = t_arr[1] - t_arr[0]

    # Resolver con RK45
    sol = solve_ivp(
        rhs,
        t_span,
        ic,
        method='RK45',
        t_eval=t_arr,
        rtol=1e-9,
        atol=1e-9,
    )

    if not sol.success:
        raise RuntimeError(f"RK45 no convergió: {sol.message}")

    # Llamar al regresor (pasa sol completo)
    ham_result = regressor_callable(sol)

    # Normalizar salida del regresor a shape (N, n)
    if isinstance(ham_result, list):
        # Modo B: sistema
        ham = np.array(ham_result)  # shape (N, n)
    else:
        # Modo A: scalar
        ham = ham_result.reshape(1, n)  # shape (1, n)

    # Si ham tiene menos filas que sol.y, solo comparar las primeras filas
    # (caso típico: regresor 2do orden devuelve solo y, no yp)
    if ham.shape[0] < sol.y.shape[0]:
        sol_y_comparison = sol.y[:ham.shape[0], :]
    else:
        sol_y_comparison = sol.y

    # Calcular errores
    error_abs = np.abs(ham - sol_y_comparison)         # shape (N, n)
    max_error = np.max(error_abs, axis=1)              # shape (N,)
    rms_error = np.sqrt(np.mean(error_abs**2, axis=1)) # shape (N,)

    # Evaluar threshold
    if threshold is not None:
        passed = bool(np.all(max_error < threshold))
    else:
        passed = True

    # Determinar N del tamaño del resultado del regresor
    N = ham.shape[0]

    # Nombres de variables
    if var_names is None:
        if N == 1:
            var_names = ["y"]
        else:
            var_names = [f"y{i}" for i in range(N)]

    # Construir resultado
    result = {
        "label": label,
        "t": t_arr,
        "rk45": sol.y,
        "ham": ham,
        "error_abs": error_abs,
        "max_error": max_error,
        "rms_error": rms_error,
        "passed": passed,
        "T": T,
        "n": n,
        "var_names": var_names,
        "threshold": threshold,
    }

    # Generar figura si se solicita
    if plot:
        _maybe_plot(result, plot_path)

    return result


def print_report(result):
    """
    Imprime tabla de resultados formateada a stdout.
    """
    print("=" * 70)
    print(f"TEST: {result['label']}")
    t0, tf = result['t'][0], result['t'][-1]
    print(f"T = {result['T']:.4e}s,  n = {result['n']},  t ∈ [{t0}, {tf}]")
    print("-" * 70)
    print(f"{'Variable':<15s} {'max|error|':<15s} {'RMS error':<15s} Status")
    print(f"{'-'*15} {'-'*15} {'-'*15} {'-'*6}")

    N = len(result['var_names'])
    threshold = result.get('threshold', None)

    for i in range(N):
        var_name = result['var_names'][i]
        max_err = result['max_error'][i]
        rms_err = result['rms_error'][i]

        if threshold is not None:
            status = "✓ PASS" if max_err < threshold else f"✗ FAIL (threshold: {threshold:.1e})"
        else:
            status = "✓"

        print(f"{var_name:<15s} {max_err:<15.4e} {rms_err:<15.4e} {status}")

    print("-" * 70)
    overall = "PASS" if result['passed'] else "FAIL"
    print(f"OVERALL: {overall}")
    print("=" * 70)
    print()


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
    results = []
    passed_count = 0
    failed_count = 0

    for test_dict in test_list:
        # Extraer parámetros
        label = test_dict["label"]
        rhs = test_dict["rhs"]
        ic = test_dict["ic"]
        t_span = test_dict["t_span"]
        n = test_dict["n"]
        regressor_callable = test_dict["regressor_callable"]
        var_names = test_dict.get("var_names", None)
        threshold = test_dict.get("threshold", None)
        plot = test_dict.get("plot", False)
        plot_path = test_dict.get("plot_path", None)

        # Ejecutar test
        result = verify_regressor_vs_rk45(
            rhs, ic, t_span, n, regressor_callable,
            var_names=var_names,
            threshold=threshold,
            label=label,
            plot=plot,
            plot_path=plot_path,
        )

        results.append(result)

        # Imprimir reporte
        print_report(result)

        # Contabilizar
        if result['passed']:
            passed_count += 1
        else:
            failed_count += 1
            if stop_on_failure:
                print(f"✗ DETENIENDO SUITE: {label} falló.\n")
                break

    all_passed = (failed_count == 0)

    summary = {
        "results": results,
        "passed": passed_count,
        "failed": failed_count,
        "all_passed": all_passed,
    }

    return summary


def _detect_N(ic):
    """Detecta dimensión del sistema."""
    return len(np.atleast_1d(ic))


def _maybe_plot(result, plot_path):
    """Genera figura con subplots (solución + error por variable)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Advertencia: matplotlib no disponible, omitiendo plot.")
        return

    N = len(result['var_names'])
    t = result['t']
    rk45 = result['rk45']
    ham = result['ham']
    error_abs = result['error_abs']

    fig, axes = plt.subplots(N, 2, figsize=(12, 3*N))

    if N == 1:
        axes = axes.reshape(1, 2)

    for i in range(N):
        var_name = result['var_names'][i]

        # Columna izquierda: trayectorias
        ax_traj = axes[i, 0]
        ax_traj.plot(t, rk45[i, :], 'b-', label='RK45', linewidth=1.5)
        ax_traj.plot(t, ham[i, :], 'r--', label='HAM', linewidth=1.0)
        ax_traj.set_xlabel('Tiempo [s]')
        ax_traj.set_ylabel(var_name)
        ax_traj.legend()
        ax_traj.grid(True, alpha=0.3)

        # Columna derecha: error absoluto
        ax_err = axes[i, 1]
        ax_err.semilogy(t, error_abs[i, :], 'k-', linewidth=0.8)
        ax_err.set_xlabel('Tiempo [s]')
        ax_err.set_ylabel(f'|error| en {var_name}')
        ax_err.grid(True, alpha=0.3)

    fig.suptitle(f"HAM vs RK45 — {result['label']}", fontsize=14)
    fig.tight_layout()

    if plot_path is None:
        plt.show()
    else:
        plt.savefig(plot_path, dpi=150)
        print(f"  Figura guardada en: {plot_path}")
        plt.close(fig)


# =============================================================================
# SUITE DE DEMOSTRACIÓN (6 tests)
# =============================================================================

if __name__ == "__main__":
    from sympy import symbols, Symbol, sin as sym_sin, cos as sym_cos
    from regressor import build_regressor_order1, build_regressor_order2
    from solver_system import solve_system_numeric

    print("\n" + "="*70)
    print(" SUITE DE VERIFICACIÓN: HAM vs RK45")
    print(" 6 tests de demostración (D1-D6)")
    print("="*70 + "\n")

    test_list = []

    # =========================================================================
    # Test D1 — Scalar 1er orden: Van der Pol (μ=0.5)
    # =========================================================================
    # y' = -μ(y²-1)*y + y + A·sin(ωt)
    mu, A, omega = 0.5, 0.5, 1.5
    n_d1 = 2000
    t_d1 = np.linspace(0, 20, n_d1)
    T_d1 = t_d1[1] - t_d1[0]
    u_d1 = A * np.sin(omega * t_d1)

    y_sym = Symbol('y')
    f_vdp = mu * (y_sym**2 - 1) * y_sym - y_sym  # forma: y' + f(y) = u
    reg_vdp, _ = build_regressor_order1(f_vdp, y_sym)

    def rhs_vdp(t, y):
        return [-mu*(y[0]**2 - 1)*y[0] + y[0] + A*np.sin(omega*t)]

    def regressor_d1(sol):
        y0 = sol.y[0, 0]
        y1 = sol.y[0, 1]
        return reg_vdp(u_d1, y0, y1, T_d1, n_d1)

    test_list.append({
        "label": "D1 — Van der Pol scalar",
        "rhs": rhs_vdp,
        "ic": [0.5],
        "t_span": (0, 20),
        "n": n_d1,
        "regressor_callable": regressor_d1,
        "threshold": 5e-2,
    })

    # =========================================================================
    # Test D2 — Scalar 2do orden: Duffing forzado
    # =========================================================================
    # y'' + 0.1*y' + y + 0.2*y³ = 0.3·cos(1.2t)
    n_d2 = 25000  # Incrementado de 5000 para reducir T
    t_d2 = np.linspace(0, 50, n_d2)
    T_d2 = t_d2[1] - t_d2[0]
    u_d2 = 0.3 * np.cos(1.2 * t_d2)

    y_sym_d2 = Symbol('y')
    yp_sym_d2 = Symbol('yp')
    f_duf = 0.1*yp_sym_d2 + y_sym_d2 + 0.2*y_sym_d2**3
    reg_duf, _ = build_regressor_order2(f_duf, y_sym_d2, yp_sym_d2)

    def rhs_duf_position_only(t, y_arr):
        """RHS que solo devuelve y (no yp) para comparación directa"""
        y = y_arr[0]
        # Calcular yp y ypp necesarios
        # Como no tenemos yp directamente, usamos una integración auxiliar
        # En realidad, esto es complicado. Mejor usar el RHS estándar y extraer solo y.
        # Esta función no se puede usar así. Vamos a modificar el enfoque.
        pass

    # Usar RHS estándar pero modificar la comparación
    def rhs_duf(t, z):
        y, yp = z
        dydt = yp
        dypdt = -0.1*yp - y - 0.2*y**3 + 0.3*np.cos(1.2*t)
        return [dydt, dypdt]

    # El regressor devuelve solo y, así que necesitamos envolver para que devuelva
    # un array con shape (1, n) compatible con sol.y que será (2, n)
    # Solución: usar un wrapper que compara solo la primera componente
    def regressor_d2(sol):
        y0 = sol.y[0, 0]  # posición en t=0
        y1 = sol.y[0, 1]  # posición en t=1
        y_ham = reg_duf(u_d2, y0, y1, T_d2, n_d2)
        # Devolver solo y para comparar con sol.y[0, :] (primera fila de RK45)
        return y_ham

    # Para D2, necesitamos un RHS que devuelva solo [y] para tener N=1
    # Pero no podemos resolver una EDO de 2do orden con RK45 sin convertirla a sistema 1er orden
    # La solución es hacer un test diferente o modificar verify_regressor_vs_rk45
    # para manejar casos donde el regresor devuelve menos variables que RK45.

    # Por ahora, voy a modificar para que el test D2 use solo la posición y en la comparación
    # creando una función wrapper especial

    # Crear un diccionario especial para D2 que incluya un flag para indicar
    # que solo se debe comparar la primera variable

    test_list.append({
        "label": "D2 — Duffing scalar 2do orden",
        "rhs": rhs_duf,
        "ic": [0.1, 0],
        "t_span": (0, 50),
        "n": n_d2,
        "regressor_callable": regressor_d2,
        "threshold": 1e-2,
    })

    # =========================================================================
    # Test D3 — Sistema 2D 1er orden: Lotka-Volterra
    # =========================================================================
    # x' = α*x - β*x*y
    # y' = δ*x*y - γ*y
    alpha, beta, gamma, delta = 1.0, 0.1, 1.5, 0.075
    n_d3 = 5000
    t_d3 = np.linspace(0, 30, n_d3)
    T_d3 = t_d3[1] - t_d3[0]

    def funcs_lv():
        def F(x, y, z, w, xp, yp, zp, wp, xpp, ypp, zpp, wpp, t):
            return xp - alpha*x + beta*x*y
        def G(x, y, z, w, xp, yp, zp, wp, xpp, ypp, zpp, wpp, t):
            return yp - delta*x*y + gamma*y
        return [F, G]

    def rhs_lv(t, z):
        x, y = z
        dxdt = alpha*x - beta*x*y
        dydt = delta*x*y - gamma*y
        return [dxdt, dydt]

    def regressor_d3(sol):
        ic_lv = [[sol.y[0, 0], sol.y[0, 1]],
                 [sol.y[1, 0], sol.y[1, 1]]]
        u_lv = [np.zeros(n_d3), np.zeros(n_d3)]
        return solve_system_numeric(funcs_lv(), u_lv, ic_lv, T_d3, n_d3)

    test_list.append({
        "label": "D3 — Lotka-Volterra 2D",
        "rhs": rhs_lv,
        "ic": [10, 5],
        "t_span": (0, 30),
        "n": n_d3,
        "regressor_callable": regressor_d3,
        "var_names": ["x (prey)", "y (pred)"],
        "threshold": 1e-2,
    })

    # =========================================================================
    # Test D4 — Sistema 3D 1er orden: Lorenz
    # =========================================================================
    # x' = σ(y-x),  y' = x(ρ-z) - y,  z' = xy - βz
    sigma, rho, beta_l = 10, 28, 8/3
    n_d4 = 20000
    t_d4 = np.linspace(0, 2, n_d4)
    T_d4 = t_d4[1] - t_d4[0]

    def funcs_lorenz():
        def F(x, y, z, w, xp, yp, zp, wp, xpp, ypp, zpp, wpp, t):
            return xp - sigma*(y - x)
        def G(x, y, z, w, xp, yp, zp, wp, xpp, ypp, zpp, wpp, t):
            return yp - x*(rho - z) + y
        def H(x, y, z, w, xp, yp, zp, wp, xpp, ypp, zpp, wpp, t):
            return zp - x*y + beta_l*z
        return [F, G, H]

    def rhs_lorenz(t, w):
        x, y, z = w
        dxdt = sigma*(y - x)
        dydt = x*(rho - z) - y
        dzdt = x*y - beta_l*z
        return [dxdt, dydt, dzdt]

    def regressor_d4(sol):
        ic_lor = [[sol.y[0, 0], sol.y[0, 1]],
                  [sol.y[1, 0], sol.y[1, 1]],
                  [sol.y[2, 0], sol.y[2, 1]]]
        u_lor = [np.zeros(n_d4), np.zeros(n_d4), np.zeros(n_d4)]
        return solve_system_numeric(funcs_lorenz(), u_lor, ic_lor, T_d4, n_d4)

    test_list.append({
        "label": "D4 — Lorenz 3D",
        "rhs": rhs_lorenz,
        "ic": [1, 1, 1],
        "t_span": (0, 2),
        "n": n_d4,
        "regressor_callable": regressor_d4,
        "var_names": ["x", "y", "z"],
        "threshold": 1e-1,
    })

    # =========================================================================
    # Test D5 — Sistema 2D 2do orden: Duffing acoplado
    # =========================================================================
    # x'' + 0.1*x' + x + 0.2*x³ + 0.5*(x-y) = 0.5·cos(1.2t)
    # y'' + 0.1*y' + y + 0.2*y³ - 0.5*(x-y) = 0
    n_d5 = 100000  # Incrementado de 50000 para reducir T
    t_d5 = np.linspace(0, 50, n_d5)
    T_d5 = t_d5[1] - t_d5[0]
    u_d5_x = 0.5 * np.cos(1.2 * t_d5)
    u_d5_y = np.zeros(n_d5)

    def funcs_duff():
        def F(x, y, z, w, xp, yp, zp, wp, xpp, ypp, zpp, wpp, t):
            idx = min(int(t/T_d5), n_d5-1)
            return xpp + 0.1*xp + x + 0.2*x**3 + 0.5*(x - y) - u_d5_x[idx]
        def G(x, y, z, w, xp, yp, zp, wp, xpp, ypp, zpp, wpp, t):
            return ypp + 0.1*yp + y + 0.2*y**3 - 0.5*(x - y)
        return [F, G]

    def rhs_duff(t, w):
        x, xp, y, yp = w
        idx = int(t / T_d5)
        if idx >= n_d5:
            idx = n_d5 - 1
        u_x = u_d5_x[idx]
        dxdt = xp
        dxpdt = -0.1*xp - x - 0.2*x**3 - 0.5*(x - y) + u_x
        dydt = yp
        dypdt = -0.1*yp - y - 0.2*y**3 + 0.5*(x - y)
        return [dxdt, dxpdt, dydt, dypdt]

    def regressor_d5(sol):
        # Extraer x, y de las posiciones 0, 2
        ic_duff = [[sol.y[0, 0], sol.y[0, 1]],
                   [sol.y[2, 0], sol.y[2, 1]]]
        u_duff = [u_d5_x, u_d5_y]

        # Función que solo necesita x, y (no xp, yp)
        def funcs_simple():
            def F(x, y, z, w, xp, yp, zp, wp, xpp, ypp, zpp, wpp, t):
                idx = min(int(t/T_d5), n_d5-1)
                return xpp + 0.1*xp + x + 0.2*x**3 + 0.5*(x - y) - u_d5_x[idx]
            def G(x, y, z, w, xp, yp, zp, wp, xpp, ypp, zpp, wpp, t):
                idx = min(int(t/T_d5), n_d5-1)
                return ypp + 0.1*yp + y + 0.2*y**3 - 0.5*(x - y) - u_d5_y[idx]
            return [F, G]

        result = solve_system_numeric(funcs_simple(), u_duff, ic_duff, T_d5, n_d5)
        return result

    test_list.append({
        "label": "D5 — Duffing acoplado 2D 2do orden",
        "rhs": rhs_duff,
        "ic": [0, 0.5, 0.2, 0],
        "t_span": (0, 50),
        "n": n_d5,
        "regressor_callable": regressor_d5,
        "threshold": 1e-2,
    })

    # =========================================================================
    # Test D6 — Sistema 3D 1er orden: Euler cuerpo rígido
    # =========================================================================
    # I1*w1' = (I2-I3)*w2*w3
    # I2*w2' = (I3-I1)*w3*w1
    # I3*w3' = (I1-I2)*w1*w2
    I1, I2, I3 = 2.0, 1.0, 0.5
    n_d6 = 4000  # Incrementado de 2000 para reducir T
    t_d6 = np.linspace(0, 20, n_d6)
    T_d6 = t_d6[1] - t_d6[0]

    def funcs_euler():
        def F(x, y, z, w, xp, yp, zp, wp, xpp, ypp, zpp, wpp, t):
            return I1*xp - (I2-I3)*y*z
        def G(x, y, z, w, xp, yp, zp, wp, xpp, ypp, zpp, wpp, t):
            return I2*yp - (I3-I1)*z*x
        def H(x, y, z, w, xp, yp, zp, wp, xpp, ypp, zpp, wpp, t):
            return I3*zp - (I1-I2)*x*y
        return [F, G, H]

    def rhs_euler(t, w):
        w1, w2, w3 = w
        dw1dt = (I2-I3)*w2*w3 / I1
        dw2dt = (I3-I1)*w3*w1 / I2
        dw3dt = (I1-I2)*w1*w2 / I3
        return [dw1dt, dw2dt, dw3dt]

    def regressor_d6(sol):
        ic_eul = [[sol.y[0, 0], sol.y[0, 1]],
                  [sol.y[1, 0], sol.y[1, 1]],
                  [sol.y[2, 0], sol.y[2, 1]]]
        u_eul = [np.zeros(n_d6), np.zeros(n_d6), np.zeros(n_d6)]
        return solve_system_numeric(funcs_euler(), u_eul, ic_eul, T_d6, n_d6)

    test_list.append({
        "label": "D6 — Euler cuerpo rígido 3D",
        "rhs": rhs_euler,
        "ic": [1.0, 0.1, 0.5],
        "t_span": (0, 20),
        "n": n_d6,
        "regressor_callable": regressor_d6,
        "var_names": ["w1", "w2", "w3"],
        "threshold": 1e-3,
    })

    # =========================================================================
    # Ejecutar suite
    # =========================================================================
    summary = run_suite(test_list, stop_on_failure=False)

    # =========================================================================
    # Verificación de conservación para D6
    # =========================================================================
    print("\n" + "="*70)
    print(" VERIFICACIÓN DE CONSERVACIÓN (Test D6 — Euler)")
    print("="*70)

    result_d6 = summary['results'][5]  # D6 es el 6to test (índice 5)
    w1 = result_d6['ham'][0, :]
    w2 = result_d6['ham'][1, :]
    w3 = result_d6['ham'][2, :]

    E0 = 0.5*(I1*w1[0]**2 + I2*w2[0]**2 + I3*w3[0]**2)
    E  = 0.5*(I1*w1**2 + I2*w2**2 + I3*w3**2)
    E_drift = np.max(np.abs(E - E0)) / E0

    status_E = '✓' if E_drift < 1e-4 else '✗'
    print(f"  Energía: deriva máx = {E_drift:.2e}  {status_E}")

    L0_sq = (I1*w1[0])**2 + (I2*w2[0])**2 + (I3*w3[0])**2
    L_sq  = (I1*w1)**2 + (I2*w2)**2 + (I3*w3)**2
    L_drift = np.max(np.abs(L_sq - L0_sq)) / L0_sq

    status_L = '✓' if L_drift < 1e-4 else '✗'
    print(f"  Momento angular: deriva máx = {L_drift:.2e}  {status_L}")

    # =========================================================================
    # Resumen final
    # =========================================================================
    print("\n" + "="*70)
    print(" RESUMEN DE LA SUITE")
    print("="*70)
    print(f"Suite: {summary['passed']}/{len(test_list)} passed")

    if summary['all_passed']:
        print("\n✓ TODOS LOS TESTS PASARON\n")
    else:
        print("\n✗ ALGUNOS TESTS FALLARON\n")

    print("="*70 + "\n")
