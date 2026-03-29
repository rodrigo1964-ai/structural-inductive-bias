"""
operators.py — Operadores lineales auxiliares L para el HAM continuo

El operador L define la clase de ecuaciones lineales que se resuelven
en cada paso del HAM. Una buena eleccion de L acelera la convergencia.

Regla de Liao: L debe ser invertible y del mismo orden que N.

Author: Rodolfo H. Rodrigo — UNSJ / INAUT
"""

from sympy import diff, Symbol, Function, Derivative


def L_derivative(expr, t_sym):
    """
    L[u] = u'  (derivada primera)

    Uso tipico: EDOs de primer orden.
    Inversion: u_m = integral de RHS.
    """
    return diff(expr, t_sym)


def L_second(expr, t_sym):
    """
    L[u] = u''  (derivada segunda)

    Uso tipico: EDOs de segundo orden sin amortiguamiento.
    Inversion: doble integracion.
    """
    return diff(expr, t_sym, 2)


def L_damped(expr, t_sym, alpha=1.0):
    """
    L[u] = u'' + alpha * u'

    Uso tipico: Osciladores amortiguados.
    Las soluciones de L[u] = 0 son {1, exp(-alpha*t)}.

    Parameters
    ----------
    alpha : float
        Coeficiente de amortiguamiento (> 0).
    """
    from sympy import S
    return diff(expr, t_sym, 2) + S(alpha) * diff(expr, t_sym)


def L_harmonic(expr, t_sym, omega=1.0):
    """
    L[u] = u'' + omega^2 * u

    Uso tipico: Osciladores no lineales.
    Las soluciones de L[u] = 0 son {cos(omega*t), sin(omega*t)}.

    Ventaja: las funciones base son periodicas, lo que es natural
    para sistemas oscilatorios (Rule of Solution Expression de Liao).

    Parameters
    ----------
    omega : float
        Frecuencia natural del operador auxiliar.
    """
    from sympy import S
    return diff(expr, t_sym, 2) + S(omega)**2 * expr


def L_exponential(expr, t_sym, mu=1.0):
    """
    L[u] = u' + mu * u

    Uso tipico: EDOs de primer orden con decaimiento.
    Las soluciones de L[u] = 0 son {exp(-mu*t)}.

    Ventaja: si la solucion real decae exponencialmente,
    esta L produce convergencia mas rapida.

    Parameters
    ----------
    mu : float
        Tasa de decaimiento (> 0).
    """
    from sympy import S
    return diff(expr, t_sym) + S(mu) * expr


def L_custom(coeffs):
    """
    Construye un operador L customizado a partir de coeficientes.

    L[u] = c_n * u^(n) + c_{n-1} * u^{(n-1)} + ... + c_1 * u' + c_0 * u

    Parameters
    ----------
    coeffs : list of float
        [c_0, c_1, ..., c_n] coeficientes desde orden 0 hasta orden n.
        Ejemplo: [1, 0, 1] -> L[u] = u + u'' (armonico con omega=1)

    Returns
    -------
    L : callable(expr, t_sym) -> sympy.Expr
    """
    from sympy import S

    def L(expr, t_sym):
        result = S.Zero
        for k, c in enumerate(coeffs):
            if c != 0:
                if k == 0:
                    result += S(c) * expr
                else:
                    result += S(c) * diff(expr, t_sym, k)
        return result

    return L


def describe_operator(L_operator, t_sym):
    """
    Muestra una descripcion del operador L aplicado a un simbolo generico.

    Parameters
    ----------
    L_operator : callable
    t_sym : Symbol

    Returns
    -------
    str : descripcion
    """
    u = Function('u')(t_sym)
    try:
        result = L_operator(u, t_sym)
        return f"L[u] = {result}"
    except Exception as e:
        return f"L[u] = (error: {e})"
