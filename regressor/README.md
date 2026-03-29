# Homotopy Regressors (HAM) - Proyecto UNSJ

Implementacion de regresores homotopicos discretos para resolver e identificar EDOs no lineales con esquema backward de 3 puntos.

Autor: Rodolfo H. Rodrigo (UNSJ)

## Objetivo

Este proyecto implementa:

- Solucion de EDOs escalares de 1er y 2do orden con correcciones homotopicas `z1`, `z2`, `z3`.
- Extension a sistemas acoplados (N=2 y N=3) con Jacobiano/Hessiano/Tensor simbolicos.
- Verificacion contra `RK45` (`scipy.integrate.solve_ivp`).
- Identificacion de parametros (LIP y no-LIP) con regularizacion tipo Tikhonov.
- Regresor inverso para reconstruir entrada `u(t)` a partir de `y(t)`.

## Estructura principal

- `solver.py`: solver escalar numerico (orden 1/2 + version con derivadas numericas).
- `regressor.py`: construccion simbolica de regressors escalares + inverso.
- `solver_system.py`: solver vectorial (N=2/3) con 3 correcciones.
- `regressor_system.py`: generacion simbolica de Jacobiano/Hessiano/Tensor para sistemas.
- `verify_regressor.py`: comparacion unificada HAM vs RK45.
- `identify_parameters.py`: identificacion de parametros y construccion de matriz `Phi`.
- `parser.py`: parser de ODE en texto hacia forma estandar.
- `derivatives.py`: utilidades de derivadas discretas/Taylor.
- `examples.py`: ejemplos escalares.
- `test_*.py`: suites de validacion y comparacion.
- `UML/`: diagramas PlantUML del proyecto.

## Formulacion discreta base

Para una variable `q[k]`:

- `q'[k] = (3*q[k] - 4*q[k-1] + q[k-2]) / (2*T)`
- `q''[k] = (q[k] - 2*q[k-1] + q[k-2]) / T**2`

La evolucion por paso aplica:

1. `z1`: correccion tipo Newton.
2. `z2`: correccion de curvatura (Hessiano).
3. `z3`: correccion de tercer orden (Tensor), cuando corresponde.

## Requisitos

- Python 3.8+
- `numpy`
- `scipy`
- `sympy`

Instalacion local:

```bash
python -m venv venv
source venv/bin/activate
pip install -e .
```

## Uso rapido

### 1) Regresor escalar de 1er orden

```python
import numpy as np
from sympy import Symbol
from regressor import build_regressor_order1

y = Symbol("y")
reg, info = build_regressor_order1(y**2, y)

n = 1000
T = 0.01
t = np.linspace(0, T*(n-1), n)
u = np.sin(5*t)

sol = reg(u, y0=0.1, y1=0.1, T=T, n=n)
```

### 2) Verificacion HAM vs RK45

```python
import numpy as np
from verify_regressor import verify_regressor_vs_rk45, print_report

def rhs(t, y):
    return [-(y[0]**2) + np.sin(5*t)]

def reg_wrapper(sol):
    T = sol.t[1] - sol.t[0]
    return reg(np.sin(5*sol.t), sol.y[0, 0], sol.y[0, 1], T, len(sol.t))

result = verify_regressor_vs_rk45(
    rhs=rhs,
    ic=[0.1],
    t_span=(0.0, 10.0),
    n=1000,
    regressor_callable=reg_wrapper,
    threshold=1e-1,
    label="Test escalar",
)
print_report(result)
```

### 3) Identificacion de parametros

Ver `identify_parameters.py` (funciones `check_lip`, `identify_lip`, `identify_nonlip`, `build_parametric_regressor`) y los contratos en:

- `CLAUDE_identify_parameters.md`
- `CLAUDE_tikhonov_identification.md`

## Ejecucion de pruebas

```bash
python test_regressor_vs_rk4.py
python test_step_size_effect.py
python test_inverse_regressor.py
```

## Documentacion existente

- `INDICE_DOCUMENTACION.txt`
- `REPORTE_FINAL.md`
- `RESUMEN_FINAL.txt`
- `CLAUDE.md`
- `CLAUDE_inverse_regressor.md`
- `CLAUDE_verify_regressor.md`
- `CLAUDE_identify_parameters.md`
- `CLAUDE_tikhonov_identification.md`

## UML

Se creo la carpeta `UML/` con diagramas editables en PlantUML:

- `UML/architecture_components.puml`
- `UML/sequence_scalar_solver.puml`
- `UML/activity_parameter_identification.puml`

Instrucciones de generacion en `UML/README.md`.
