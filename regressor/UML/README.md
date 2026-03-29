# UML del proyecto

Esta carpeta contiene diagramas fuente en PlantUML para documentar arquitectura y flujos del HAM regressor.

## Archivos

- `architecture_components.puml`: vista de componentes y dependencias.
- `sequence_scalar_solver.puml`: secuencia de construccion y ejecucion del regresor escalar.
- `activity_parameter_identification.puml`: flujo de identificacion LIP/no-LIP.

## Renderizado rapido

Si tenes `plantuml` instalado:

```bash
cd UML
plantuml *.puml
```

Esto genera PNG/SVG segun configuracion local.
