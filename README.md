# TP2 - Sistemas de Control

## Entorno de ejecucion

Se utilizo Python 3.11.8. Las versiones de las librerias estan en `requirements.txt`.
Desde la carpeta del TP, crear un entorno e instalar las dependencias:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -E -m pip install -r requirements.txt
```

En Windows, activar el entorno con `.venv\Scripts\activate` en lugar de `source .venv/bin/activate`.

## Ejecutar el TP completo

```bash
python -E ejecutar_todo.py
```

El script ejecuta A1 a A6 y despues B1 a B6 con el mismo Python. A2 contiene las
expresiones desarrolladas en el informe y no genera archivos. A5 guarda los
parametros identificados antes de ejecutar los puntos que los necesitan.

La opcion `-E` evita que variables como `PYTHONPATH` carguen librerias de otro
entorno. El script tambien la utiliza al ejecutar cada ejercicio.

Las figuras y los resultados quedan en `A/resultados/1`, `A/resultados/3`, etc,
y `B/resultados/1` a `B/resultados/6`. Las versiones utilizadas y la salida de
consola quedan en `ejecucion_completa.txt`. Las figuras se guardan sin abrir
ventanas. Cada ejecucion reemplaza los resultados y el registro anterior.

Si un ejercicio falla, el script se detiene. El detalle del error queda al final
del registro. Para correr un punto por separado se puede usar, por ejemplo,
`python -E A/5.py`. Antes de A6 o cualquier punto de B hay que ejecutar A5.

Se debe conservar la carpeta `consigna`, que contiene `datos_motor.csv` y
`script_identificacion.py`, junto con las carpetas A y B.
