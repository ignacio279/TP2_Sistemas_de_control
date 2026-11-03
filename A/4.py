"""Parte A, ejercicio 4: promedio móvil causal.
A4(a): N = VENTANA_S / DT. A4(b): completar promedio_movil aquí.
A4(c): la justificación del filtrado se desarrolla en el informe.
Ejecutar desde la raíz del repo: python A/4.py
"""
from pathlib import Path
import sys

RAIZ = Path(__file__).resolve().parent.parent

# La configuración se toma del archivo de la cátedra, sin duplicarla.
sys.path.insert(0, str(RAIZ / 'consigna'))
from script_identificacion import (DT, VENTANA_S, N)




def promedio_movil(velocidad):
    # Completar promedio_movil segun el enunciado.
    raise NotImplementedError('Completar promedio_movil.')


if __name__ == '__main__':
    print('DT =', DT, 's; ventana =', VENTANA_S, 's; N =', N, 'muestras.')
    print('Pendiente: completar promedio_movil en A/4.py (ejercicio 4b).')
