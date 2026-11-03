"""Parte A, ejercicio 5: identificación NLS y parámetros del modelo.
Reutiliza Euler de A/3.py y el filtro de A/4.py.
Requiere completar primero promedio_movil; no se agregó una solución.
Se conserva el optimizador que ya estaba en script_identificacion.py.
Ejecutar desde la raíz del repo: python A/5.py [datos_motor.csv]
A5(a) y la deducción de A5(c) se explican en el informe.
"""
from pathlib import Path
import sys
import json
from importlib import import_module
import numpy as np
from scipy.optimize import least_squares

RAIZ = Path(__file__).resolve().parent.parent
CARPETA_RESULTADOS = Path(__file__).resolve().parent / 'resultados'

# La configuración se toma del archivo de la cátedra, sin duplicarla.
sys.path.insert(0, str(RAIZ / 'consigna'))
from script_identificacion import (DT, VENTANA_S, DURACION_CICLO_S, CICLOS_AJUSTE, CICLOS_VALIDACION, DUTY_REFERENCIA, TIEMPO_INICIAL_DESCARTADO_S, N)


def archivo_datos():
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    return RAIZ / 'consigna' / 'datos_motor.csv'


def cargar_ejercicio(numero):
    # Solo carga funciones: el bloque principal del ejercicio no se ejecuta.
    return import_module(str(numero))


def carpeta_resultados(numero):
    carpeta = CARPETA_RESULTADOS / str(numero)
    carpeta.mkdir(parents=True, exist_ok=True)
    return carpeta


def guardar_resultados(numero, nombre, resultados):
    destino = carpeta_resultados(numero) / nombre
    with destino.open('w', encoding='utf-8') as salida:
        json.dump(resultados, salida, indent=2, ensure_ascii=False)
    print(json.dumps(resultados, indent=2, ensure_ascii=False))
    print('Resultados en:', destino)
    return destino


def ejercicio_5():
    calcular_velocidad = cargar_ejercicio(3).calcular_velocidad
    promedio_movil = cargar_ejercicio(4).promedio_movil
    # LEER LAS TRES COLUMNAS. La velocidad medida no se vuelve a filtrar.
    archivo = archivo_datos()
    datos   = np.genfromtxt(archivo, delimiter=',', names=True, encoding='utf-8-sig')
    tiempo  = datos['time_s']
    duty    = datos['duty_effective']
    velocidad_medida = datos['omega_prom_measured']
    tiempo  = tiempo - tiempo[0]  # Origen temporal al inicio del primer ciclo.


    # Algunas verificaciones de datos del archivo
    if not np.all(np.isfinite([tiempo, duty, velocidad_medida])):
        raise ValueError('Hay datos no finitos en el archivo.')
    if not np.allclose(np.diff(tiempo), DT, rtol=0.02, atol=1e-6):
        raise ValueError('El muestreo del CSV no coincide con DT. Revisar DT en CONFIGURACION.')
    if np.mean(velocidad_medida) < 0:
        raise ValueError('Esta solucion espera la velocidad positiva del CSV para alumnos.')


    # SEPARAR 7 CICLOS PARA AJUSTAR Y 3 PARA VALIDAR.
    # Los primeros 7 ciclos se utilizan para obtener los parámetros del sistema y los últimos 3
    #para validad que los resultados concuerdan con estos datos.
    # Se considera el protocolo de 6 escalones de 6 s por ciclo (36s de duración por ciclo).
    # Los limites se obtienen por tiempo para tolerar la pequena deriva del reloj.
    fin_ajuste_s = CICLOS_AJUSTE * DURACION_CICLO_S
    fin_total_s  = (CICLOS_AJUSTE + CICLOS_VALIDACION) * DURACION_CICLO_S

    if abs(tiempo[-1] + DT - fin_total_s) > 0.5: # Se verifica que se cuente con los 360s de data
        raise ValueError('Se esperaba un ensayo de diez ciclos de 36 s, aproximadamente 360 s.')

    # Cuenta las muestras anteriores a fin_ajuste_s: se usarán para estimar
    #los parámetros. Las muestras restantes se reservan para validación.    
    cantidad_ajuste = int(np.count_nonzero(tiempo < fin_ajuste_s))

    # Convierte el tiempo inicial descartado en una cantidad de muestras.
    # El modelo se simula desde el comienzo, pero el error se evalúa desde esta posición para
    #reducir el efecto del arranque del promedio móvil y de la condición inicial. Se aplica una
    #sola vez al ensayo.
    inicio_error = int(round(TIEMPO_INICIAL_DESCARTADO_S / DT)) # El tiempo a descartar es el 1er segundo

    if TIEMPO_INICIAL_DESCARTADO_S < VENTANA_S:
        raise ValueError('Descartar al inicio al menos una ventana completa.')

    duty_ajuste       = duty[:cantidad_ajuste]              # Señal de entrada
    medida_ajuste     = velocidad_medida[:cantidad_ajuste]  # Señal de salida
    velocidad_inicial = float(velocidad_medida[0])


    # A5(a): RESIDUOS QUE RECIBE NLS EN CADA ITERACION.
    def residuos(parametros):
        # Calcular la velocidad de los 7 ciclos (duty_ajust) con el sistema y usando velocidad_inicial.
        #Note que debe calcular todo el vector de datos, no remover los datos de inicio_error, dado que se
        #los necesita para calcular la velocidad hasta el punto que no se descarta la información.
        velocidad_calculada    = calcular_velocidad(parametros, duty_ajuste, velocidad_inicial)

        # A la velocidad calculada se le debe aplicar el promediador movil de 0.1 segundos.
        velocidad_calculada_ma = promedio_movil(velocidad_calculada)

        # Calcule la diferencia (el residuo) entre la velocidad calculada y pasada por el promediador
        #movil para todo el vector de datos.                
        error = velocidad_calculada_ma - medida_ajuste

        # Devuelva los residuos desde inicio_error hasta el final del vector.
        return error[inicio_error:]


    # A5(b): ESTIMACION NLS.
    # Valores iniciales aproximados, calculados SOLO con los datos de ajuste.
    Kdc_inicial = 1; omega_ref_inicial = 1; polo_inicial = 10.0
    parametros_iniciales = [Kdc_inicial, omega_ref_inicial, polo_inicial]

    print(f'DT = {DT} s; ventana = {VENTANA_S} s; N = {N} muestras.')
    print(f'Ventana implementada: N*DT = {N * DT} s.')
    print('Estimando Kdc, omega_ref y polo mediante NLS...')

    resultado = least_squares(residuos, parametros_iniciales,
                              bounds=([0.00001, -10000.0, 0.1], [100000.0, 10000.0, 1000.0]),
                              x_scale='jac',
                              max_nfev=150)

    if not resultado.success:
        raise RuntimeError('NLS no convergio: ' + resultado.message)

    Kdc, omega_ref, polo = resultado.x
    b   = Kdc * polo
    c   = polo * (omega_ref - Kdc * DUTY_REFERENCIA)
    parametros = dict(Kdc=float(Kdc), omega_ref=float(omega_ref), duty_ref=DUTY_REFERENCIA,
                      p=float(polo), b=float(b), c=float(c))

    print('\nModelo: omega_dot + p*omega = b*duty + c')

    for nombre, valor in parametros.items():
        print(f'{nombre} = {valor:.8g}')


    guardar_resultados(5, 'parametros_identificados.json', parametros)
    return parametros


if __name__ == '__main__':
    ejercicio_5()
