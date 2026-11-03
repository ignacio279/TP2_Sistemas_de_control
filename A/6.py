"""Parte A, ejercicio 6: velocidades, error e indicadores de validación.
Requiere los parámetros de A/5.py y el promedio móvil de A/4.py.
Se conservan sin completar RMSE, sesgo y FIT del código original.
Ejecutar desde la raíz del repo: python A/6.py [datos_motor.csv]
A6(d): discusión del alcance del modelo en el informe.
"""
from pathlib import Path
import sys
import json
from importlib import import_module
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RAIZ = Path(__file__).resolve().parent.parent
CARPETA_RESULTADOS = Path(__file__).resolve().parent / 'resultados'

# La configuración se toma del archivo de la cátedra, sin duplicarla.
sys.path.insert(0, str(RAIZ / 'consigna'))
from script_identificacion import (DT, VENTANA_S, DURACION_CICLO_S, CICLOS_AJUSTE, CICLOS_VALIDACION, TIEMPO_INICIAL_DESCARTADO_S)

plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': 0.35,
                     'savefig.dpi': 200, 'savefig.bbox': 'tight',
                     'font.family': 'serif'})


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


def ejercicio_6():
    archivo_parametros = carpeta_resultados(5) / 'parametros_identificados.json'
    if not archivo_parametros.exists():
        raise SystemExit('Primero completar A/4.py y ejecutar A/5.py para generar los parámetros.')
    with archivo_parametros.open(encoding='utf-8') as entrada:
        parametros = json.load(entrada)
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

    velocidad_inicial = float(velocidad_medida[0])


    # A6(a): CALCULAR LA RESPUESTA COMPLETA CON LOS PARAMETROS YA ESTIMADOS.
    # Se simula de forma continua; no se reinicia el estado en cada ciclo.
    # Los ultimos tres ciclos no intervienen en la estimacion.
    calculada    = calcular_velocidad([parametros['Kdc'], parametros['omega_ref'], parametros['p']], duty, velocidad_inicial)
    calculada_ma = promedio_movil(calculada)


    # A6(c): INDICADORES: todos se calculan entre las dos velocidades promediadas.
    indicadores = {}
    tramos = [('ajuste', inicio_error, cantidad_ajuste), ('validacion', cantidad_ajuste, len(tiempo))]
    for nombre, inicio, fin in tramos:
        medida = velocidad_medida[inicio:fin]
        modelo = calculada_ma[inicio:fin]
        error  = medida - modelo
        rmse   = None   # Completar el calculo del RMSE.
        sesgo  = None   # Completar el calculo del sesgo.
        error_maximo  = np.max(np.abs(error))
        fit    = None   # Completar el ajuste porcentual en caso de que esté definido.
        if rmse is None or sesgo is None:
            raise NotImplementedError('Completar RMSE, sesgo y ajuste porcentual.')
        indicadores[nombre] = dict(RMSE_rad_s=float(rmse), bias_rad_s=float(sesgo),
                                   FIT_percent=float(fit) if fit is not None else None,
                                   max_abs_error_rad_s=float(error_maximo))
        print(f'\n{nombre.upper()}: {indicadores[nombre]}')


    guardar_resultados(6, 'validacion_A6.json', dict(parameters=parametros, metrics=indicadores, dt_s=DT))

    # A6(a-b): guardar gráficos de comparación y error.
    fig, ejes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    ejes[0].plot(tiempo, velocidad_medida, label='Mediciones con Promedio Móvil', linewidth=0.8)
    ejes[0].plot(tiempo, calculada_ma, '--', label='Modelo Estimado', linewidth=0.8)
    ejes[0].set_ylabel('Velocidad [rad/s]')
    ejes[0].set_ylim(top=70)
    ejes[0].legend(loc='upper left')
    ejes[1].plot(tiempo, velocidad_medida-calculada_ma, linewidth=0.8,
                 label='Error entre mediciones con promedio móvil y modelo estimado.')
    ejes[1].set_ylabel('Error [rad/s]')
    ejes[1].legend(loc='upper left')
    ejes[2].step(tiempo, duty, where='post', label='Duty utilizado en experimento.')
    ejes[2].set_ylabel('Duty')
    ejes[2].set_ylim(top=0.8)
    ejes[2].set_xlabel('Tiempo [s]')
    ejes[2].legend(loc='upper left')
    for eje in ejes:
        eje.axvline(fin_ajuste_s, color='red', linestyle='--')
        eje.grid(True)
    fig.tight_layout()
    fig.savefig(carpeta_resultados(6) / 'comparacion_A6.pdf')
    plt.close(fig)


    fig, eje = plt.subplots(figsize=(10, 4))
    eje.plot(tiempo, velocidad_medida, label='Mediciones con Promedio Móvil')
    eje.plot(tiempo, calculada_ma, '--', label='Modelo Estimado')
    eje.set_xlim(fin_ajuste_s + 5.8, fin_ajuste_s + 7.5)
    eje.set_xlabel('Tiempo [s]')
    eje.set_ylabel('Velocidad [rad/s]')
    eje.grid(True)
    eje.legend(loc='upper left')
    fig.tight_layout()
    fig.savefig(carpeta_resultados(6) / 'detalle_A6.pdf')
    plt.close(fig)

if __name__ == '__main__':
    ejercicio_6()
