"""
Instalar dependencias: pip install numpy scipy matplotlib
Ejecutar: python script_identificacion.py datos_motor.csv

CSV: time_s,duty_effective,omega_prom_measured
La velocidad del CSV YA tiene un promedio movil causal aplicado.
VENTANA_S debe coincidir con la usada para preparar ese CSV (no modificar).
DT debe coincidir con el periodo de muestreo del CSV (no modificar).
"""
from pathlib import Path
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares



# 1. CONFIGURACION: tiempos en segundos (no modificar estos parámetros).
DT = 0.001                          # dt utilizado para obtener las muestras medidas
VENTANA_S = 0.1                     # Ventana utilizada para reducir el ruido de mediciones
DURACION_CICLO_S = 36.0             # La duración de cada ciclo de datos.
CICLOS_AJUSTE = 7                   # 7 de los 10 ciclos para obtener los parámetros.
CICLOS_VALIDACION = 3               # 3 de los 10 ciclos para verificar el modelo obtenido.
DUTY_REFERENCIA = 0.675             # Valor medio del duty, las muestras usaron un duty entre 0.6 y 0.75
TIEMPO_INICIAL_DESCARTADO_S = 1.0

# Una evaluacion del error cada 0.01 s para acelerar NLS.
# La simulacion y el promedio conservan TODAS las muestras a intervalo DT.
INTERVALO_ERROR_S = 0.01

# N se calcula: no se fija en 100.
if not np.isfinite(DT) or not np.isfinite(VENTANA_S) or DT <= 0 or VENTANA_S <= 0:
    raise ValueError('DT y VENTANA_S deben ser positivos y finitos.')

N = int(round(VENTANA_S / DT))  # Cantidad de muestras en la ventana de 0.1 segundos

if DT <= 0 or VENTANA_S <= 0 or N < 1:
    raise ValueError('DT y VENTANA_S deben ser positivos; la ventana debe contener al menos una muestra.')





# 2. ECUACION DEL MOTOR.
# Parametros a estimar: Kdc, omega_ref, polo.
# omega_dot = -polo*(omega-omega_ref) + polo*Kdc*(duty-DUTY_REFERENCIA)
# La ecuacion se integra mediante Euler hacia adelante.
def calcular_velocidad(parametros, duty, velocidad_inicial):
    # Ejercicio 3: integracion por Euler hacia adelante con paso DT.
    #   omega_dot[k] = -polo*(omega[k]-omega_ref) + polo*Kdc*(duty[k]-DUTY_REFERENCIA)
    #   omega[k+1]   = omega[k] + DT*omega_dot[k]
    # Se usa omega[0] = velocidad_inicial y se devuelve un vector del mismo largo que duty.
    # Euler es estable si |1 - polo*DT| < 1, es decir polo < 2/DT = 2000 1/s (las cotas del NLS llegan a 1000).
    Kdc, omega_ref, polo = (float(valor) for valor in parametros)
    duty = np.asarray(duty, dtype=float)
    velocidad = np.empty(len(duty))
    velocidad[0] = velocidad_inicial
    for k in range(len(duty) - 1):
        omega_dot = -polo * (velocidad[k] - omega_ref) + polo * Kdc * (duty[k] - DUTY_REFERENCIA)
        velocidad[k + 1] = velocidad[k] + DT * omega_dot
    return velocidad



# 3. PROMEDIO MOVIL DE LAS ULTIMAS N MUESTRAS.
# Se agrega la muestra actual y se retira la que salio de la ventana.
# Al comienzo sólo se usan las muestras que entren en la ventana de tamaño creciente.
def promedio_movil(velocidad):
    # Completar promedio_movil segun el enunciado.
    raise NotImplementedError('Completar promedio_movil.')




if __name__ == '__main__':
    # 4. LEER LAS TRES COLUMNAS. La velocidad medida no se vuelve a filtrar.
    archivo = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('datos_motor.csv')
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


    # 5. SEPARAR 7 CICLOS PARA AJUSTAR Y 3 PARA VALIDAR.
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



    # 6. RESIDUOS QUE RECIBE NLS EN CADA ITERACION.
    def residuos(parametros):
        # 6.1) Calcular la velocidad de los 7 ciclos (duty_ajust) con el sistema y usando velocidad_inicial.
        #Note que debe calcular todo el vector de datos, no remover los datos de inicio_error, dado que se
        #los necesita para calcular la velocidad hasta el punto que no se descarta la información.
        velocidad_calculada    = calcular_velocidad(parametros, duty_ajuste, velocidad_inicial)
        
        # 6.2) A la velocidad calculada se le debe aplicar el promediador movil de 0.1 segundos.
        velocidad_calculada_ma = promedio_movil(velocidad_calculada)
        
        # 6.3) Calcule la diferencia (el residuo) entre la velocidad calculada y pasada por el promediador
        #movil para todo el vector de datos.                
        error = velocidad_calculada_ma - medida_ajuste
        
        # 6.4) Devuelva los residuos desde inicio_error hasta el final del vector.
        return error[inicio_error:]




    # 7. ESTIMACION NLS.
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




    # 8. CALCULAR LA RESPUESTA COMPLETA CON LOS PARAMETROS YA ESTIMADOS.
    # Se simula de forma continua; no se reinicia el estado en cada ciclo.
    # Los ultimos tres ciclos no intervienen en la estimacion.
    calculada    = calcular_velocidad(resultado.x, duty, velocidad_inicial)
    calculada_ma = promedio_movil(calculada)




    # 9. INDICADORES: todos se calculan entre las dos velocidades promediadas.
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





    # 10. GUARDAR RESULTADOS Y GRAFICOS JUNTO AL CSV.
    carpeta = archivo.resolve().parent / 'resultados_identificacion' # Para guardar JSON de salida
    carpeta.mkdir(exist_ok=True)
    reporte = dict(parameters=parametros, metrics=indicadores, dt_s=DT, 
                   average_window_requested_s=VENTANA_S, average_samples=N, 
                   average_window_actual_s=N*DT, optimizer_success=bool(resultado.success))
    
    
    # Se guardan los resultados en un json (parámetros y metricas de validacion).
    with (carpeta / 'parametros_y_validacion.json').open('w', encoding='utf-8') as salida:
        json.dump(reporte, salida, indent=2)
    print('\nJSON guardado en:', carpeta / 'parametros_y_validacion.json\n\n')
    
    
    # Se muestran las figuras por pantalla.
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
    

    fig, eje = plt.subplots(figsize=(10, 4))
    eje.plot(tiempo, velocidad_medida, label='Mediciones con Promedio Móvil')
    eje.plot(tiempo, calculada_ma, '--', label='Modelo Estimado')
    eje.set_xlim(fin_ajuste_s + 5.8, fin_ajuste_s + 7.5)
    eje.set_xlabel('Tiempo [s]')
    eje.set_ylabel('Velocidad [rad/s]')
    eje.grid(True)
    eje.legend(loc='upper left')
    fig.tight_layout()
    plt.show()