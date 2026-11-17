# Parte A - Ejercicio 5: Estimacion de parametros por minimos cuadrados no lineales (NLS)
import os
import sys
import numpy as np
from scipy.optimize import least_squares


carpeta = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(carpeta, '..', 'consigna'))
from script_identificacion import (DT, N, DURACION_CICLO_S, CICLOS_AJUSTE, DUTY_REFERENCIA,
                                   TIEMPO_INICIAL_DESCARTADO_S, calcular_velocidad, promedio_movil)


datos = np.genfromtxt(os.path.join(carpeta, '..', 'consigna', 'datos_motor.csv'),
                      delimiter=',', names=True, encoding='utf-8-sig')
tiempo = datos['time_s'] - datos['time_s'][0]
duty = datos['duty_effective']
velocidad_medida = datos['omega_prom_measured']

# Los primeros 7 ciclos (0 a 252 s) se usan para el ajuste
fin_ajuste_s = CICLOS_AJUSTE * DURACION_CICLO_S
cantidad_ajuste = np.count_nonzero(tiempo < fin_ajuste_s)
duty_ajuste = duty[:cantidad_ajuste]
medida_ajuste = velocidad_medida[:cantidad_ajuste]
velocidad_inicial = velocidad_medida[0]

# El error se evalua desde t = 1 s (k = 1000) para no usar el arranque del promedio
inicio_error = int(round(TIEMPO_INICIAL_DESCARTADO_S / DT))


def residuos(parametros):
    velocidad_calculada = calcular_velocidad(parametros, duty_ajuste, velocidad_inicial)
    velocidad_calculada_prom = promedio_movil(velocidad_calculada)
    error = velocidad_calculada_prom - medida_ajuste
    return error[inicio_error:]


parametros_iniciales = [1, 1, 10.0]          # Kdc, omega_ref, p
print("N =", N, "muestras. Estimando Kdc, omega_ref y p ...")
resultado = least_squares(residuos, parametros_iniciales,
                          bounds=([0.00001, -10000.0, 0.1], [100000.0, 10000.0, 1000.0]),
                          x_scale='jac', max_nfev=150)
print("Convergio:", resultado.success, "-", resultado.message)

# Si el ajuste no converge se detiene el programa y no se guardan parametros
if not resultado.success:
    raise RuntimeError('NLS no convergio: ' + resultado.message)

Kdc, omega_ref, p = resultado.x
b = p * Kdc
c = p * (omega_ref - Kdc * DUTY_REFERENCIA)
J = np.sum(resultado.fun**2)

print("\nParametros identificados:")
print("Kdc       =", round(Kdc, 3), "(rad/s)/duty")
print("omega_ref =", round(omega_ref, 3), "rad/s")
print("p         =", round(p, 3), "1/s   (tau = 1/p =", round(1000 / p, 1), "ms)")
print("b = p*Kdc                     =", round(b, 2), "rad/s^2")
print("c = p*(omega_ref - Kdc*d_ref) =", round(c, 2), "rad/s^2")
print("J (suma de residuos^2)        =", round(J, 1), "con", len(resultado.fun), "muestras")
print("RMS de los residuos           =", round(np.sqrt(J / len(resultado.fun)), 3), "rad/s")

print("\nModelo identificado:")
print(f"dw/dt + {p:.3f} w = {b:.2f} d + ({c:.2f})")


num = [b]
den = [1, p]
print("\nFuncion de transferencia G(s) = dW(s)/dD(s):")
print(f"G(s) = {b:.2f} / (s + {p:.3f})")
print("Polo:", np.roots(den), "1/s ; ganancia estatica G(0) =", round(b / p, 3), "= Kdc")

# Se guardan los parametros para el ejercicio 6 y la Parte B: Kdc, omega_ref, p
os.makedirs(os.path.join(carpeta, 'resultados', '5'), exist_ok=True)
np.savetxt(os.path.join(carpeta, 'resultados', '5', 'parametros_identificados.txt'),
           [Kdc, omega_ref, p], header='Kdc [(rad/s)/duty], omega_ref [rad/s], p [1/s]')
