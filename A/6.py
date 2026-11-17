# Parte A - Ejercicio 6: Evaluacion del modelo (ajuste y validacion)
# Ejecutar desde la carpeta del TP, despues de A/5.py: python A/6.py
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# letra grande y en negrita para que las figuras se lean bien en el informe!
plt.rcParams.update({'font.size': 13, 'font.weight': 'bold', 'axes.labelweight': 'bold',
                     'axes.titleweight': 'bold', 'axes.linewidth': 1.5})


carpeta = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(carpeta, '..', 'consigna'))
from script_identificacion import (DT, DURACION_CICLO_S, CICLOS_AJUSTE, TIEMPO_INICIAL_DESCARTADO_S,
                                   calcular_velocidad, promedio_movil)


parametros = np.loadtxt(os.path.join(carpeta, 'resultados', '5', 'parametros_identificados.txt'))
print("Kdc, omega_ref, p =", parametros)


datos = np.genfromtxt(os.path.join(carpeta, '..', 'consigna', 'datos_motor.csv'),
                      delimiter=',', names=True, encoding='utf-8-sig')
tiempo = datos['time_s'] - datos['time_s'][0]
duty = datos['duty_effective']
velocidad_medida = datos['omega_prom_measured']        


calculada = calcular_velocidad(parametros, duty, velocidad_medida[0])
calculada_prom = promedio_movil(calculada)              
error = velocidad_medida - calculada_prom               

# Tramos: ajuste de 1 s a 252 s  y validacion de 252 s a 360 s
fin_ajuste_s = CICLOS_AJUSTE * DURACION_CICLO_S
cantidad_ajuste = np.count_nonzero(tiempo < fin_ajuste_s)
inicio_error = int(round(TIEMPO_INICIAL_DESCARTADO_S / DT))
tramos = [('Ajuste', inicio_error, cantidad_ajuste),
          ('Validacion', cantidad_ajuste, len(tiempo))]


for nombre, inicio, fin in tramos:
    e = error[inicio:fin]
    wm = velocidad_medida[inicio:fin]
    rmse = np.sqrt(np.mean(e**2))
    sesgo = np.mean(e)
    fit = 100 * (1 - np.sqrt(np.sum(e**2)) / np.sqrt(np.sum((wm - np.mean(wm))**2)))
    print(f"\n{nombre} ({tiempo[inicio]:.0f} s a {tiempo[fin - 1] + DT:.0f} s, {fin - inicio} muestras)")
    print(f"  RMSE  = {rmse:.3f} rad/s")
    print(f"  Sesgo = {sesgo:.3f} rad/s")
    print(f"  FIT   = {fit:.2f} %")
    print(f"  Error maximo = {np.max(np.abs(e)):.3f} rad/s")

os.makedirs(os.path.join(carpeta, 'resultados', '6'), exist_ok=True)


plt.figure(figsize=(7, 6.5))
plt.subplot(2, 1, 1)
plt.plot(tiempo, velocidad_medida, 'b-', linewidth=1, label='Medida promediada')
plt.plot(tiempo, calculada_prom, 'r--', linewidth=2, label='Modelo promediado')
plt.axvline(fin_ajuste_s, color='k', linestyle='--', linewidth=1.8, label='Fin del ajuste (252 s)')
plt.ylabel('Velocidad [rad/s]')
plt.title('Velocidad medida y calculada')
plt.ylim(5, 65)
plt.grid(True)
plt.legend(loc='upper center', ncol=2, fontsize=10)
plt.subplot(2, 1, 2)
plt.plot(tiempo, error, 'g-', linewidth=1)
plt.axvline(fin_ajuste_s, color='k', linestyle='--', linewidth=1.8)
plt.xlabel('Tiempo [s]')
plt.ylabel('Error [rad/s]')
plt.title('Error e[k] = medida - modelo')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(carpeta, 'resultados', '6', 'fig_A6_ensayo_completo.pdf'))


plt.figure(figsize=(7, 6.5))
plt.subplot(2, 1, 1)
plt.plot(tiempo, velocidad_medida, 'b-', linewidth=2, label='Medida promediada')
plt.plot(tiempo, calculada_prom, 'r--', linewidth=2, label='Modelo promediado')
plt.xlim(288, 324)
plt.ylabel('Velocidad [rad/s]')
plt.title('Detalle de un ciclo de validacion')
plt.ylim(5, 65)
plt.grid(True)
plt.legend(loc='upper left', ncol=2, fontsize=10)
plt.subplot(2, 1, 2)
plt.plot(tiempo, error, 'g-', linewidth=1.5)
plt.xlim(288, 324)
plt.xlabel('Tiempo [s]')
plt.ylabel('Error [rad/s]')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(carpeta, 'resultados', '6', 'fig_A6_detalle_validacion.pdf'))
plt.show()
