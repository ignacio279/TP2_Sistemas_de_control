# Parte A - Ejercicio 4
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Letra grande y en negrita para que las figuras se lean bien en el informe
plt.rcParams.update({'font.size': 13, 'font.weight': 'bold', 'axes.labelweight': 'bold',
                     'axes.titleweight': 'bold', 'axes.linewidth': 1.5})


carpeta = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(carpeta, '..', 'consigna'))
from script_identificacion import DT, VENTANA_S, N, promedio_movil

# (a) Cantidad de muestras de la ventana: N = Tv / Ts
print("Ts =", DT, "s ; Tv =", VENTANA_S, "s")
print("N = Tv / Ts =", VENTANA_S / DT, "->", N, "muestras")

x = np.arange(1.0, 301.0)                 
y = promedio_movil(x)
y_directo = np.zeros(len(x))
for k in range(len(x)):
    inicio = max(0, k - N + 1)             # al comienzo solo hay k+1 muestras
    y_directo[k] = np.mean(x[inicio:k + 1])

print("Largo de entrada y salida:", len(x), len(y))
print("Primeras salidas (ventana creciente):", y[:4])     # 1, 1.5, 2, 2.5
print("Salida en k = 299:", y[299], "(promedio de 201..300 = 250.5)")
print("Maxima diferencia con el calculo directo:", np.max(np.abs(y - y_directo)))


# Prueba extra con la exponencial que vimos en clase para ver como el promedio movil cambia el transitorio
# elegimos tau = 0.11 para esta ilustracion
t = np.arange(0, 1.5, DT)
tau = 0.11
w = np.where(t >= 0.5, 1 - np.exp(-(t - 0.5) / tau), 0.0)
w_prom = promedio_movil(w)

plt.figure(figsize=(6.5, 4.2))
plt.plot(t, w, 'b-', linewidth=2.2, label='Primer orden sin promediar')
plt.plot(t, w_prom, 'r--', linewidth=2.2, label='Con promedio movil de 0.1 s')
plt.axhline(1 - np.exp(-1), color='k', linestyle=':', linewidth=1.8, label='63.2 %')
plt.xlabel('Tiempo [s]')
plt.ylabel('Respuesta normalizada')
plt.title('Efecto del promedio movil sobre un escalon')
plt.grid(True)
plt.legend(loc='lower right', fontsize=10)
plt.tight_layout()
os.makedirs(os.path.join(carpeta, 'resultados', '4'), exist_ok=True)
plt.savefig(os.path.join(carpeta, 'resultados', '4', 'fig_A4_promedio_movil.pdf'))


t63 = t[np.argmax(w >= 1 - np.exp(-1))] - 0.5
t63_prom = t[np.argmax(w_prom >= 1 - np.exp(-1))] - 0.5
print("t63 sin promedio:", round(t63, 3), "s ; con promedio:", round(t63_prom, 3), "s")
plt.show()
