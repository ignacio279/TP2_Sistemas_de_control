# Parte B - Ejercicio 2: respuesta lenta.
# Ejecutar A/5.py primero. Luego: python B/2.py
# El inciso (d), tiempo de establecimiento y maximo del control, se desarrolla
# analiticamente en el informe
# Se guardan las curvas y sus datos para realizar ese analisis en el informe
import os
import numpy as np
import control as ctrl
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

plt.rcParams.update({'font.size': 13})

carpeta = os.path.dirname(os.path.abspath(__file__))
parametros = np.loadtxt(os.path.join(carpeta, '..', 'A', 'resultados', '5',
                                    'parametros_identificados.txt'))
Kdc, omega_ref, p = parametros
b = p * Kdc
A = np.array([[0, 1],
              [0, -p]])
B = np.array([[0],
              [b]])

new_poles = [-0.5, -1]
KL = ctrl.place(A, B, new_poles)
A_cl = A - B @ KL
poles_cl = np.linalg.eigvals(A_cl)
print('KL =', KL)
print('Polos solicitados =', new_poles)
print('Polos de A-B@KL =', poles_cl)

# La referencia es cero: se simula REGULACION desde una desviacion inicial
theta0 = 5 * np.pi / 180
z0 = np.array([theta0, 0.0])
t0 = 0.0
tf = 20.0
t_eval = np.linspace(t0, tf, 20001)


def closed_loop_error_system(t, z):
    return A_cl @ z


sol = solve_ivp(closed_loop_error_system, [t0, tf],
                z0, t_eval=t_eval, rtol=1e-9, atol=1e-9)
t = sol.t
z = sol.y
theta = z[0, :]
omega = z[1, :]
u = -KL @ z
u = u[0, :]

# (d) El calculo de ts y del maximo absoluto del control figura en el informe!

os.makedirs(os.path.join(carpeta, 'resultados', '2'), exist_ok=True)
np.savetxt(os.path.join(carpeta, 'resultados', '2', 'KL.txt'), KL,
           header='kL1 [duty/rad], kL2 [duty/(rad/s)], u=-KL*x')
np.savetxt(os.path.join(carpeta, 'resultados', '2', 'polos_L.txt'), poles_cl)
np.savetxt(os.path.join(carpeta, 'resultados', '2', 'respuesta_L.txt'),
           np.array([t, theta, omega, u]).T,
           header='t [s], Delta(theta) [rad], Delta(omega) [rad/s], Delta(d) [duty]')

# Graficos: Guia 9, pp. 2 y 5. Solo se convierte theta a grados al graficar
plt.figure(figsize=(6.5, 4.2))
plt.plot(t, theta * 180 / np.pi, 'b-', linewidth=2, label='Lento: -0.5, -1')
plt.axhline(0.1, color='r', linestyle='--', label='Banda: +/-0.1 grados')
plt.axhline(-0.1, color='r', linestyle='--')
plt.xlabel('Tiempo [s]')
plt.ylabel('Desviacion angular [grados]')
plt.title('B2 - Posición angular')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(carpeta, 'resultados', '2', 'fig_B2_posicion.pdf'))

plt.figure(figsize=(6.5, 4.2))
plt.plot(t, omega, 'b-', linewidth=2, label='Lento: -0.5, -1')
plt.xlabel('Tiempo [s]')
plt.ylabel('Desviacion de velocidad [rad/s]')
plt.title('B2 - Velocidad angular')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(carpeta, 'resultados', '2', 'fig_B2_velocidad.pdf'))

plt.figure(figsize=(6.5, 4.2))
plt.plot(t, u, 'b-', linewidth=2, label='u = -KL x')
plt.xlabel('Tiempo [s]')
plt.ylabel('Control u [variación de duty]')
plt.title('B2 - Entrada de control')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(carpeta, 'resultados', '2', 'fig_B2_control.pdf'))
plt.show()
