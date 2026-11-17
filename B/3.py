# Parte B - Ejercicio 3: respuesta rapida y comparacion con la lenta.
# Ejecutar A/5.py primero. Luego: python B/3.py
# Las metricas de (b) y la comparacion de (d) se desarrollan en el informe!
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

new_poles = [-4, -8]
KR = ctrl.place(A, B, new_poles)
A_cl_R = A - B @ KR
poles_R = np.linalg.eigvals(A_cl_R)
print('KR =', KR)
print('Polos solicitados =', new_poles)
print('Polos de A-B@KR =', poles_R)

#se repite el bloque para comparar con B2 usando exactamente el mismo modelo
KL = ctrl.place(A, B, [-0.5, -1])
A_cl_L = A - B @ KL
poles_L = np.linalg.eigvals(A_cl_L)

theta0 = 5 * np.pi / 180
z0 = np.array([theta0, 0.0])
t0 = 0.0
tf = 20.0
t_eval = np.linspace(t0, tf, 20001)


def closed_loop_R(t, z):
    return A_cl_R @ z


def closed_loop_L(t, z):
    return A_cl_L @ z


sol_R = solve_ivp(closed_loop_R, [t0, tf],
                  z0, t_eval=t_eval, rtol=1e-9, atol=1e-9)
sol_L = solve_ivp(closed_loop_L, [t0, tf],
                  z0, t_eval=t_eval, rtol=1e-9, atol=1e-9)
t_R = sol_R.t
z_R = sol_R.y
t_L = sol_L.t
z_L = sol_L.y
theta_R = z_R[0, :]
omega_R = z_R[1, :]
theta_L = z_L[0, :]
omega_L = z_L[1, :]
u_R = -KR @ z_R
u_R = u_R[0, :]
u_L = -KL @ z_L
u_L = u_L[0, :]

# Las metricas de (b) se calculan analiticamente en el informe!
# Los graficos de (c) permiten comparar las respuestas en (d)

os.makedirs(os.path.join(carpeta, 'resultados', '3'), exist_ok=True)
np.savetxt(os.path.join(carpeta, 'resultados', '3', 'KR.txt'), KR,
           header='kR1 [duty/rad], kR2 [duty/(rad/s)], u=-KR*x')
np.savetxt(os.path.join(carpeta, 'resultados', '3', 'polos_R.txt'), poles_R)
np.savetxt(os.path.join(carpeta, 'resultados', '3', 'respuesta_R.txt'),
           np.array([t_R, theta_R, omega_R, u_R]).T,
           header='t [s], Delta(theta) [rad], Delta(omega) [rad/s], Delta(d) [duty]')

plt.figure(figsize=(6.5, 4.2))
plt.plot(t_L, theta_L * 180 / np.pi, 'b-', linewidth=2, label='Lento: -0.5, -1')
plt.plot(t_R, theta_R * 180 / np.pi, 'r--', linewidth=2, label='Rapido: -4, -8')
plt.axhline(0.1, color='g', linestyle=':', label='Banda: +/-0.1 grados')
plt.axhline(-0.1, color='g', linestyle=':')
plt.xlabel('Tiempo [s]')
plt.ylabel('Desviacion angular [grados]')
plt.title('B3 - Posición angular')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(carpeta, 'resultados', '3', 'fig_B3_posicion_comparacion.pdf'))

plt.figure(figsize=(6.5, 4.2))
plt.plot(t_L, omega_L, 'b-', linewidth=2, label='Lento: -0.5, -1')
plt.plot(t_R, omega_R, 'r--', linewidth=2, label='Rapido: -4, -8')
plt.xlabel('Tiempo [s]')
plt.ylabel('Desviacion de velocidad [rad/s]')
plt.title('B3 - Velocidad angular')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(carpeta, 'resultados', '3', 'fig_B3_velocidad_comparacion.pdf'))

plt.figure(figsize=(6.5, 4.2))
plt.plot(t_L, u_L, 'b-', linewidth=2, label='Lento: u = -KL x')
plt.plot(t_R, u_R, 'r--', linewidth=2, label='Rapido: u = -KR x')
plt.xlabel('Tiempo [s]')
plt.ylabel('Control u [variación de duty]')
plt.title('B3 - Entrada de control')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(carpeta, 'resultados', '3', 'fig_B3_control_comparacion.pdf'))

plt.figure(figsize=(6.5, 4.2))
plt.plot(poles_L.real, poles_L.imag, 'bo', label='Lento: -0.5, -1')
plt.plot(poles_R.real, poles_R.imag, 'rx', label='Rapido: -4, -8')
plt.axhline(0, color='gray', linewidth=0.7)
plt.axvline(0, color='gray', linewidth=0.7)
plt.xlabel('Parte real [1/s]')
plt.ylabel('Parte imaginaria [1/s]')
plt.title('B3 - Polos verificados de A - BK')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(carpeta, 'resultados', '3', 'fig_B3_polos.pdf'))
plt.show()
