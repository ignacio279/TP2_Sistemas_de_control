# Parte B - Ejercicio 4(c): seguimiento de una referencia angular constante
# Los incisos (a) y (b) se desarrollan en el informe, las metricas de (d)
# se calculan a partir de estas mismas respuestas.
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

# Mismas ganancias que en B2 y B3, sin redondear sus componentes.
KL = ctrl.place(A, B, [-0.5, -1])
KR = ctrl.place(A, B, [-4, -8])
A_cl_L = A - B @ KL
A_cl_R = A - B @ KR
B_r_L = B * KL[0, 0]
B_r_R = B * KR[0, 0]
poles_L = np.linalg.eigvals(A_cl_L)
poles_R = np.linalg.eigvals(A_cl_R)
print('KL =', KL)
print('KR =', KR)
print('Polos de A-B@KL =', poles_L)
print('Polos de A-B@KR =', poles_R)

# Escalon de referencia de 0 a 5 grados aplicado en t=0.
# Para todo el intervalo simulado r=5*pi/180; antes del escalon r=0.
# theta, omega y u son desviaciones respecto de la condicion nominal.
# Todos los calculos se hacen en radianes; solo se convierten los graficos.
r0 = 5 * np.pi / 180
z0 = np.array([0.0, 0.0])
t0 = 0.0
tf = 20.0
t_eval = np.linspace(t0, tf, 20001)

# Ley de control de (a): u=-K*x+k1*r.
# Al sustituirla en la planta: x_dot=(A-B*K)*x+B*k1*r.
def closed_loop_L(t, z):
    return A_cl_L @ z + B_r_L[:, 0] * r0


def closed_loop_R(t, z):
    return A_cl_R @ z + B_r_R[:, 0] * r0


# Se conservan el integrador, las tolerancias y los tiempos de salida de B3.
# solve_ivp ajusta internamente su paso de integracion.
sol_L = solve_ivp(closed_loop_L, [t0, tf],
                  z0, t_eval=t_eval, rtol=1e-9, atol=1e-9)
sol_R = solve_ivp(closed_loop_R, [t0, tf],
                  z0, t_eval=t_eval, rtol=1e-9, atol=1e-9)
if not sol_L.success or not sol_R.success:
    raise RuntimeError(f'Error de integracion: L={sol_L.message}; R={sol_R.message}')

t = sol_L.t
z_L = sol_L.y
z_R = sol_R.y
theta_L = z_L[0, :]
omega_L = z_L[1, :]
theta_R = z_R[0, :]
omega_R = z_R[1, :]
r = np.full_like(t, r0)
e_L = r - theta_L
e_R = r - theta_R
u_L = (-KL @ z_L)[0, :] + KL[0, 0] * r
u_R = (-KR @ z_R)[0, :] + KR[0, 0] * r

salida = os.path.join(carpeta, 'resultados', '4')
os.makedirs(salida, exist_ok=True)
np.savetxt(os.path.join(salida, 'KL.txt'), KL,
           header='kL1 [duty/rad], kL2 [duty/(rad/s)], u=-KL*x+kL1*r')
np.savetxt(os.path.join(salida, 'KR.txt'), KR,
           header='kR1 [duty/rad], kR2 [duty/(rad/s)], u=-KR*x+kR1*r')
np.savetxt(os.path.join(salida, 'polos_L.txt'), poles_L)
np.savetxt(os.path.join(salida, 'polos_R.txt'), poles_R)
np.savetxt(os.path.join(salida, 'respuesta_L.txt'),
           np.array([t, r, theta_L, omega_L, e_L, u_L]).T,
           header='t [s], r [rad], Delta(theta) [rad], Delta(omega) [rad/s], e=r-theta [rad], Delta(d) [duty]')
np.savetxt(os.path.join(salida, 'respuesta_R.txt'),
           np.array([t, r, theta_R, omega_R, e_R, u_R]).T,
           header='t [s], r [rad], Delta(theta) [rad], Delta(omega) [rad/s], e=r-theta [rad], Delta(d) [duty]')

plt.figure(figsize=(6.5, 4.2))
plt.plot(t, theta_L * 180 / np.pi, 'b-', linewidth=2, label='Lento: -0.5, -1')
plt.plot(t, theta_R * 180 / np.pi, 'r--', linewidth=2, label='Rapido: -4, -8')
plt.plot(t, r * 180 / np.pi, 'k-.', linewidth=1.5, label='Referencia: 5 grados')
plt.axhline(5.1, color='g', linestyle=':', label='Banda: +/-0.1 grados')
plt.axhline(4.9, color='g', linestyle=':')
plt.xlabel('Tiempo [s]')
plt.ylabel('Desviacion angular [grados]')
plt.title('B4 - Seguimiento de posición angular')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(salida, 'fig_B4_posicion_comparacion.pdf'))

plt.figure(figsize=(6.5, 4.2))
plt.plot(t, e_L * 180 / np.pi, 'b-', linewidth=2, label='Lento: -0.5, -1')
plt.plot(t, e_R * 180 / np.pi, 'r--', linewidth=2, label='Rapido: -4, -8')
plt.axhline(0, color='gray', linewidth=0.7)
plt.axhline(0.1, color='g', linestyle=':', label='Banda: +/-0.1 grados')
plt.axhline(-0.1, color='g', linestyle=':')
plt.xlabel('Tiempo [s]')
plt.ylabel('Error de seguimiento [grados]')
plt.title('B4 - Error de seguimiento')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(salida, 'fig_B4_error_comparacion.pdf'))

plt.figure(figsize=(6.5, 4.2))
plt.plot(t, u_L, 'b-', linewidth=2, label='Lento: u = -KL x + kL1 r')
plt.plot(t, u_R, 'r--', linewidth=2, label='Rapido: u = -KR x + kR1 r')
plt.xlabel('Tiempo [s]')
plt.ylabel('Control u [variación de duty]')
plt.title('B4 - Entrada de control')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(salida, 'fig_B4_control_comparacion.pdf'))
print('Resultados guardados en:', salida)
plt.show()
