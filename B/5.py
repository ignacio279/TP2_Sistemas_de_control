# Parte B - Ejercicio 5(c)-(e): controlador rapido con observador de Luenberger
# Se guardan y grafican los estados reales, estimados y errores de estimacion.
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
C = np.array([[1, 0]])

# Mismo controlador rapido de B3, con las ganancias sin redondear.
KR = ctrl.place(A, B, [-4, -8])
# Se aplica asignacion de polos al sistema dual y se transpone el resultado.
# Equivale a L=[60-p, 800-p*(60-p)]^T, obtenido en el inciso (b).
L = ctrl.place(A.T, C.T, [-20, -40]).T
poles_R = np.linalg.eigvals(A - B @ KR)
poles_o = np.linalg.eigvals(A - L @ C)
print('KR =', KR)
print('L =', L)
print('Polos de A-B@KR =', poles_R)
print('Polos de A-L@C =', poles_o)

# Escalon de 0 a 5 grados aplicado en t=0. calculos en radianes.
# La planta parte del origen, pero el observador estima inicialmente 2 grados.
# z agrupa [theta, omega, theta_estimada, omega_estimada].
r0 = 5 * np.pi / 180
x0 = np.array([0.0, 0.0])
xhat0 = np.array([2 * np.pi / 180, 0.0])
z0 = np.concatenate((x0, xhat0))
t0 = 0.0
tf = 4.0
t_eval = np.linspace(t0, tf, 4001)


def closed_loop_observer(t, z):
    x = z[:2]
    xhat = z[2:]
    # El controlador utiliza solo los estados ESTIMADOS y la referencia.
    u = -(KR @ xhat).item() + KR[0, 0] * r0
    # La unica medicion disponible para el observador es y=theta.
    y = (C @ x).item()
    yhat = (C @ xhat).item()
    # Planta y observador reciben exactamente la misma entrada aplicada.
    dx = A @ x + B[:, 0] * u
    dxhat = A @ xhat + B[:, 0] * u + L[:, 0] * (y - yhat)
    return np.concatenate((dx, dxhat))


# Mismo integrador y tolerancias que en B3 y B4.
# Los tiempos de salida estan separados 1 ms, el paso interno es adaptativo.
sol = solve_ivp(closed_loop_observer, [t0, tf],
                z0, t_eval=t_eval, rtol=1e-9, atol=1e-9)
if not sol.success:
    raise RuntimeError(f'Error de integracion: {sol.message}')

t = sol.t
x = sol.y[:2, :]
xhat = sol.y[2:, :]
theta = x[0, :]
omega = x[1, :]
theta_hat = xhat[0, :]
omega_hat = xhat[1, :]
r = np.full_like(t, r0)
u = (-KR @ xhat)[0, :] + KR[0, 0] * r
# Convencion del inciso (b): error de estimacion = estado real - estimado.
error_estimacion = x - xhat
error_seguimiento = r - theta

salida = os.path.join(carpeta, 'resultados', '5')
os.makedirs(salida, exist_ok=True)
np.savetxt(os.path.join(salida, 'KR.txt'), KR,
           header='kR1 [duty/rad], kR2 [duty/(rad/s)], u=-KR*xhat+kR1*r')
np.savetxt(os.path.join(salida, 'L.txt'), L,
           header='l1 [1/s], l2 [1/s^2], xhat_dot=A*xhat+B*u+L*(y-C*xhat)')
np.savetxt(os.path.join(salida, 'polos_R.txt'), poles_R)
np.savetxt(os.path.join(salida, 'polos_observador.txt'), poles_o)
np.savetxt(os.path.join(salida, 'respuesta_observador.txt'),
           np.array([t, r, theta, omega, theta_hat, omega_hat,
                     error_estimacion[0, :], error_estimacion[1, :],
                     error_seguimiento, u]).T,
           header=('t [s], r [rad], theta [rad], omega [rad/s], '
                   'theta_hat [rad], omega_hat [rad/s], '
                   'theta-theta_hat [rad], omega-omega_hat [rad/s], '
                   'r-theta [rad], u [variacion de duty]'))
print('Control inicial [variacion de duty] =', u[0])
print('Posicion final [grados] =', theta[-1] * 180 / np.pi)
print('Error final de estimacion [rad, rad/s] =', error_estimacion[:, -1])
print('Resultados guardados en:', salida)

# (d) Se compara cada estado real con su estimacion durante los 4 segundos.
# La posicion y su error se convierten a grados solo para graficar.
plt.figure(figsize=(6.5, 4.2))
plt.plot(t, theta * 180 / np.pi, 'b-', linewidth=2, label='Posicion real')
plt.plot(t, theta_hat * 180 / np.pi, 'r--', linewidth=2, label='Posicion estimada')
plt.plot(t, r * 180 / np.pi, 'k-.', linewidth=1.5, label='Referencia: 5 grados')
plt.xlabel('Tiempo [s]')
plt.ylabel('Desviacion angular [grados]')
plt.title('B5 - Posición real y estimada')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(salida, 'fig_B5_posicion_estimada.pdf'))

plt.figure(figsize=(6.5, 4.2))
plt.plot(t, omega, 'b-', linewidth=2, label='Velocidad real')
plt.plot(t, omega_hat, 'r--', linewidth=2, label='Velocidad estimada')
plt.xlabel('Tiempo [s]')
plt.ylabel('Desviacion de velocidad [rad/s]')
plt.title('B5 - Velocidad real y estimada')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(salida, 'fig_B5_velocidad_estimada.pdf'))

# Error de estimacion = real - estimado, no es el error de seguimiento r-theta.
plt.figure(figsize=(6.5, 4.2))
plt.plot(t, error_estimacion[0, :] * 180 / np.pi, 'b-', linewidth=2,
         label=r'$\theta-\hat{\theta}$')
plt.axhline(0, color='gray', linewidth=0.7)
plt.xlabel('Tiempo [s]')
plt.ylabel('Error de estimacion [grados]')
plt.title('B5 - Error de estimación de posición')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(salida, 'fig_B5_error_posicion.pdf'))

plt.figure(figsize=(6.5, 4.2))
plt.plot(t, error_estimacion[1, :], 'b-', linewidth=2,
         label=r'$\omega-\hat{\omega}$')
plt.axhline(0, color='gray', linewidth=0.7)
plt.xlabel('Tiempo [s]')
plt.ylabel('Error de estimacion [rad/s]')
plt.title('B5 - Error de estimación de velocidad')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(salida, 'fig_B5_error_velocidad.pdf'))

# Valores de apoyo para contrastar la convergencia observada en los graficos.
indice_medio_segundo = np.argmin(np.abs(t - 0.5))
print('Error de posicion a 0.5 s [grados] =',
      error_estimacion[0, indice_medio_segundo] * 180 / np.pi)
print('Error de velocidad a 0.5 s [rad/s] =',
      error_estimacion[1, indice_medio_segundo])

# (e) Simulacion independiente con realimentacion de estados reales.
# Se conservan KR, la referencia, las condiciones iniciales de la planta,
# los 4 segundos, los tiempos de salida y las tolerancias del caso con observador.
# No basta con evaluar -KR*x sobre la trayectoria del caso con observador
# al cambiar la entrada de control tambien cambia la trayectoria de la planta.
A_cl_R = A - B @ KR
B_r_R = B * KR[0, 0]


def closed_loop_real(t, x_real):
    return A_cl_R @ x_real + B_r_R[:, 0] * r0


sol_real = solve_ivp(closed_loop_real, [t0, tf],
                     x0, t_eval=t_eval, rtol=1e-9, atol=1e-9)
if not sol_real.success:
    raise RuntimeError(f'Error de integracion sin observador: {sol_real.message}')

x_real = sol_real.y
theta_real = x_real[0, :]
omega_real = x_real[1, :]
u_real = (-KR @ x_real)[0, :] + KR[0, 0] * r
np.savetxt(os.path.join(salida, 'respuesta_estados_reales.txt'),
           np.array([t, r, theta_real, omega_real, r - theta_real, u_real]).T,
           header=('t [s], r [rad], theta [rad], omega [rad/s], '
                   'r-theta [rad], u [variacion de duty]'))

plt.figure(figsize=(6.5, 4.2))
plt.plot(t, theta_real * 180 / np.pi, 'b-', linewidth=2,
         label='Estados reales')
plt.plot(t, theta * 180 / np.pi, 'r--', linewidth=2,
         label='Con observador')
plt.plot(t, r * 180 / np.pi, 'k-.', linewidth=1.5,
         label='Referencia: 5 grados')
plt.xlabel('Tiempo [s]')
plt.ylabel('Desviacion angular [grados]')
plt.title('B5 - Comparación de posición angular')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(salida, 'fig_B5_posicion_comparacion.pdf'))

plt.figure(figsize=(6.5, 4.2))
plt.plot(t, u_real, 'b-', linewidth=2, label='Estados reales')
plt.plot(t, u, 'r--', linewidth=2, label='Con observador')
plt.xlabel('Tiempo [s]')
plt.ylabel('Control u [variación de duty]')
plt.title('B5 - Comparación de entrada de control')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(salida, 'fig_B5_control_comparacion.pdf'))

indice_diferencia = np.argmax(np.abs(theta - theta_real))
indice_max_u = np.argmax(np.abs(u))
print('Control inicial con estados reales [variacion de duty] =', u_real[0])
print('Maxima diferencia de posicion entre casos [grados] =',
      np.abs(theta[indice_diferencia] - theta_real[indice_diferencia]) * 180 / np.pi,
      'en t [s] =', t[indice_diferencia])
print('Maximo absoluto del control con observador [variacion de duty] =',
      np.abs(u[indice_max_u]), 'en t [s] =', t[indice_max_u])
plt.show()
