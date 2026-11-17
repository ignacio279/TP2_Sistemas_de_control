# Parte B - Ejercicio 5(c): controlador rapido con observador de Luenberger
# Se debe tener ejecutado A/5.py primero para tener los parametros identificados.
# Se mantiene el formato de simulacion de B/3.py y B/4.py.
# Se guardan los estados reales, estimados y errores para los incisos siguientes.
import os
import numpy as np
import control as ctrl
from scipy.integrate import solve_ivp

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

# Escalon de 0 a 5 grados aplicado en t=0; calculos en radianes.
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
# Los tiempos de salida estan separados 1 ms; el paso interno es adaptativo.
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
