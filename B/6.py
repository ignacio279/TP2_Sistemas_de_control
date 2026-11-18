# Parte B - Ejercicio 6(b)-(d): diseno, simulacion e indicadores de control
# Se debe tener ejecutado A/5.py primero para tener los parametros identificados.
# Formulacion: Teorica 16, Control y Estimacion Optima, paginas 18-19.
# Se usa el vector del TP z=[theta-r, omega, xi], con xi_dot=r-theta.
# Para referencia constante y t>0: z_dot=A_a*z+B_a*u, u=-K_a*z.
import os
import numpy as np
import control as ctrl
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, quad
from scipy.optimize import brentq

plt.rcParams.update({'font.size': 13})

carpeta = os.path.dirname(os.path.abspath(__file__))
parametros = np.loadtxt(os.path.join(carpeta, '..', 'A', 'resultados', '5',
                                    'parametros_identificados.txt'))
Kdc, omega_ref, p = parametros
b = p * Kdc
A_a = np.array([[0, 1, 0],
                [0, -p, 0],
                [-1, 0, 0]])
B_a = np.array([[0],
                [b],
                [0]])
Q = np.diag([1.0, 1.0, 1.0])

# ctrl.lqr aplicado al sistema aumentado realiza el diseno LQI.
# Resuelve A_a.T*P+P*A_a-P*B_a*R^(-1)*B_a.T*P+Q=0
# y devuelve K_a=R^(-1)*B_a.T*P, P y los polos de A_a-B_a*K_a.
# No se reutiliza KR de B3: se calculan nuevas ganancias optimas.
R_1 = np.array([[1.0]])
R_100 = np.array([[100.0]])
K_1, P_1, _ = ctrl.lqr(A_a, B_a, Q, R_1)
K_100, P_100, _ = ctrl.lqr(A_a, B_a, Q, R_100)
A_cl_1 = A_a - B_a @ K_1
A_cl_100 = A_a - B_a @ K_100
poles_1 = np.linalg.eigvals(A_cl_1)
poles_100 = np.linalg.eigvals(A_cl_100)

salida = os.path.join(carpeta, 'resultados', '6')
os.makedirs(salida, exist_ok=True)
np.savetxt(os.path.join(salida, 'A_a.txt'), A_a)
np.savetxt(os.path.join(salida, 'B_a.txt'), B_a)
np.savetxt(os.path.join(salida, 'Q.txt'), Q)
for nombre, R, K, P, poles in [('R1', R_1, K_1, P_1, poles_1),
                              ('R100', R_100, K_100, P_100, poles_100)]:
    print(nombre)
    print('K_a =', K)
    print('Polos de A_a-B_a@K_a =', poles)
    print('Ganancia integral (tercera componente, u=-K_a*z) =', K[0, 2])
    np.savetxt(os.path.join(salida, 'K_' + nombre + '.txt'), K,
               header='k1 [duty/rad], k2 [duty/(rad/s)], ki [duty/(rad*s)]; u=-K_a*z, ki<0')
    np.savetxt(os.path.join(salida, 'P_' + nombre + '.txt'), P,
               header='Solucion estabilizante de la ecuacion algebraica de Riccati')
    np.savetxt(os.path.join(salida, 'polos_' + nombre + '.txt'),
               np.column_stack((poles.real, poles.imag)),
               header='Parte real [1/s], parte imaginaria [1/s]')
print('Resultados guardados en:', salida)

# (c) Escalon de 0 a 5 grados en t=0, con theta(0)=omega(0)=xi(0)=0.
# El primer estado aumentado es theta-r, por lo que z1(0+)=-r0.
r0 = 5 * np.pi / 180
z0 = np.array([-r0, 0.0, 0.0])
t0 = 0.0
tf = 20.0
# El polo rapido de R=1 tiene una constante de tiempo de unos 0.45 ms.
# Se guardan muestras cada 10 us en los primeros 50 ms y cada 1 ms despues.
# Esta malla permite representar el pico inicial de u sin perder su caida.
t_eval = np.unique(np.concatenate((np.linspace(t0, 0.05, 5001),
                                   np.linspace(0.05, tf, 19951))))


def closed_loop_R1(t, z):
    return A_cl_1 @ z


def closed_loop_R100(t, z):
    return A_cl_100 @ z


# Radau permite integrar eficientemente las escalas rapida y lenta del LQI.
# Se mantienen las tolerancias de B4 y B5; el paso interno es adaptativo.
sol_1 = solve_ivp(closed_loop_R1, [t0, tf], z0, method='Radau',
                  jac=A_cl_1, t_eval=t_eval, dense_output=True,
                  rtol=1e-9, atol=1e-9)
sol_100 = solve_ivp(closed_loop_R100, [t0, tf], z0, method='Radau',
                    jac=A_cl_100, t_eval=t_eval, dense_output=True,
                    rtol=1e-9, atol=1e-9)
if not sol_1.success or not sol_100.success:
    raise RuntimeError(f'Error de integracion: R1={sol_1.message}; R100={sol_100.message}')

t = sol_1.t
r = np.full_like(t, r0)
z_1 = sol_1.y
z_100 = sol_100.y
# Se recupera la posicion sumando la referencia al primer estado aumentado.
theta_1 = z_1[0, :] + r
theta_100 = z_100[0, :] + r
omega_1 = z_1[1, :]
omega_100 = z_100[1, :]
xi_1 = z_1[2, :]
xi_100 = z_100[2, :]
u_1 = (-K_1 @ z_1)[0, :]
u_100 = (-K_100 @ z_100)[0, :]

for nombre, z, theta, omega, xi, u in [
        ('R1', z_1, theta_1, omega_1, xi_1, u_1),
        ('R100', z_100, theta_100, omega_100, xi_100, u_100)]:
    np.savetxt(os.path.join(salida, 'respuesta_' + nombre + '.txt'),
               np.array([t, r, theta, omega, xi, z[0, :], -z[0, :], u]).T,
               header=('t [s], r [rad], theta [rad], omega [rad/s], xi [rad*s], '
                       'theta-r [rad], r-theta [rad], u [variacion de duty]; '
                       'malla temporal no uniforme'))
    print(nombre, 'control inicial [variacion de duty] =', u[0])
    print(nombre, 'maxima posicion [grados] =', np.max(theta) * 180 / np.pi)
    print(nombre, 'posicion a 20 s [grados] =', theta[-1] * 180 / np.pi)

plt.figure(figsize=(6.5, 4.2))
plt.plot(t, theta_1 * 180 / np.pi, 'b-', linewidth=2, label='LQI: R = 1')
plt.plot(t, theta_100 * 180 / np.pi, 'r--', linewidth=2, label='LQI: R = 100')
plt.plot(t, r * 180 / np.pi, 'k-.', linewidth=1.5, label='Referencia: 5 grados')
plt.axhline(5.1, color='g', linestyle=':', label='Banda: +/-0.1 grados')
plt.axhline(4.9, color='g', linestyle=':')
plt.xlabel('Tiempo [s]')
plt.ylabel('Desviacion angular [grados]')
plt.title('B6 - Seguimiento de posición angular')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(salida, 'fig_B6_posicion_comparacion.pdf'))

fig, ax = plt.subplots(figsize=(6.5, 4.2))
ax.plot(t, u_1, 'b-', linewidth=2, label='LQI: R = 1')
ax.plot(t, u_100, 'r--', linewidth=2, label='LQI: R = 100')
ax.set_xlabel('Tiempo [s]')
ax.set_ylabel('Control u [variación de duty]')
ax.set_title('B6 - Entrada de control')
ax.grid(True)
ax.legend(loc='upper right')
# Ampliacion del transitorio inicial: en la escala de 20 s se ve casi vertical.
detalle = ax.inset_axes([0.43, 0.25, 0.53, 0.43])
inicio = t <= 0.05
detalle.plot(t[inicio] * 1000, u_1[inicio], 'b-', linewidth=2)
detalle.plot(t[inicio] * 1000, u_100[inicio], 'r--', linewidth=2)
detalle.set_xlim(0, 50)
detalle.set_xlabel('Tiempo [ms]', fontsize=10)
detalle.set_title('Primeros 50 ms', fontsize=10)
detalle.tick_params(labelsize=9)
detalle.grid(True)
fig.tight_layout()
fig.savefig(os.path.join(salida, 'fig_B6_control_comparacion.pdf'))

# (d) Comparacion con el controlador rapido de B4, con estados reales.
# Se integra en coordenadas [theta-r, omega] para usar la misma referencia
# y una dinamica autonoma, equivalente a x_dot=(A-B*KR)*x+B*kR1*r.
A = A_a[:2, :2]
B = B_a[:2, :]
KR = ctrl.place(A, B, [-4, -8])
A_cl_R = A - B @ KR
z0_R = np.array([-r0, 0.0])


def closed_loop_rapido(t, z):
    return A_cl_R @ z


sol_R = solve_ivp(closed_loop_rapido, [t0, tf], z0_R, method='Radau',
                  jac=A_cl_R, t_eval=t_eval, dense_output=True,
                  rtol=1e-9, atol=1e-9)
if not sol_R.success:
    raise RuntimeError(f'Error de integracion del controlador rapido: {sol_R.message}')
theta_R = sol_R.y[0, :] + r
u_R = (-KR @ sol_R.y)[0, :]
np.savetxt(os.path.join(salida, 'respuesta_rapido.txt'),
           np.array([t, r, theta_R, sol_R.y[1, :], -sol_R.y[0, :], u_R]).T,
           header=('t [s], r [rad], theta [rad], omega [rad/s], '
                   'r-theta [rad], u [variacion de duty]; malla temporal no uniforme'))


def calcular_indicadores(sol, A_cl, K):
    # La primera coordenada siempre es theta-r. El criterio es |error|<=0.02*r0.
    # Se localiza el ULTIMO ingreso a la banda, porque los LQI tienen sobrepico.
    tolerancia = 0.02 * abs(r0)
    fuera = np.flatnonzero(np.abs(sol.y[0, :]) > tolerancia)
    if len(fuera) == 0:
        ts = t0
    elif fuera[-1] == len(sol.t) - 1:
        ts = np.nan  # No se establece dentro del horizonte simulado.
    else:
        i = fuera[-1]
        ts = brentq(lambda tiempo: abs(sol.sol(tiempo)[0]) - tolerancia,
                    sol.t[i], sol.t[i + 1], xtol=1e-12)

    # Se comparan los extremos del intervalo y los extremos interiores de u.
    # u_dot=-K*A_cl*z; las raices se refinan usando la solucion continua.
    derivada = (-K @ A_cl @ sol.y)[0, :]
    cruces = np.flatnonzero(((derivada[:-1] < 0) & (derivada[1:] > 0)) |
                           ((derivada[:-1] > 0) & (derivada[1:] < 0)))
    candidatos = [t0, tf]
    for i in cruces:
        candidatos.append(brentq(
            lambda tiempo: (-K @ A_cl @ sol.sol(tiempo)).item(),
            sol.t[i], sol.t[i + 1], xtol=1e-12))
    candidatos.extend(sol.t[derivada == 0].tolist())
    controles = np.array([abs((-K @ sol.sol(tiempo)).item())
                          for tiempo in candidatos])
    indice = np.argmax(controles)
    umax = controles[indice]
    t_umax = candidatos[indice]

    # La consigna dice '10 segundos' en el texto, pero muestra integral de 0 a 20.
    # Se adopta [0,20] conforme a la formula y al horizonte de simulacion.
    # La cuadratura adaptativa se separa en intervalos para captar el pico rapido.
    limites = [0.0, 0.005, 0.05, 0.5, 5.0, 20.0]
    Eu = sum(quad(lambda tiempo: (-K @ sol.sol(tiempo)).item() ** 2,
                  inicio, fin, epsabs=1e-13, epsrel=1e-9, limit=200)[0]
             for inicio, fin in zip(limites[:-1], limites[1:]))

    # Para los tres sistemas autonomos estables, z_ss=0 y theta_ss=r0.
    # Se distingue el error estacionario analitico del error numerico a los 20 s.
    if not np.all(np.linalg.eigvals(A_cl).real < 0):
        raise ValueError('El calculo del error estacionario requiere estabilidad.')
    e_ss = 0.0
    e_final = -sol.y[0, -1]
    return ts, e_ss, umax, Eu, t_umax, e_final


indicadores = []
for nombre, sol, A_cl, K in [('LQI_R1', sol_1, A_cl_1, K_1),
                            ('LQI_R100', sol_100, A_cl_100, K_100),
                            ('Rapido_B4', sol_R, A_cl_R, KR)]:
    valores = calcular_indicadores(sol, A_cl, K)
    indicadores.append(valores)
    ts, e_ss, umax, Eu, t_umax, e_final = valores
    print(f'{nombre}: ts={ts:.8f} s, e_ss={e_ss:.1f} rad, '
          f'max|u|={umax:.10e}, Eu(0,20)={Eu:.10e}, '
          f't_max|u|={t_umax:.8f} s, e(20)={e_final:.6e} rad')
np.savetxt(os.path.join(salida, 'indicadores.txt'), np.array(indicadores),
           header=('Filas: LQI_R1, LQI_R100, Rapido_B4 (estados reales)\n'
                   'Columnas: ts [s], e_ss [rad], max|u| [variacion de duty], '
                   'Eu integral 0..20 [s], t_max|u| [s], e(20) [rad]\n'
                   'ts: ultimo ingreso a banda +/-2%; e_ss: limite analitico; '
                   'Eu: cuadratura adaptativa de u^2 con solucion continua'))
plt.show()
