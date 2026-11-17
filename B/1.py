# Parte B - Ejercicio 1: representacion del sistema y controlabilidad
# Ejecutar A/5.py primero. Luego: python B/1.py
# La deduccion de las matrices y el rango se desarrollan en el informe!
# Este script solo evalua las matrices con los parametros identificados
import os
import numpy as np
import control as ctrl

carpeta = os.path.dirname(os.path.abspath(__file__))
parametros = np.loadtxt(os.path.join(carpeta, '..', 'A', 'resultados', '5',
                                    'parametros_identificados.txt'))
Kdc, omega_ref, p = parametros
b = p * Kdc

# (a) Matrices del modelo, su deduccion se desarrolla en el informe!!
# x = [Delta(theta), Delta(omega)]^T, u = Delta(d), y = Delta(theta)
# Se calcula en radianes, rad/s y duty (adimensional), respectivamente

A = np.array([[0, 1],
              [0, -p]])
B = np.array([[0],
              [b]])
C = np.array([[1, 0]])
D = np.array([[0]])
sys = ctrl.ss(A, B, C, D)

print('Kdc, omega_ref, p =', parametros)
print('b = p*Kdc =', b)
print('A =\n', A)
print('B =\n', B)
print('C =\n', C)
print('D =\n', D)

# El rango y la justificacion de la asignacion de polos se analizan en el informe!
Cmat = ctrl.ctrb(A, B)
print('Matriz de controlabilidad =\n', Cmat)

# Guardado con la misma estructura de la Parte A.
os.makedirs(os.path.join(carpeta, 'resultados', '1'), exist_ok=True)
np.savetxt(os.path.join(carpeta, 'resultados', '1', 'A.txt'), A)
np.savetxt(os.path.join(carpeta, 'resultados', '1', 'B.txt'), B)
np.savetxt(os.path.join(carpeta, 'resultados', '1', 'C.txt'), C)
np.savetxt(os.path.join(carpeta, 'resultados', '1', 'D.txt'), D)
np.savetxt(os.path.join(carpeta, 'resultados', '1', 'controlabilidad.txt'), Cmat,
           header='Matriz de controlabilidad [B, AB].')
