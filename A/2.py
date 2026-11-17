# Parte A, ejercicio 2: modelo propuesto.
# Este ejercicio es analítico y se resuelve en el informe; no requiere cálculo numérico.
# Resumen de lo deducido allí:
#   (a) Kdc [(rad/s)/duty]: ganancia estática; omega_ref [rad/s]: velocidad de equilibrio
#       con d = d_ref; p [1/s]: polo, inversa de la constante de tiempo (tau = 1/p).
#   (b) Equilibrio (omega_dot = 0, d constante): omega* = omega_ref + Kdc (d - d_ref).
#   (c) omega_dot + p omega = b d + c, con b = p Kdc y c = p (omega_ref - Kdc d_ref).
# Los valores numéricos de Kdc, omega_ref, p, b y c se obtienen en el ejercicio 5 (A/5.py).
