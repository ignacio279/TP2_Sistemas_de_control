# Parte A, ejercicio 2: modelo propuesto.
# Este punto va en el informe, aca dejamos las expresiones que obtuvimos
# (a)Kdc [(rad/s)/duty] es la ganancia estatica
# omega_ref [rad/s] es la velocidad de equilibrio con d = d_ref
# p [1/s] es la inversa de tau, el polo es -p
# (b)Con omega_dot = 0 queda omega* = omega_ref + Kdc*(d-d_ref)
# (c)omega_dot + p*omega = b*d + c, con b = p*Kdc y c = p*(omega_ref-Kdc*d_ref)
# Los valores numericos salen de A/5.py
