"""Parte A, ejercicio 3: Euler hacia adelante y su verificación.
A3(a-b): se completó calcular_velocidad en consigna/script_identificacion.py.
Aquí se importa esa función y se realiza únicamente la verificación de A3(c).
Ejecutar desde la raíz del repo: python A/3.py
"""
from pathlib import Path
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RAIZ = Path(__file__).resolve().parent.parent
CARPETA_RESULTADOS = Path(__file__).resolve().parent / 'resultados'

# La configuración se toma del archivo de la cátedra, sin duplicarla.
sys.path.insert(0, str(RAIZ / 'consigna'))
# A3(a-b) se completó allí; aquí se importa Euler para verificar A3(c).
from script_identificacion import (calcular_velocidad, DT, DUTY_REFERENCIA)

plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': 0.35,
                     'savefig.dpi': 200, 'savefig.bbox': 'tight',
                     'font.family': 'serif'})


def carpeta_resultados(numero):
    carpeta = CARPETA_RESULTADOS / str(numero)
    carpeta.mkdir(parents=True, exist_ok=True)
    return carpeta


def guardar_resultados(numero, nombre, resultados):
    destino = carpeta_resultados(numero) / nombre
    with destino.open('w', encoding='utf-8') as salida:
        json.dump(resultados, salida, indent=2, ensure_ascii=False)
    print(json.dumps(resultados, indent=2, ensure_ascii=False))
    print('Resultados en:', destino)
    return destino


def guardar(fig, carpeta, nombre):
    carpeta.mkdir(parents=True, exist_ok=True)
    fig.savefig(carpeta / f'{nombre}.pdf')
    plt.close(fig)


def ejercicio_3(carpeta_fig):
    Kdc, omega_ref, p = 240.0, 30.0, 8.0          # Parametros de prueba del enunciado
    d_cte, w0, duracion = 0.70, 30.0, 2.0
    n = int(round(duracion / DT))
    t = np.arange(n) * DT
    duty = np.full(n, d_cte)
    w_euler = calcular_velocidad([Kdc, omega_ref, p], duty, w0)

    # Equilibrio analitico: w* = w_ref + Kdc (d - d_ref).
    w_eq = omega_ref + Kdc * (d_cte - DUTY_REFERENCIA)
    # Solucion exacta de la EDO lineal de primer orden con entrada constante.
    w_exacta = w_eq + (w0 - w_eq) * np.exp(-p * t)

    resultados = dict(w_equilibrio_analitico=w_eq, w_euler_final=float(w_euler[-1]),
                      error_final=float(w_euler[-1] - w_eq),
                      max_error_euler_vs_exacta=float(np.max(np.abs(w_euler - w_exacta))),
                      tau_s=1 / p, t_98_s=4 / p,
                      factor_euler=1 - p * DT, factor_exacto=float(np.exp(-p * DT)))

    fig, ejes = plt.subplots(2, 1, figsize=(3.5, 3.9), sharex=True,
                             gridspec_kw={'height_ratios': [2.2, 1]})
    ejes[0].plot(t, w_euler, lw=1.6, label='Euler, $\\Delta t$=1 ms')
    ejes[0].plot(t, w_exacta, 'k:', lw=1.2, label='solución exacta')
    ejes[0].axhline(w_eq, color='C3', ls='--', lw=0.8, label=fr'$\omega^*$={w_eq:.0f} rad/s')
    ejes[0].axvline(1 / p, color='gray', lw=0.6, ls='-.')
    ejes[0].text(1 / p + 0.03, 31, r'$\tau=1/p$', fontsize=8)
    ejes[0].set_ylabel(r'$\omega$ [rad/s]')
    ejes[0].legend(fontsize=7, loc='center right')
    ejes[1].plot(t, (w_euler - w_exacta) * 1e3, color='C2', lw=1)
    ejes[1].set_ylabel('Euler − exacta\n[mrad/s]')
    ejes[1].set_xlabel('Tiempo [s]')
    guardar(fig, carpeta_fig, 'fig_A3c_verificacion_euler')
    return resultados


if __name__ == '__main__':
    guardar_resultados(3, 'resultados_A3.json', ejercicio_3(carpeta_resultados(3)))
