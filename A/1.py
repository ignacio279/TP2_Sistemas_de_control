# Parte A, ejercicio 1: datos experimentales, transitorios y recta de régimen.
from pathlib import Path
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# Agregamos algunos extras de lo que vimos en clase
# abajo dejamos las slides que usamos para las formulas y el ajuste

RAIZ = Path(__file__).resolve().parent.parent
CARPETA_RESULTADOS = Path(__file__).resolve().parent / 'resultados'

# La configuración se toma del archivo de la cátedra, sin duplicarla
sys.path.insert(0, str(RAIZ / 'consigna'))
from script_identificacion import (DT, VENTANA_S, DUTY_REFERENCIA, DURACION_CICLO_S, CICLOS_AJUSTE,
                                  promedio_movil)

plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': 0.35,
                     'savefig.dpi': 200, 'savefig.bbox': 'tight',
                     'font.family': 'serif'})


def archivo_datos():
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    return RAIZ / 'consigna' / 'datos_motor.csv'


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


N_VENTANA = int(round(VENTANA_S / DT))
TIEMPO_REGIMEN_S = 1.0
VENTANA_ESCALON_S = 1.5

def detectar_tramos(duty):
    """Devuelve los indices [inicio, fin) de cada tramo de duty constante."""
    cambios = np.flatnonzero(np.diff(duty) != 0) + 1
    bordes = np.concatenate(([0], cambios, [len(duty)]))
    return list(zip(bordes[:-1], bordes[1:]))


def ejercicio_1(tiempo, duty, velocidad, carpeta_fig):
    resultados = {}
    tramos = detectar_tramos(duty)
    fin_ajuste_s = CICLOS_AJUSTE * DURACION_CICLO_S
    duraciones = np.array([(f - i) * DT for i, f in tramos])
    niveles = np.unique(np.round(duty, 6))
    resultados['cantidad_tramos'] = len(tramos)
    resultados['duracion_tramo_s'] = [float(duraciones.min()), float(duraciones.max())]
    resultados['niveles_duty'] = [float(v) for v in niveles]
    resultados['niveles_duty_x255'] = [float(v * 255) for v in niveles]

    # (a)Ensayo completo: entrada y velocidad.
    fig, ejes = plt.subplots(2, 1, figsize=(7.2, 4.2), sharex=True,
                             gridspec_kw={'height_ratios': [2.2, 1]})
    ejes[0].plot(tiempo, velocidad, lw=0.4, color='C0')
    ejes[0].set_ylabel(r'$\omega_m$ [rad/s]')
    ejes[1].step(tiempo, duty, where='post', lw=0.8, color='C3')
    ejes[1].set_ylabel('duty $d$')
    ejes[1].set_xlabel('Tiempo [s]')
    for eje in ejes:
        eje.axvline(fin_ajuste_s, color='k', ls='--', lw=0.8)
    ejes[0].text(fin_ajuste_s / 2, 53, 'ajuste (ciclos 1-7)', ha='center')
    ejes[0].text((fin_ajuste_s + tiempo[-1]) / 2, 53, 'validación (8-10)', ha='center')
    ejes[0].set_ylim(5, 58)
    guardar(fig, carpeta_fig, 'fig_A1a_ensayo_completo')

    # (a)Ciclo 2: usamos sus seis tramos completos y evitamos el arranque del registro
    inicio, fin = tramos[6][0], tramos[11][1]
    fig, ejes = plt.subplots(2, 1, figsize=(3.5, 3.0), sharex=True,
                             gridspec_kw={'height_ratios': [2.2, 1]})
    ejes[0].plot(tiempo[inicio:fin], velocidad[inicio:fin], lw=0.6)
    ejes[0].set_ylabel(r'$\omega_m$ [rad/s]')
    ejes[1].step(tiempo[inicio:fin], duty[inicio:fin], where='post', color='C3', lw=0.9)
    ejes[1].set_ylabel('duty $d$')
    ejes[1].set_xlabel('Tiempo [s]')
    guardar(fig, carpeta_fig, 'fig_A1a_un_ciclo')

    # (c)Velocidad de regimen: promedio del ultimo segundo de cada tramo.
    n_reg = int(round(TIEMPO_REGIMEN_S / DT))
    d_tramo = np.array([duty[i] for i, f in tramos])
    w_ss = np.array([velocidad[f - n_reg:f].mean() for i, f in tramos])
    ciclo = np.arange(len(tramos)) // 6 + 1

    # (b)Probamos comparar las tau usando la exponencial y el ajuste de T10, slides 16-17
    # normalizamos para ajustar solo tau y al modelo le ponemos el promedio movil del TP
    n_esc = int(round(VENTANA_ESCALON_S / DT))
    t_rel = np.arange(-N_VENTANA, n_esc) * DT     
    normalizadas, taus, t63s, subida = [], [], [], []
    for j in range(1, len(tramos)):
        k0 = tramos[j][0]
        w0, w1 = w_ss[j - 1], w_ss[j]
        y = (velocidad[k0:k0 + n_esc] - w0) / (w1 - w0)
        normalizadas.append(y)
        # En T5 slide 6 vimos que el 63.2 % se alcanza en tau, aca el filtro agrega retardo
        t63s.append(np.argmax(y >= 1 - np.exp(-1)) * DT)

        def residuo_tau(x, y=y):
            escalon = np.where(t_rel >= 0, 1 - np.exp(-np.maximum(t_rel, 0) / x[0]), 0.0)
            return promedio_movil(escalon)[N_VENTANA:] - y
        taus.append(least_squares(residuo_tau, [0.1], bounds=([0.005], [2.0])).x[0])
        subida.append(w1 > w0)
    normalizadas, taus, t63s, subida = map(np.array, (normalizadas, taus, t63s, subida))
    # Comparamos subidas y bajadas, usamos la mediana para que pesen menos los extremos
    tau_med = float(np.median(taus))
    resultados['escalones'] = dict(
        t63_crudo_media_s=float(t63s.mean()),
        tau_mediana_s=tau_med, tau_media_s=float(taus.mean()), tau_desvio_s=float(taus.std()),
        tau_subida_media_s=float(taus[subida].mean()), tau_bajada_media_s=float(taus[~subida].mean()),
        p_preliminar_1_s=float(1 / tau_med))

    # Figura del detalle de un escalon (0.651 -> 0.702, t = 42 s, ciclo 2).
    j = 7
    k0 = tramos[j][0]
    t0 = tiempo[k0]
    s = slice(k0 - 300, k0 + n_esc)
    w0, w1 = w_ss[j - 1], w_ss[j]
    t_fino = tiempo[s] - t0
    primer_orden = np.where(t_fino >= 0, 1 - np.exp(-np.maximum(t_fino, 0) / tau_med), 0.0)
    fig, ejes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    ejes[0].plot(t_fino, velocidad[s], lw=0.9, label='medida (con MA)')
    ejes[0].plot(t_fino, w0 + (w1 - w0) * primer_orden, 'k:', lw=1.0,
                 label=fr'1er orden, $\tau$={tau_med*1e3:.0f} ms')
    ejes[0].plot(t_fino, w0 + (w1 - w0) * promedio_movil(primer_orden), 'C3--', lw=1.0,
                 label='1er orden + MA 0.1 s')
    ejes[0].axhline(w0 + (w1 - w0) * (1 - np.exp(-1)), color='gray', lw=0.6, ls='-.')
    ejes[0].set_xlabel(r'$t - t_{esc}$ [s]')
    ejes[0].set_ylabel(r'$\omega$ [rad/s]')
    ejes[0].set_title(f'Escalón d: {duty[k0-1]:.3f} → {duty[k0]:.3f}', fontsize=9)
    ejes[0].legend(fontsize=7, loc='lower right')
    t_n = np.arange(n_esc) * DT
    for y, sube in zip(normalizadas, subida):
        ejes[1].plot(t_n, y, lw=0.3, color='C0' if sube else 'C1', alpha=0.5)
    ejes[1].plot(t_n, promedio_movil(np.concatenate((np.zeros(N_VENTANA), 1 - np.exp(-t_n / tau_med))))[N_VENTANA:],
                 'k--', lw=1.2, label='1er orden + MA')
    ejes[1].plot([], [], color='C0', label='subidas'); ejes[1].plot([], [], color='C1', label='bajadas')
    ejes[1].set_xlim(0, 1.0)
    ejes[1].set_xlabel(r'$t - t_{esc}$ [s]')
    ejes[1].set_ylabel('respuesta normalizada')
    ejes[1].set_title('59 escalones superpuestos', fontsize=9)
    ejes[1].legend(fontsize=7, loc='lower right')
    fig.tight_layout()
    guardar(fig, carpeta_fig, 'fig_A1b_detalle_escalon')

    # (c)Aca pasamos a Python las ecuaciones de minimos cuadrados de T10 slide 8
    Phi = np.column_stack((d_tramo, np.ones_like(d_tramo)))
    m, q = np.linalg.solve(Phi.T @ Phi, Phi.T @ w_ss)
    residuos = w_ss - (m * d_tramo + q)
    # Tambien miramos R2 para ver como queda el ajuste, no es el FIT de A6
    r2 = 1 - np.sum(residuos**2) / np.sum((w_ss - w_ss.mean())**2)
    # Vemos si al repetir la entrada da una velocidad parecida
    por_nivel = []
    for nivel in np.unique(np.round(d_tramo, 6)):
        sel = np.isclose(d_tramo, nivel)
        por_nivel.append(dict(duty=float(nivel), w_ss_media=float(w_ss[sel].mean()),
                              w_ss_desvio=float(w_ss[sel].std(ddof=1)), puntos=int(sel.sum()),
                              residuo_medio=float(residuos[sel].mean())))
    medias = np.array([p['w_ss_media'] for p in por_nivel])
    duties = np.array([p['duty'] for p in por_nivel])
    # Pendientes entre niveles para comparar con la m de la recta
    pendientes_locales = np.diff(medias) / np.diff(duties)
    # Medias por ciclo para ver si la velocidad va cambiando durante el ensayo
    deriva = [float(w_ss[ciclo == c].mean()) for c in range(1, 11)]
    # Ojo que el corte -q/m queda fuera de lo medido, es una extrapolacion
    resultados['recta'] = dict(m=float(m), q=float(q), R2=float(r2),
                               rmse=float(np.sqrt(np.mean(residuos**2))),
                               max_abs_residuo=float(np.max(np.abs(residuos))),
                               duty_velocidad_nula=float(-q / m),
                               omega_en_dref=float(m * DUTY_REFERENCIA + q),
                               por_nivel=por_nivel,
                               pendientes_locales=[float(v) for v in pendientes_locales],
                               w_ss_media_por_ciclo=deriva)

    fig, ejes = plt.subplots(2, 1, figsize=(3.5, 4.2), sharex=True,
                             gridspec_kw={'height_ratios': [2.3, 1]})
    d_linea = np.linspace(0.59, 0.76, 50)
    ejes[0].plot(d_linea, m * d_linea + q, 'k-', lw=0.9,
                 label=fr'$\omega_{{ss}}$={m:.1f}$d${q:+.1f}')
    sc = ejes[0].scatter(d_tramo, w_ss, c=ciclo, cmap='viridis', s=12, zorder=3)
    ejes[0].set_ylabel(r'$\omega_{ss}$ [rad/s]')
    ejes[0].legend(fontsize=7, loc='upper left')
    cb = fig.colorbar(sc, ax=ejes, shrink=0.6, pad=0.02)
    cb.set_label('ciclo')
    ejes[1].scatter(d_tramo, residuos, c=ciclo, cmap='viridis', s=10)
    ejes[1].axhline(0, color='k', lw=0.8)
    ejes[1].set_ylabel('residuo [rad/s]')
    ejes[1].set_xlabel('duty $d$')
    guardar(fig, carpeta_fig, 'fig_A1c_regimen_permanente')
    return resultados


if __name__ == '__main__':
    datos = np.genfromtxt(archivo_datos(), delimiter=',', names=True, encoding='utf-8-sig')
    tiempo = datos['time_s'] - datos['time_s'][0]
    carpeta = carpeta_resultados(1)
    resultados = ejercicio_1(tiempo, datos['duty_effective'],
                             datos['omega_prom_measured'], carpeta)
    guardar_resultados(1, 'resultados_A1.json', resultados)
