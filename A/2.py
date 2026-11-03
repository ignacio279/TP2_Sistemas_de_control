"""Parte A, ejercicio 2: comprobaciones numéricas del modelo propuesto.
Las comprobaciones de A2 son externas al script de identificación y quedan aquí.
El desarrollo analítico sigue correspondiendo al informe.
Ejecutar primero A/1.py y luego: python A/2.py
Los valores preliminares de este ejercicio no reemplazan la identificación de A5.
"""
from pathlib import Path
import sys
import json

RAIZ = Path(__file__).resolve().parent.parent
CARPETA_RESULTADOS = Path(__file__).resolve().parent / 'resultados'

# La configuración se toma del archivo de la cátedra, sin duplicarla.
sys.path.insert(0, str(RAIZ / 'consigna'))
from script_identificacion import (DUTY_REFERENCIA)


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


def ejercicio_2(resultados_1):
    # En equilibrio el modelo predice w* = w_ref + Kdc (d - d_ref) = Kdc d + (w_ref - Kdc d_ref),
    # por lo que Kdc ~ m y w_ref ~ m d_ref + q. Con p ~ 1/tau se obtienen b = p Kdc y c = p q.
    m, q = resultados_1['recta']['m'], resultados_1['recta']['q']
    p = resultados_1['escalones']['p_preliminar_1_s']
    Kdc, omega_ref = m, m * DUTY_REFERENCIA + q
    return dict(Kdc_desde_recta=Kdc, omega_ref_desde_recta=omega_ref, p_desde_escalones=p,
                b_preliminar=p * Kdc, c_preliminar=p * (omega_ref - Kdc * DUTY_REFERENCIA))


if __name__ == '__main__':
    archivo = carpeta_resultados(1) / 'resultados_A1.json'
    if not archivo.exists():
        raise SystemExit('Primero ejecutar A/1.py para generar resultados_A1.json.')
    with archivo.open(encoding='utf-8') as entrada:
        resultados_1 = json.load(entrada)
    guardar_resultados(2, 'resultados_A2.json', ejercicio_2(resultados_1))
