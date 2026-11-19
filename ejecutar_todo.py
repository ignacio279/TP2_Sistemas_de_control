import os
import platform
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path


def main():
    carpeta_tp = Path(__file__).resolve().parent
    # Primero A5 tiene que guardar los parametros que usan A6 y la Parte B
    archivos = [carpeta_tp / parte / f"{numero}.py"
                for parte in ("A", "B") for numero in range(1, 7)]
    entorno = os.environ.copy()
    entorno["MPLBACKEND"] = "Agg"

    with (carpeta_tp / "ejecucion_completa.txt").open("w", encoding="utf-8") as registro:
        registro.write(f"Python {platform.python_version()}\n")
        for libreria in ("numpy", "scipy", "matplotlib", "control"):
            registro.write(f"{libreria} {version(libreria)}\n")

        for numero, archivo in enumerate(archivos, start=1):
            nombre = str(archivo.relative_to(carpeta_tp))
            print(f"[{numero}/{len(archivos)}] {nombre}", flush=True)
            resultado = subprocess.run(
                [sys.executable, "-E", str(archivo)],
                cwd=archivo.parent,
                env=entorno,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            registro.write(f"\n--- {nombre} ---\n")
            registro.write(resultado.stdout)
            registro.write(f"\nCodigo de salida: {resultado.returncode}\n")
            registro.flush()
            # Si falla un punto, paramos para no seguir con resultados de una corrida anterior
            if resultado.returncode != 0:
                print("ERROR: consultar ejecucion_completa.txt", flush=True)
                return 1

        resumen = f"Completados sin errores: {len(archivos)}/{len(archivos)}"
        registro.write("\n" + resumen + "\n")
    print(resumen)
    print("Resultados de consola: ejecucion_completa.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
