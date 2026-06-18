"""Calibración de GOLES: corrige el volumen de goles del modelo para que el
marcador esperado sea realista (clave para acertar marcadores).

Compara los goles esperados (xG del modelo out-of-sample, sin fuga) con los goles
reales de los partidos jugados y fija un multiplicador `goals_mult`, con
CONTRACCIÓN hacia 1.0 según el tamaño de muestra (no sobreajusta pocos partidos)
y tope [0.85, 1.25]. Se guarda en outputs/calibration.json y se aplica al
construir la matriz de marcadores.

Uso:  python -m src.goles
"""
import json
import sys

import numpy as np

from src.ingest import fixtures, results
from src.model import dixon_coles
from src.predict import DATA, OUT, adv_side

WC_START = "2026-06-11"
SHRINK_K = 30      # mitad de peso a las 30 muestras
CAP = (0.85, 1.25)


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    fixtures.download(DATA / "fixtures_wc2026.json")
    fx = fixtures.load(DATA / "fixtures_wc2026.json")
    jug = fx[fx["played"] & fx["group"].str.startswith("Group", na=False)]

    hist = results.load_training(DATA / "results.csv")
    pre = hist[hist["date"] < WC_START]
    model = dixon_coles.fit(pre, ref_date=WC_START)

    pred_tot, real_tot, altos = [], [], 0
    for r in jug.itertuples():
        if r.home_team not in model.attack or r.away_team not in model.attack:
            continue
        lam, mu = model.rates(r.home_team, r.away_team, adv_side=adv_side(r))
        pred_tot.append(lam + mu)
        real = int(r.home_score) + int(r.away_score)
        real_tot.append(real)
        altos += real >= 4

    pred_tot, real_tot = np.array(pred_tot), np.array(real_tot)
    n = len(pred_tot)
    ratio = real_tot.sum() / pred_tot.sum()
    shrink = n / (n + SHRINK_K)
    mult = float(np.clip(1 + shrink * (ratio - 1), *CAP))

    print(f"Partidos evaluados: {n}")
    print(f"  Goles esperados por el modelo (prom): {pred_tot.mean():.2f}")
    print(f"  Goles reales (prom):                  {real_tot.mean():.2f}")
    print(f"  Ratio real/predicho: {ratio:.2f}  ({'predice de MENOS' if ratio > 1.05 else 'predice de MÁS' if ratio < 0.95 else 'OK'})")
    print(f"  Partidos con 4+ goles: {altos}/{n} ({altos / n:.0%})")
    print(f"  Contracción (n={n}, K={SHRINK_K}): {shrink:.2f}")
    print(f"  → goals_mult = ×{mult:.3f}  (ratio crudo ×{ratio:.2f} contraído hacia 1.0)")

    cal = json.loads((OUT / "calibration.json").read_text(encoding="utf-8"))
    cal["goals_mult"] = mult
    cal["goals_n"] = int(n)
    (OUT / "calibration.json").write_text(json.dumps(cal, indent=1), encoding="utf-8")
    print(f"  Escrito en outputs/calibration.json")


if __name__ == "__main__":
    main()
