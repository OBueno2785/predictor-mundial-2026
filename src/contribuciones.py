"""¿Cuánto pesa cada capa (prior, debate, mercado) en la predicción final?

Descomposición aditiva exacta de la probabilidad final:
    final = prior_cal + debate_term + market_term
con
    prior_cal    = temperature(prior, T)
    model_cal    = temperature(modelo+debate, T)
    debate_term  = (1-w)·(model_cal − prior_cal)      # lo que el debate mueve el final
    market_term  =    w ·(mercado   − prior_cal)      # lo que el mercado mueve el final
(w = peso de mercado). El debate vive en el canal del modelo, que pesa (1−w) en
la mezcla: ese es su techo estructural. Aquí se mide cuánto mueve DE VERDAD.

Uso:  python -m src.contribuciones
"""
import json
import sys

import numpy as np

from src import blend
from src.calibration import temperature
from src.predict import OUT


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    T = json.loads((OUT / "calibration.json").read_text(encoding="utf-8"))["temperature"]
    w = blend.W_MARKET  # peso vigente, aplicado a todos para ver el régimen ACTUAL

    por_match = {}
    for f in sorted((OUT / "debates").glob("match_*.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        if str(d.get("veredicto", {}).get("resumen", "")).startswith("Degradado"):
            continue
        if not (d.get("prior") and d.get("model_final") and d.get("odds")):
            continue
        por_match[int(f.stem.split("_")[1])] = d  # último por match gana

    if not por_match:
        sys.exit("Sin debates con las 3 señales.")

    filas = []
    for m, d in por_match.items():
        prior = np.array([d["prior"]["p_home"], d["prior"]["p_draw"], d["prior"]["p_away"]])
        mf = np.array([d["model_final"]["p_home"], d["model_final"]["p_draw"], d["model_final"]["p_away"]])
        o = d["odds"]
        market = np.array([o["p_home"], o["p_draw"], o["p_away"]])
        prior_cal = temperature.apply(prior, T)
        model_cal = temperature.apply(mf, T)
        debate_term = (1 - w) * (model_cal - prior_cal)
        market_term = w * (market - prior_cal)
        filas.append({
            "match": m, "partido": d["partido"], "w": w,
            "debate_pp": float(np.abs(debate_term).sum() / 2 * 100),   # pp movidos (L1/2)
            "market_pp": float(np.abs(market_term).sum() / 2 * 100),
        })

    deb = np.mean([r["debate_pp"] for r in filas])
    mkt = np.mean([r["market_pp"] for r in filas])
    activos = [r for r in filas if r["debate_pp"] > 0.5]

    print(f"Peso de mercado actual w = {w:.2f}  →  canal del modelo (1−w) = {1 - w:.2f}")
    print(f"  Techo ESTRUCTURAL del debate en el final: {(1 - w) * 100:.0f}%")
    print(f"  (todo el debate ocurre dentro de ese {(1 - w) * 100:.0f}%)\n")

    print(f"Movimiento REAL sobre el 1X2 final (promedio de {len(filas)} partidos con debate):")
    print(f"  Debate  : {deb:.2f} pp")
    print(f"  Mercado : {mkt:.2f} pp")
    tot = deb + mkt
    if tot > 0:
        print(f"  Peso relativo del debate vs mercado: {deb / tot:.0%} debate / {mkt / tot:.0%} mercado")
    print(f"  Partidos donde el debate movió algo (>0.5pp): {len(activos)}/{len(filas)}")
    if activos:
        top = max(activos, key=lambda r: r["debate_pp"])
        print(f"  Mayor efecto del debate: {top['partido']} ({top['debate_pp']:.1f} pp)\n")

    print(f"  {'#':<4}{'partido':<34}{'debate(pp)':>11}{'mercado(pp)':>12}")
    for r in sorted(filas, key=lambda r: -r["debate_pp"]):
        print(f"  {r['match']:<4}{r['partido']:<34}{r['debate_pp']:>11.2f}{r['market_pp']:>12.2f}")


if __name__ == "__main__":
    main()
