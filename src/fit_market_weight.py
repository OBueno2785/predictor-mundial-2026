"""Punto 1 del plan: fija el peso de mercado por datos.

Sobre los partidos jugados que tienen las tres señales (modelo+debate, mercado,
resultado), busca el peso w que minimiza el RPS de la mezcla
final = (1-w)·modelo + w·mercado. Aplica un TOPE de movimiento por corrida
(±0.15) para no sobreajustar a una muestra chica. Escribe outputs/blend_config.json.

Uso:  python -m src.fit_market_weight
"""
import json
import sys

import numpy as np

from src import blend
from src.calibration import metrics, temperature
from src.ingest import fixtures
from src.predict import DATA, OUT

CAP = 0.15  # máximo cambio de w por corrida


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    fixtures.download(DATA / "fixtures_wc2026.json")
    fx = fixtures.load(DATA / "fixtures_wc2026.json")
    jugados = {int(r.match_number): r for r in fx[fx["played"]].itertuples()}

    T = json.loads((OUT / "calibration.json").read_text(encoding="utf-8"))["temperature"]

    # último JSON por match gana (re-runs sobreescriben); solo partidos jugados
    # con cuotas y debate válido
    by_match = {}
    deb = OUT / "debates"
    for f in sorted(deb.glob("match_*.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        m = int(f.stem.split("_")[1])
        if m not in jugados or not d.get("odds"):
            continue
        if str(d.get("veredicto", {}).get("resumen", "")).startswith("Degradado"):
            continue
        mf = d.get("model_final") or d.get("prior")
        if not mf:
            continue
        r = jugados[m]
        out = 0 if r.home_score > r.away_score else (1 if r.home_score == r.away_score else 2)
        by_match[m] = (
            temperature.apply(np.array([mf["p_home"], mf["p_draw"], mf["p_away"]]), T),
            np.array([d["odds"]["p_home"], d["odds"]["p_draw"], d["odds"]["p_away"]]),
            out,
        )

    if len(by_match) < 5:
        sys.exit(f"Muy pocos partidos con las 3 señales ({len(by_match)}); no ajusto.")

    M = np.array([v[0] for v in by_match.values()])
    Q = np.array([v[1] for v in by_match.values()])
    y = np.array([v[2] for v in by_match.values()])

    ws = np.linspace(0, 1, 101)
    rps_by_w = [metrics.rps((1 - w) * M + w * Q, y) for w in ws]
    w_opt = float(ws[int(np.argmin(rps_by_w))])

    w_old = blend.W_MARKET
    w_new = float(np.clip(w_opt, w_old - CAP, w_old + CAP))
    w_new = round(min(max(w_new, 0.0), 0.95), 2)

    print(f"Partidos con las 3 señales: {len(by_match)}")
    print(f"  RPS modelo (w=0):  {metrics.rps(M, y):.4f}")
    print(f"  RPS mercado (w=1): {metrics.rps(Q, y):.4f}")
    print(f"  w óptimo en muestra: {w_opt:.2f}  (RPS {min(rps_by_w):.4f})")
    print(f"  w actual: {w_old:.2f}  ->  w nuevo (con tope ±{CAP}): {w_new:.2f}")

    (OUT / "blend_config.json").write_text(json.dumps({
        "w_market": w_new,
        "w_optimo_muestra": round(w_opt, 2),
        "n_partidos": len(by_match),
        "tope_por_corrida": CAP,
    }, indent=1), encoding="utf-8")
    print(f"  Escrito outputs/blend_config.json (w_market={w_new})")


if __name__ == "__main__":
    main()
