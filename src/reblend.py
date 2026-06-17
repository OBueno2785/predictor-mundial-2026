"""Recalcula la mezcla final de debates ya hechos con la T y el peso de mercado
vigentes, SIN volver a llamar al LLM ni reentrenar: reusa el model_final (crudo)
y las cuotas guardadas en cada JSON, y preserva el veredicto y la transcripción.

Útil tras correr src.recalibrate o src.fit_market_weight: actualiza las
predicciones congeladas a los nuevos parámetros.

Uso:  python -m src.reblend <match> [<match> ...]
      python -m src.reblend            (todos los partidos NO jugados con debate)
"""
import json
import sys

import numpy as np

from src import blend
from src.calibration import temperature
from src.ingest import fixtures
from src.predict import DATA, OUT


def _latest(match: int):
    fs = sorted((OUT / "debates").glob(f"match_{match}_*.json"))
    for f in reversed(fs):
        d = json.loads(f.read_text(encoding="utf-8"))
        if not str(d.get("veredicto", {}).get("resumen", "")).startswith("Degradado"):
            return f, d
    return (fs[-1], json.loads(fs[-1].read_text(encoding="utf-8"))) if fs else (None, None)


def reblend(match: int, T: float, w: float) -> str | None:
    f, d = _latest(match)
    if d is None:
        return None
    mf = d.get("model_final")
    if not mf:
        return f"  match {match}: sin model_final (debate previo al formato), omitido"
    raw = np.array([mf["p_home"], mf["p_draw"], mf["p_away"]])
    cal = temperature.apply(raw, T)
    o = d.get("odds")
    if o:
        market = np.array([o["p_home"], o["p_draw"], o["p_away"]])
        fin = blend.linear_pool([cal, market], [1 - w, w])
        d["blend_weight_market"] = w
    else:
        fin = cal / cal.sum()
        d["blend_weight_market"] = 0.0
    base = d.get("final") or {}
    base.update({"p_home": float(fin[0]), "p_draw": float(fin[1]), "p_away": float(fin[2])})
    d["final"] = base
    d["reblend"] = {"T": T, "w_market": w}
    f.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    return (f"  {d['partido']}: final {fin[0]:.1%}/{fin[1]:.1%}/{fin[2]:.1%} "
            f"(T={T:.3f}, w={w:.2f})")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    T = json.loads((OUT / "calibration.json").read_text(encoding="utf-8"))["temperature"]
    w = blend.W_MARKET

    args = [int(a) for a in sys.argv[1:] if a.isdigit()]
    if not args:
        fixtures.download(DATA / "fixtures_wc2026.json")
        fx = fixtures.load(DATA / "fixtures_wc2026.json")
        hechos = {int(f.stem.split("_")[1]) for f in (OUT / "debates").glob("match_*.json")}
        pend = set(fx[~fx["played"]]["match_number"].astype(int))
        args = sorted(hechos & pend)

    print(f"Re-mezclando {len(args)} partido(s) con T={T:.3f}, w_market={w:.2f}:")
    for m in args:
        msg = reblend(m, T, w)
        if msg:
            print(msg)


if __name__ == "__main__":
    main()
