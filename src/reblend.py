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
from src.model import dixon_coles
from src.predict import DATA, OUT


# Puente para debates SIN la bandera explícita del Juez (formato antiguo):
# frases que indican ausencia/indisponibilidad confirmada de un jugador.
_KEYS_BAJAS = ("baja confirmada", "confirmad", "suspend", "excluid", "descartad",
               "denegada", "no jugará", "no jugara", "fuera del mundial",
               "convocatoria", "rotura", "omitid")


def _bajas_confirmadas(d: dict) -> bool:
    """Bandera del Juez si existe (autoritativa); si no (debates antiguos),
    heurística por palabras clave en los factores."""
    v = d.get("veredicto", {})
    if "bajas_confirmadas" in v:
        return bool(v["bajas_confirmadas"])
    texto = " ".join(v.get("factores", [])).lower()
    return any(k in texto for k in _KEYS_BAJAS)


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
    bajas = _bajas_confirmadas(d)
    raw = np.array([mf["p_home"], mf["p_draw"], mf["p_away"]])
    cal = temperature.apply(raw, T)
    pr = d.get("prior")
    prior_cal = temperature.apply(
        np.array([pr["p_home"], pr["p_draw"], pr["p_away"]]), T) if pr else None
    o = d.get("odds")
    if o:
        market = np.array([o["p_home"], o["p_draw"], o["p_away"]])
        fin, _ = blend.blend_final(cal, prior_cal, market, bajas)
        d["blend_weight_market"] = blend.W_MARKET
        d["override_bajas"] = bool(bajas)
    else:
        fin = cal / cal.sum()
        d["blend_weight_market"] = 0.0
    base = d.get("final") or {}
    base.update({"p_home": float(fin[0]), "p_draw": float(fin[1]), "p_away": float(fin[2])})
    # marcador consistente con la prob final; si no hay ritmos guardados,
    # reconstruirlos reentrenando una vez (preserva la transcripción del debate)
    rates = d.get("model_rates")
    if not rates and d.get("delta_aplicado"):
        from src.debate_match import build_context
        ctx, model, side = build_context(match, offline=True)
        da = d["delta_aplicado"]
        lam, mu = model.rates(ctx["home"], ctx["away"], adv_side=side,
                              delta_home=da["home"], delta_away=da["away"])
        rates = {"lam": float(lam), "mu": float(mu), "rho": float(model.rho)}
        d["model_rates"] = rates
    if rates:
        P = dixon_coles.matrix_from_rates(rates["lam"], rates["mu"], rates["rho"])
        base.update(blend.score_summary(blend.rescale_matrix(P, fin), 3))
    d["final"] = base
    d["reblend"] = {"T": T, "w_market": blend.W_MARKET, "override_bajas": bool(bajas)}
    f.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    ov = "  ⚑ override bajas" if bajas else ""
    return (f"  {d['partido']}: final {fin[0]:.1%}/{fin[1]:.1%}/{fin[2]:.1%} "
            f"score {base.get('score_pred','?')}{ov}")


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
