"""Aplica el veredicto de un Juez externo (p. ej. el agente cloud) y congela
la predicción final.

Uso:  python -m src.aplicar_veredicto <match> <delta_home> <delta_away>
          [--confianza alta|media|baja] [--resumen "..."] [--factor "..." ...]

Los deltas se acotan a ±0.25 (mismo tope que el debate local).
"""
import json
import sys
from datetime import datetime, timezone

import numpy as np

from src import blend
from src.agents import debate
from src.agents.schemas import DELTA_MAX, VeredictoJuez
from src.calibration import temperature
from src.debate_match import DEBATES, build_context
from src.predict import OUT

OUT_CAL = OUT / "calibration.json"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) < 3:
        sys.exit("Uso: py -m src.aplicar_veredicto <match> <delta_home> <delta_away>")
    match_number = int(args[0])
    dh = float(np.clip(float(args[1]), -DELTA_MAX, DELTA_MAX))
    da = float(np.clip(float(args[2]), -DELTA_MAX, DELTA_MAX))

    def _opt(flag, default=""):
        return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default

    factores = [sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == "--factor"]
    veredicto = VeredictoJuez(
        delta_log_xg_home=dh, delta_log_xg_away=da,
        confianza=_opt("--confianza", "media"),
        bajas_confirmadas="--bajas" in sys.argv,
        factores=factores, resumen=_opt("--resumen", "veredicto externo"))

    ctx, model, side = build_context(match_number, offline=True)
    lam, mu = model.rates(ctx["home"], ctx["away"], adv_side=side,
                          delta_home=dh, delta_away=da)
    P = dixon_coles.matrix_from_rates(lam, mu, model.rho)
    ph, pdr, pa = model.outcome_probs(P)

    cal = json.loads(OUT_CAL.read_text(encoding="utf-8"))
    T, g = cal["temperature"], cal.get("goals_mult", 1.0)
    mf_cal = temperature.apply(np.array([ph, pdr, pa]), T)
    prior_cal = temperature.apply(np.array([ctx["p_home"], ctx["p_draw"], ctx["p_away"]]), T)
    market = None
    if ctx.get("odds"):
        o = ctx["odds"]
        market = [o["p_home"], o["p_draw"], o["p_away"]]
    blended, usa_mkt = blend.blend_final(mf_cal, prior_cal, market, veredicto.bajas_confirmadas)
    bh, bd, ba = (float(x) for x in blended)
    P_score = dixon_coles.matrix_from_rates(lam * g, mu * g, model.rho)
    resumen = blend.score_summary(blend.rescale_matrix(P_score, [bh, bd, ba]), 3)

    ctx["model_final"] = {"p_home": float(ph), "p_draw": float(pdr), "p_away": float(pa)}
    ctx["model_rates"] = {"lam": float(lam), "mu": float(mu), "rho": float(model.rho)}
    ctx["blend_weight_market"] = blend.W_MARKET if usa_mkt else 0.0
    ctx["override_bajas"] = bool(veredicto.bajas_confirmadas and usa_mkt)
    ctx["final"] = {"p_home": bh, "p_draw": bd, "p_away": ba, **resumen}

    resultado = debate.ResultadoDebate(veredicto=veredicto, delta_home=dh, delta_away=da)
    DEBATES.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M")
    dest = DEBATES / f"match_{match_number}_{ts}.json"
    debate.guardar(resultado, ctx, dest)

    print(f"{ctx['home']} vs {ctx['away']}")
    print(f"  Prior:         {ctx['p_home']:.1%} / {ctx['p_draw']:.1%} / {ctx['p_away']:.1%}")
    print(f"  Modelo+debate: {ph:.1%} / {pdr:.1%} / {pa:.1%}  (deltas {dh:+.3f}/{da:+.3f})")
    if usa_mkt:
        print(f"  Mercado:       {market[0]:.1%} / {market[1]:.1%} / {market[2]:.1%}")
    print(f"  FINAL:         {bh:.1%} / {bd:.1%} / {ba:.1%}")
    print(f"  Marcadores: {resumen['top_scores']}  · xG {resumen['xg_home']:.2f}-"
          f"{resumen['xg_away']:.2f} · por 2+: {resumen['p_home_by2']:.0%}/{resumen['p_away_by2']:.0%}")
    print(f"  Guardado: {dest}")


if __name__ == "__main__":
    main()
