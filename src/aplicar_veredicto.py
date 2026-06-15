"""Aplica el veredicto de un Juez externo (p. ej. el agente cloud) y congela
la predicción final.

Uso:  python -m src.aplicar_veredicto <match> <delta_home> <delta_away>
          [--confianza alta|media|baja] [--resumen "..."] [--factor "..." ...]

Los deltas se acotan a ±0.25 (mismo tope que el debate local).
"""
import sys
from datetime import datetime, timezone

import numpy as np

from src import blend
from src.agents import debate
from src.agents.schemas import DELTA_MAX, VeredictoJuez
from src.debate_match import DEBATES, build_context


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
        factores=factores, resumen=_opt("--resumen", "veredicto externo"))

    ctx, model, side = build_context(match_number, offline=True)
    P = model.score_matrix(ctx["home"], ctx["away"], adv_side=side,
                           delta_home=dh, delta_away=da)
    ph, pdr, pa = model.outcome_probs(P)
    top = model.top_scores(P, 3)

    market = None
    if ctx.get("odds"):
        o = ctx["odds"]
        market = [o["p_home"], o["p_draw"], o["p_away"]]
    blended, usa_mkt = blend.blend_market([ph, pdr, pa], market)
    bh, bd, ba = (float(x) for x in blended)

    ctx["model_final"] = {"p_home": float(ph), "p_draw": float(pdr), "p_away": float(pa)}
    ctx["blend_weight_market"] = blend.W_MARKET if usa_mkt else 0.0
    ctx["final"] = {
        "p_home": bh, "p_draw": bd, "p_away": ba,
        "score_pred": f"{top[0][0]}-{top[0][1]}",
        "top_scores": "; ".join(f"{i}-{j} ({p:.1%})" for i, j, p in top),
    }

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
    print(f"  Marcadores: {'; '.join(f'{i}-{j} ({p:.1%})' for i, j, p in top)}")
    print(f"  Guardado: {dest}")


if __name__ == "__main__":
    main()
