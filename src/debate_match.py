"""Pipeline completo para UN partido: prior → cuotas → debate → predicción final.

Uso:  py -m src.debate_match <match_number> [--rounds 2] [--offline]
Ejemplo:  py -m src.debate_match 7

Por defecto 1 ronda de debate (cuota de suscripción); --rounds 2 activa las
réplicas. Si el debate falla (p. ej. límite de sesión), degrada al prior.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from src import blend, groups
from src.agents import debate
from src.calibration import temperature
from src.ingest import fixtures, odds, results
from src.model import dixon_coles
from src.predict import DATA, OUT, adv_side, build_training, ingest

DEBATES = OUT / "debates"


def build_context(match_number: int, offline: bool):
    ingest(offline)
    fx = fixtures.load(DATA / "fixtures_wc2026.json")
    row = fx[fx["match_number"] == match_number]
    if row.empty:
        sys.exit(f"No existe el partido {match_number}")
    m = row.iloc[0]
    if m["played"]:
        print(f"AVISO: el partido ya se jugó ({int(m['home_score'])}-{int(m['away_score'])})")

    hist = build_training(fx)
    model = dixon_coles.fit(hist)

    side = adv_side(m)
    P = model.score_matrix(m["home_team"], m["away_team"], adv_side=side)
    ph, pdr, pa = model.outcome_probs(P)
    lam, mu = model.rates(m["home_team"], m["away_team"], adv_side=side)
    top = model.top_scores(P, 3)

    cuotas = None
    try:
        events = odds.fetch_all()
        if events:
            cuotas = odds.implied_probs(events, m["home_team"], m["away_team"])
            print(f"  Cuotas: {'OK (' + str(cuotas['n_bookies']) + ' casas)' if cuotas else 'partido no encontrado en el feed'}")
        else:
            print("  Cuotas: sin ODDS_API_KEY — el Agente Mercado se omite")
    except Exception as e:
        print(f"  Cuotas: error ({e}) — el Agente Mercado se omite")

    tabla = groups.group_table(fx, m["group"])
    def _pos(team):
        r = tabla[tabla["team"] == team]
        return f"{int(r['pos'].iloc[0])}º, {int(r['pts'].iloc[0])} pts, DG {int(r['dg'].iloc[0]):+d}" \
            if not r.empty else "—"
    def _f5(team):
        f = groups.last5(hist, team)
        return f"{f['racha']} · {f['ppg']:.2f} pts/p · {f['gf']:.1f} GF / {f['ga']:.1f} GC por partido"

    ventaja = m["home_team"] if side == 1 else (m["away_team"] if side == -1 else None)
    ctx = {
        "match_number": match_number,
        "home": m["home_team"], "away": m["away_team"],
        "group": m["group"], "fecha": m["date_utc"],
        "estadio": m["location"], "sede_pais": m["venue_country"],
        "ventaja": ventaja,
        "xg_home": float(lam), "xg_away": float(mu),
        "p_home": float(ph), "p_draw": float(pdr), "p_away": float(pa),
        "top_scores": "; ".join(f"{i}-{j} ({p:.1%})" for i, j, p in top),
        "form_home": results.recent_form(hist, m["home_team"]),
        "form_away": results.recent_form(hist, m["away_team"]),
        "form5_home": _f5(m["home_team"]), "form5_away": _f5(m["away_team"]),
        "pos_home": _pos(m["home_team"]), "pos_away": _pos(m["away_team"]),
        "outlook_home": groups.outlook_text(fx, m["home_team"]),
        "outlook_away": groups.outlook_text(fx, m["away_team"]),
        "odds": cuotas,
    }
    return ctx, model, side


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        sys.exit("Uso: py -m src.debate_match <match_number> [--rounds 1] [--offline]")
    match_number = int(args[0])
    rounds = 2 if "--rounds" in sys.argv \
        and sys.argv[sys.argv.index("--rounds") + 1] == "2" else 1
    offline = "--offline" in sys.argv

    print(f"[1/3] Construyendo contexto del partido {match_number}")
    ctx, model, side = build_context(match_number, offline)
    print(f"  {ctx['home']} vs {ctx['away']} · prior: "
          f"{ctx['p_home']:.0%}/{ctx['p_draw']:.0%}/{ctx['p_away']:.0%}")

    print(f"[2/3] Debate multiagente ({rounds} ronda(s), agentes={debate.MODEL}, "
          f"juez={debate.MODEL_JUEZ})")
    degraded = None
    try:
        resultado = debate.run_debate(ctx, rounds=rounds)
    except debate.CuotaAgotada as e:
        degraded = f"cuota agotada: {e}"
    except Exception as e:
        degraded = f"error en debate: {e}"
    if degraded:
        print(f"  AVISO: {degraded} — se usa el prior sin ajuste (modo degradado)")
        from src.agents.schemas import VeredictoJuez
        resultado = debate.ResultadoDebate(
            veredicto=VeredictoJuez(delta_log_xg_home=0, delta_log_xg_away=0,
                                    confianza="baja", factores=[],
                                    resumen=f"Degradado: {degraded}"),
            delta_home=0.0, delta_away=0.0)

    print("[3/3] Predicción final: modelo+debate, mezclado con mercado")
    import numpy as np
    lam, mu = model.rates(ctx["home"], ctx["away"], adv_side=side,
                          delta_home=resultado.delta_home, delta_away=resultado.delta_away)
    P = dixon_coles.matrix_from_rates(lam, mu, model.rho)
    ph, pdr, pa = model.outcome_probs(P)   # modelo+debate, crudo (sin temperatura)

    # Calibración (temperature) sobre el modelo+debate, luego mezcla con mercado.
    # model_final se guarda CRUDO; la T se aplica al consumirlo (consistente con
    # fit_market_weight y reblend).
    cal = json.loads((OUT / "calibration.json").read_text(encoding="utf-8"))
    T = cal["temperature"]
    g = cal.get("goals_mult", 1.0)
    mf_cal = temperature.apply(np.array([ph, pdr, pa]), T)
    prior_cal = temperature.apply(np.array([ctx["p_home"], ctx["p_draw"], ctx["p_away"]]), T)
    bajas = bool(getattr(resultado.veredicto, "bajas_confirmadas", False))
    market = None
    if ctx.get("odds"):
        o = ctx["odds"]
        market = [o["p_home"], o["p_draw"], o["p_away"]]
    blended, usa_mkt, override_ap = blend.blend_final(mf_cal, prior_cal, market, bajas)
    bh, bd, ba = (float(x) for x in blended)
    ch, cd, ca = (float(x) for x in mf_cal)

    # El marcador usa la matriz con calibración de goles (×g) reescalada a la
    # prob FINAL (debate + mercado): el 1X2 no cambia, solo el reparto de goles.
    P_score = dixon_coles.matrix_from_rates(lam * g, mu * g, model.rho)
    P_final = blend.rescale_matrix(P_score, [bh, bd, ba])
    resumen = blend.score_summary(P_final, 3)

    DEBATES.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M")
    dest = DEBATES / f"match_{match_number}_{ts}.json"
    ctx_save = dict(ctx)
    ctx_save["model_final"] = {"p_home": float(ph), "p_draw": float(pdr), "p_away": float(pa)}
    ctx_save["model_rates"] = {"lam": float(lam), "mu": float(mu), "rho": float(model.rho)}
    ctx_save["blend_weight_market"] = blend.W_MARKET if usa_mkt else 0.0
    ctx_save["override_bajas"] = bool(override_ap)
    ctx_save["final"] = {"p_home": bh, "p_draw": bd, "p_away": ba, **resumen}
    debate.guardar(resultado, ctx_save, dest)

    v = resultado.veredicto
    print(f"  Veredicto: delta log-xG home={resultado.delta_home:+.3f} "
          f"away={resultado.delta_away:+.3f} (confianza {v.confianza})")
    for f in v.factores:
        print(f"    - {f}")
    print(f"  {v.resumen}")
    print(f"  Prior modelo:    {ctx['p_home']:.1%} / {ctx['p_draw']:.1%} / {ctx['p_away']:.1%}")
    print(f"  Modelo+debate:   {ch:.1%} / {cd:.1%} / {ca:.1%}")
    if usa_mkt:
        if override_ap:
            ov = f"  ⚑ OVERRIDE bajas no precificadas (debate pesa {blend.W_DEBATE_OVERRIDE:.0%})"
        elif bajas:
            ov = "  (bajas ya precificadas por el mercado → sin override)"
        else:
            ov = ""
        print(f"  Mercado (24c):   {market[0]:.1%} / {market[1]:.1%} / {market[2]:.1%}")
        print(f"  FINAL (mezcla):  {bh:.1%} / {bd:.1%} / {ba:.1%}  "
              f"(peso mercado {blend.W_MARKET:.0%}){ov}")
    else:
        print(f"  FINAL:           {bh:.1%} / {bd:.1%} / {ba:.1%}  (sin cuotas)")
    print(f"  Marcadores: {resumen['top_scores']}")
    print(f"  xG final {resumen['xg_home']:.2f}-{resumen['xg_away']:.2f} · "
          f"gana local por 2+: {resumen['p_home_by2']:.0%} · "
          f"visita por 2+: {resumen['p_away_by2']:.0%}")
    u = resultado.usage
    print(f"  {u.get('llamadas', 0)} llamadas a claude · "
          f"{u.get('duracion_ms', 0) / 60000:.1f} min acumulados de agentes")
    print(f"  Guardado: {dest}")


if __name__ == "__main__":
    main()
