"""Capa de contexto de grupo: forma (últimos 5), tabla de posiciones y
escenarios de clasificación a octavos.

Mundial 2026: 12 grupos de 4. Clasifican los 2 primeros de cada grupo + los
8 mejores terceros (de 12) → 32avos de final.

Los escenarios de top-2 se calculan por fuerza bruta sobre los resultados
restantes del grupo (solo puntos; el desempate real FIFA usa dif. de gol y
goles a favor, que aquí se aproximan con los datos jugados). El paso como
"mejor tercero" depende de los otros grupos y se marca como tal, sin fingir
precisión.
"""
from itertools import product

import pandas as pd

PTS = {"W": 3, "D": 1, "L": 0}


def last5(hist: pd.DataFrame, team: str) -> dict:
    """Promedio de los últimos 5 partidos: puntos/partido, GF y GC por partido."""
    mask = (hist["home_team"] == team) | (hist["away_team"] == team)
    rows = hist[mask].sort_values("date", ascending=False).head(5)
    if rows.empty:
        return {"n": 0, "ppg": 0.0, "gf": 0.0, "ga": 0.0, "racha": "—"}
    pts = gf = ga = 0
    racha = []
    for r in rows.itertuples():
        local = r.home_team == team
        f, c = (r.home_score, r.away_score) if local else (r.away_score, r.home_score)
        res = "W" if f > c else ("D" if f == c else "L")
        pts += PTS[res]
        gf += f
        ga += c
        racha.append(res)
    n = len(rows)
    return {"n": n, "ppg": pts / n, "gf": gf / n, "ga": ga / n,
            "racha": "".join(reversed(racha))}


def _group_matches(fx: pd.DataFrame, group: str) -> pd.DataFrame:
    return fx[fx["group"] == group]


def group_table(fx: pd.DataFrame, group: str) -> pd.DataFrame:
    """Tabla de posiciones del grupo con los partidos ya jugados."""
    g = _group_matches(fx, group)
    teams = sorted(set(g["home_team"]) | set(g["away_team"]))
    stats = {t: {"team": t, "pj": 0, "pts": 0, "gf": 0, "gc": 0} for t in teams}
    for r in g[g["played"]].itertuples():
        hs, as_ = int(r.home_score), int(r.away_score)
        for t, f, c in [(r.home_team, hs, as_), (r.away_team, as_, hs)]:
            stats[t]["pj"] += 1
            stats[t]["gf"] += f
            stats[t]["gc"] += c
            stats[t]["pts"] += PTS["W" if f > c else ("D" if f == c else "L")]
    df = pd.DataFrame(stats.values())
    df["dg"] = df["gf"] - df["gc"]
    df = df.sort_values(["pts", "dg", "gf"], ascending=False).reset_index(drop=True)
    df["pos"] = df.index + 1
    return df[["pos", "team", "pj", "pts", "gf", "gc", "dg"]]


def _ranks(points: dict) -> dict:
    """rank (1=mejor) de cada equipo por puntos; empates comparten posición."""
    out = {}
    for t, p in points.items():
        mejores = sum(1 for q in points.values() if q > p)
        out[t] = mejores + 1  # mejor caso posible del equipo en el bloque empatado
    return out


REPRESENTATIVE_SCORES = [
    # Victorias Locales (15 marcadores)
    (1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2),
    (4, 0), (4, 1), (4, 2), (4, 3),
    (5, 0), (5, 1), (5, 2), (5, 3), (5, 4),
    
    # Empates (5 marcadores)
    (0, 0), (1, 1), (2, 2), (3, 3), (4, 4),
    
    # Victorias Visitantes (15 marcadores)
    (0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3),
    (0, 4), (1, 4), (2, 4), (3, 4),
    (0, 5), (1, 5), (2, 5), (3, 5), (4, 5)
]


def qualification(fx: pd.DataFrame, team: str) -> dict:
    """Calcula escenarios y probabilidad de clasificación para el equipo.
    
    Si quedan <= 2 partidos en el grupo (Ronda 3), usa simulación por marcadores (35 marcadores).
    Si quedan > 2 partidos (Rondas 1 y 2), usa simulación por W/D/L.
    """
    grp = fx[(fx["home_team"] == team) | (fx["away_team"] == team)]
    grp = grp[grp["group"].str.startswith("Group", na=False)]
    if grp.empty:
        return {}
    group = grp.iloc[0]["group"]
    g = _group_matches(fx, group)

    teams = sorted(set(g["home_team"]) | set(g["away_team"]))
    
    # Calcular estadísticas base de partidos jugados
    base_stats = {t: {"pts": 0, "gf": 0, "gc": 0} for t in teams}
    for r in g[g["played"]].itertuples():
        hs, as_ = int(r.home_score), int(r.away_score)
        base_stats[r.home_team]["gf"] += hs
        base_stats[r.home_team]["gc"] += as_
        base_stats[r.home_team]["pts"] += 3 if hs > as_ else (1 if hs == as_ else 0)
        
        base_stats[r.away_team]["gf"] += as_
        base_stats[r.away_team]["gc"] += hs
        base_stats[r.away_team]["pts"] += 3 if as_ > hs else (1 if hs == as_ else 0)

    restantes = list(g[~g["played"]].itertuples())
    n_restantes = len(restantes)
    
    # Próximo partido de nuestro equipo
    prox = next((r for r in sorted(restantes, key=lambda x: x.date_utc)
                 if team in (r.home_team, r.away_team)), None)
    
    rival_prox = None
    if prox is not None:
        rival_prox = prox.away_team if prox.home_team == team else prox.home_team

    # Heurística de clasificación para terceros lugares
    def p_third_place(pts, dg):
        if pts >= 4:
            return 1.0
        elif pts == 3:
            return 0.8 if dg >= 0 else 0.3
        elif pts == 2:
            return 0.05
        else:
            return 0.0

    # Inicializar contadores
    total_prob = 0.0
    count_scenarios = 0
    
    # Para condicionales del próximo partido
    cond_stats = {
        "W2": {"prob_sum": 0.0, "count": 0},
        "W1": {"prob_sum": 0.0, "count": 0},
        "D": {"prob_sum": 0.0, "count": 0},
        "L1": {"prob_sum": 0.0, "count": 0},
        "L2": {"prob_sum": 0.0, "count": 0},
    }

    if n_restantes == 0:
        table_teams = sorted(teams, key=lambda x: (
            base_stats[x]["pts"],
            base_stats[x]["gf"] - base_stats[x]["gc"],
            base_stats[x]["gf"]
        ), reverse=True)
        pos = table_teams.index(team) + 1
        if pos <= 2:
            p_qual = 1.0
        elif pos == 3:
            p_qual = p_third_place(base_stats[team]["pts"], base_stats[team]["gf"] - base_stats[team]["gc"])
        else:
            p_qual = 0.0
        
        return {
            "group": group,
            "prob_clasificar": p_qual,
            "necesidad_goles": "Fase de grupos finalizada." if p_qual > 0 else "Eliminado.",
            "outlook": "Clasificado" if p_qual >= 0.95 else ("Eliminado" if p_qual < 0.05 else "Pendiente de otros grupos"),
            "restantes": 0,
            "rival_prox": None,
            "prob_if_w": p_qual, "prob_if_d": p_qual, "prob_if_l": p_qual,
            "prob_if_w2": p_qual, "prob_if_w1": p_qual, "prob_if_l1": p_qual, "prob_if_l2": p_qual,
            "asegurado": p_qual >= 0.99,
            "imposible_top2": p_qual < 0.01
        }

    elif n_restantes <= 3:
        # Simulación detallada por marcadores
        for combo in product(REPRESENTATIVE_SCORES, repeat=n_restantes):
            stats = {t: dict(s) for t, s in base_stats.items()}
            
            for score, r in zip(combo, restantes):
                hs, as_ = score
                stats[r.home_team]["gf"] += hs
                stats[r.home_team]["gc"] += as_
                stats[r.home_team]["pts"] += 3 if hs > as_ else (1 if hs == as_ else 0)
                
                stats[r.away_team]["gf"] += as_
                stats[r.away_team]["gc"] += hs
                stats[r.away_team]["pts"] += 3 if as_ > hs else (1 if hs == as_ else 0)
            
            table_teams = sorted(teams, key=lambda x: (
                stats[x]["pts"],
                stats[x]["gf"] - stats[x]["gc"],
                stats[x]["gf"]
            ), reverse=True)
            
            pos = table_teams.index(team) + 1
            if pos <= 2:
                p_qual = 1.0
            elif pos == 3:
                p_qual = p_third_place(stats[team]["pts"], stats[team]["gf"] - stats[team]["gc"])
            else:
                p_qual = 0.0
            
            total_prob += p_qual
            count_scenarios += 1
            
            if prox is not None:
                prox_idx = restantes.index(prox)
                prox_score = combo[prox_idx]
                hs, as_ = prox_score
                is_home = (prox.home_team == team)
                our_goals = hs if is_home else as_
                opp_goals = as_ if is_home else hs
                diff = our_goals - opp_goals
                
                if diff >= 2:
                    cat = "W2"
                elif diff == 1:
                    cat = "W1"
                elif diff == 0:
                    cat = "D"
                elif diff == -1:
                    cat = "L1"
                else:
                    cat = "L2"
                
                cond_stats[cat]["prob_sum"] += p_qual
                cond_stats[cat]["count"] += 1

    else:
        # Simulación simplificada por W/D/L
        for combo in product("WDL", repeat=n_restantes):
            stats = {t: dict(s) for t, s in base_stats.items()}
            
            for outcome, r in zip(combo, restantes):
                if outcome == "W":
                    stats[r.home_team]["pts"] += 3
                elif outcome == "D":
                    stats[r.home_team]["pts"] += 1
                    stats[r.away_team]["pts"] += 1
                else:
                    stats[r.away_team]["pts"] += 3
            
            table_teams = sorted(teams, key=lambda x: (
                stats[x]["pts"],
                stats[x]["gf"] - stats[x]["gc"],
                stats[x]["gf"]
            ), reverse=True)
            
            pos = table_teams.index(team) + 1
            if pos <= 2:
                p_qual = 1.0
            elif pos == 3:
                p_qual = p_third_place(stats[team]["pts"], stats[team]["gf"] - stats[team]["gc"])
            else:
                p_qual = 0.0
                
            total_prob += p_qual
            count_scenarios += 1
            
            if prox is not None:
                prox_idx = restantes.index(prox)
                prox_outcome = combo[prox_idx]
                is_home = (prox.home_team == team)
                if prox_outcome == "D":
                    cat = "D"
                elif (prox_outcome == "W" and is_home) or (prox_outcome == "L" and not is_home):
                    cat = "W1"
                else:
                    cat = "L1"
                
                cond_stats[cat]["prob_sum"] += p_qual
                cond_stats[cat]["count"] += 1
                
                # Mapear a W2/L2 también para mantener coherencia en las llaves
                if cat == "W1":
                    cond_stats["W2"]["prob_sum"] += p_qual
                    cond_stats["W2"]["count"] += 1
                elif cat == "L1":
                    cond_stats["L2"]["prob_sum"] += p_qual
                    cond_stats["L2"]["count"] += 1

    prob_clasificar = total_prob / count_scenarios if count_scenarios > 0 else 0.0
    
    p_w2 = cond_stats["W2"]["prob_sum"] / cond_stats["W2"]["count"] if cond_stats["W2"]["count"] > 0 else prob_clasificar
    p_w1 = cond_stats["W1"]["prob_sum"] / cond_stats["W1"]["count"] if cond_stats["W1"]["count"] > 0 else prob_clasificar
    p_d  = cond_stats["D"]["prob_sum"]  / cond_stats["D"]["count"]  if cond_stats["D"]["count"]  > 0 else prob_clasificar
    p_l1 = cond_stats["L1"]["prob_sum"] / cond_stats["L1"]["count"] if cond_stats["L1"]["count"] > 0 else prob_clasificar
    p_l2 = cond_stats["L2"]["prob_sum"] / cond_stats["L2"]["count"] if cond_stats["L2"]["count"] > 0 else prob_clasificar
    
    if n_restantes <= 3:
        if p_w2 > 0.95 and p_w1 > 0.95 and p_d > 0.95:
            desc = "Clasificación asegurada (el empate o la victoria lo clasifican)."
        elif p_w2 > 0.95 and p_w1 > 0.95 and p_d > 0.5:
            desc = f"Un empate le basta ({p_d:.0%}); ganar asegura el pase."
        elif p_w2 > 0.95 and p_w1 > 0.95 and p_d <= 0.5:
            desc = f"Obligado a ganar. El empate le da pocas opciones ({p_d:.0%}); perder lo elimina."
        elif p_w2 > 0.95 and p_w1 <= 0.95:
            desc = f"Necesita ganar, preferiblemente por 2+ goles ({p_w2:.0%} de pase vs {p_w1:.0%} si gana por 1)."
        elif p_w2 > 0.5 and p_w2 > p_w1 + 0.15:
            desc = f"Obligado a ganar por diferencia de goles. Ganar por 2+ goles le da {p_w2:.0%} de pase; ganar por 1 gol solo {p_w1:.0%}."
        elif p_w2 <= 0.2:
            desc = f"Situación muy crítica. Incluso ganando por 2+ goles sus opciones son bajas ({p_w2:.0%})."
        else:
            desc = f"Ganar le da {p_w1:.0%}-{p_w2:.0%} de opciones. El empate da {p_d:.0%}."
            
        if p_l1 > 0.05:
            desc += f" Si pierde por 1 gol, aún conserva {p_l1:.0%} de opciones como mejor tercero."
        else:
            desc += " Perder lo elimina prácticamente."
    else:
        desc = f"Fase temprana del torneo. Probabilidad actual de clasificación: {prob_clasificar:.0%}."
        if p_w1 > prob_clasificar + 0.1:
            desc += " El próximo partido es muy importante para encaminar el pase."

    return {
        "group": group,
        "prob_clasificar": prob_clasificar,
        "necesidad_goles": desc,
        "restantes": n_restantes,
        "rival_prox": rival_prox,
        "prob_if_w": (p_w1 + p_w2) / 2.0,
        "prob_if_d": p_d,
        "prob_if_l": (p_l1 + p_l2) / 2.0,
        "prob_if_w2": p_w2,
        "prob_if_w1": p_w1,
        "prob_if_l1": p_l1,
        "prob_if_l2": p_l2,
        "asegurado": prob_clasificar >= 0.99,
        "imposible_top2": p_w2 < 0.01
    }


def outlook_text(fx: pd.DataFrame, team: str) -> str:
    q = qualification(fx, team)
    if not q:
        return "sin datos de grupo"
    if q["restantes"] == 0:
        return q.get("outlook", "grupo terminado")
    prob = q.get("prob_clasificar", 0.0)
    desc = q.get("necesidad_goles", "")
    return f"Prob: {prob:.0%} | {desc}"


def standings_markdown(fx: pd.DataFrame) -> str:
    grupos = sorted(g for g in fx["group"].dropna().unique() if g.startswith("Group"))
    lines = ["# Tabla de posiciones y escenarios — Mundial 2026", "",
             "Clasifican los 2 primeros de cada grupo + los 8 mejores terceros.", ""]
    for grp in grupos:
        t = group_table(fx, grp)
        lines += [f"## {grp}", "",
                  "| Pos | Equipo | PJ | Pts | GF | GC | DG | Escenario |",
                  "|---|---|---|---|---|---|---|---|"]
        for r in t.itertuples():
            out = outlook_text(fx, r.team)
            marca = "**" if r.pos <= 2 else ""
            lines.append(f"| {r.pos} | {marca}{r.team}{marca} | {r.pj} | {r.pts} | "
                         f"{r.gf} | {r.gc} | {r.dg:+d} | {out} |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    import sys
    from pathlib import Path
    from src.ingest import fixtures
    from src.predict import DATA, ROOT
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    fixtures.download(DATA / "fixtures_wc2026.json")
    fx = fixtures.load(DATA / "fixtures_wc2026.json")
    dest = ROOT / "POSICIONES.md"
    dest.write_text(standings_markdown(fx), encoding="utf-8")
    print(f"Tabla escrita en {dest}")


if __name__ == "__main__":
    main()
