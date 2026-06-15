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


def qualification(fx: pd.DataFrame, team: str) -> dict:
    """Escenarios de top-2 por fuerza bruta sobre los resultados restantes."""
    grp = fx[(fx["home_team"] == team) | (fx["away_team"] == team)]
    grp = grp[grp["group"].str.startswith("Group", na=False)]
    if grp.empty:
        return {}
    group = grp.iloc[0]["group"]
    g = _group_matches(fx, group)

    base = {t: 0 for t in set(g["home_team"]) | set(g["away_team"])}
    for r in g[g["played"]].itertuples():
        hs, as_ = int(r.home_score), int(r.away_score)
        base[r.home_team] += PTS["W" if hs > as_ else ("D" if hs == as_ else "L")]
        base[r.away_team] += PTS["W" if as_ > hs else ("D" if as_ == hs else "L")]

    restantes = list(g[~g["played"]].itertuples())
    # próximo partido del equipo (el más cercano sin jugar)
    prox = next((r for r in sorted(restantes, key=lambda x: x.date_utc)
                 if team in (r.home_team, r.away_team)), None)

    def top2(points, res_equipo_prox):
        """¿el equipo termina top-2? worst-case por empates de puntos."""
        peor = sum(1 for q in points.values() if q >= points[team])  # peor posición
        return peor <= 2

    cond = {"W": [], "D": [], "L": []}
    todos = []
    for combo in product("WDL", repeat=len(restantes)):
        pts = dict(base)
        prox_res = None
        for m, r in zip(combo, restantes):
            if m == "W":
                pts[r.home_team] += 3
            elif m == "D":
                pts[r.home_team] += 1
                pts[r.away_team] += 1
            else:
                pts[r.away_team] += 3
            if prox is not None and r.Index == prox.Index:
                if m == "D":
                    prox_res = "D"
                else:
                    ganador = r.home_team if m == "W" else r.away_team
                    prox_res = "W" if ganador == team else "L"
        peor = sum(1 for q in pts.values() if q >= pts[team])
        es_top2 = peor <= 2
        todos.append(es_top2)
        if prox_res:
            cond[prox_res].append(es_top2)

    n = len(todos)
    asegurado = all(todos)
    imposible = not any(todos)
    rival_prox = None
    if prox is not None:
        rival_prox = prox.away_team if prox.home_team == team else prox.home_team

    def resumen_cond(res):
        v = cond[res]
        if not v:
            return None
        if all(v):
            return "clasifica seguro"
        if any(v):
            return "sigue con opciones"
        return "queda fuera del top-2"

    return {
        "group": group,
        "asegurado": asegurado,
        "imposible_top2": imposible,
        "rival_prox": rival_prox,
        "cond": {k: resumen_cond(k) for k in ("W", "D", "L") if resumen_cond(k)},
        "restantes": len(restantes),
    }


def outlook_text(fx: pd.DataFrame, team: str) -> str:
    q = qualification(fx, team)
    if not q:
        return "sin datos de grupo"
    if q["restantes"] == 0:
        return "clasificado a 32avos (top-2)" if q["asegurado"] else \
            "fuera del top-2 (puede entrar como mejor tercero según otros grupos)"
    if q["asegurado"]:
        return "ya clasificado a 32avos (top-2 asegurado)"
    partes = []
    if q["rival_prox"]:
        etq = {"W": "gana", "D": "empata", "L": "pierde"}
        for res, txt in q["cond"].items():
            partes.append(f"{etq[res]}→{txt}")
    base = f"próximo vs {q['rival_prox']}: " + "; ".join(partes) if partes else ""
    if q["imposible_top2"]:
        return ("sin opción de top-2; solo puede avanzar como mejor tercero "
                "(depende de otros grupos)")
    return base or "en disputa"


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
