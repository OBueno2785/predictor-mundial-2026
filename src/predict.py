"""Pipeline base: ingesta → Dixon-Coles → predicciones calibradas de grupos.

Uso:  py -m src.predict  [--offline]
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.calibration import temperature
from src.ingest import elo, fixtures, results
from src.model import dixon_coles

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"

HOSTS = {"Mexico", "Canada", "United States"}


def ingest(offline: bool) -> None:
    DATA.mkdir(exist_ok=True)
    sources = [
        (results.download, DATA / "results.csv", "histórico"),
        (fixtures.download, DATA / "fixtures_wc2026.json", "fixtures"),
        (elo.download, DATA / "elo_world.tsv", "elo"),
    ]
    for fn, dest, name in sources:
        if offline and dest.exists():
            print(f"  [offline] {name}: usando caché")
            continue
        try:
            fn(dest)
            print(f"  {name}: descargado -> {dest.name}")
        except Exception as e:
            if dest.exists():
                print(f"  AVISO {name}: fallo de descarga, uso caché ({e})")
            else:
                raise


def build_training(fx: pd.DataFrame) -> pd.DataFrame:
    hist = results.load_training(DATA / "results.csv")
    played = fx[fx["played"]].copy()
    if not played.empty:
        extra = pd.DataFrame({
            "date": played["date_utc"].dt.tz_localize(None).dt.normalize(),
            "home_team": played["home_team"],
            "away_team": played["away_team"],
            "home_score": played["home_score"].astype(int),
            "away_score": played["away_score"].astype(int),
            "neutral": [r.home_team != r.venue_country and r.away_team != r.venue_country
                        for r in played.itertuples()],
            "weight_importance": 1.0,
        })
        hist = pd.concat([hist, extra], ignore_index=True)
        print(f"  {len(extra)} partidos del Mundial ya jugados incorporados al entrenamiento")
    return hist


def adv_side(row) -> int:
    if row.home_team in HOSTS and row.home_team == row.venue_country:
        return 1
    if row.away_team in HOSTS and row.away_team == row.venue_country:
        return -1
    return 0


def predict_groups(model: dixon_coles.DixonColesModel, fx: pd.DataFrame) -> pd.DataFrame:
    groups = fx[fx["group"].str.startswith("Group", na=False)].copy()
    rows = []
    for r in groups.itertuples():
        side = adv_side(r)
        P = model.score_matrix(r.home_team, r.away_team, adv_side=side)
        ph, pd_, pa = model.outcome_probs(P)
        lam, mu = model.rates(r.home_team, r.away_team, adv_side=side)
        top = model.top_scores(P, 3)
        rows.append({
            "match": r.match_number, "group": r.group,
            "date_utc": r.date_utc, "location": r.location,
            "home": r.home_team, "away": r.away_team,
            "xg_home": round(lam, 2), "xg_away": round(mu, 2),
            "p_home": round(ph, 4), "p_draw": round(pd_, 4), "p_away": round(pa, 4),
            "score_pred": f"{top[0][0]}-{top[0][1]}",
            "top_scores": "; ".join(f"{i}-{j} ({p:.1%})" for i, j, p in top),
            "played": r.played,
            "real": f"{int(r.home_score)}-{int(r.away_score)}" if r.played else "",
        })
    return pd.DataFrame(rows).sort_values("match").reset_index(drop=True)


def aplicar_debates(preds: pd.DataFrame) -> pd.DataFrame:
    """Sobrescribe con la predicción final del último debate de cada partido."""
    preds["ajustada"] = False
    debates_dir = OUT / "debates"
    if not debates_dir.exists():
        return preds
    ultimos = {}
    for f in sorted(debates_dir.glob("match_*.json")):  # orden => último gana
        try:
            ultimos[int(f.stem.split("_")[1])] = json.loads(f.read_text(encoding="utf-8"))
        except (ValueError, json.JSONDecodeError, IndexError):
            continue
    n = 0
    for m, data in ultimos.items():
        fin = data.get("final")
        if not fin:
            continue
        if str(data.get("veredicto", {}).get("resumen", "")).startswith("Degradado"):
            continue
        mask = preds["match"] == m
        if not mask.any():
            continue
        preds.loc[mask, ["p_home", "p_draw", "p_away"]] = [
            round(fin["p_home"], 4), round(fin["p_draw"], 4), round(fin["p_away"], 4)]
        if fin.get("score_pred"):
            preds.loc[mask, "score_pred"] = fin["score_pred"]
        if fin.get("top_scores"):
            preds.loc[mask, "top_scores"] = fin["top_scores"]
        preds.loc[mask, "ajustada"] = True
        n += 1
    if n:
        print(f"  {n} partido(s) con predicción final ajustada por el panel")
    return preds


def write_markdown(preds: pd.DataFrame, model: dixon_coles.DixonColesModel) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Predicciones — Fase de Grupos Mundial 2026",
        "",
        f"Generado: {ts} · Modelo: Dixon-Coles ({model.n_matches} partidos de "
        f"entrenamiento, ventaja localía={model.home_adv:.3f}, rho={model.rho:.4f})",
        "",
        "Probabilidades del prior estadístico calibrado. Los partidos marcados con"
        " **✓ panel** muestran la predicción final ajustada por el debate"
        " multiagente + cuotas de mercado (job T-60 min antes del kickoff).",
        "",
    ]
    for g in sorted(preds["group"].unique()):
        lines += [f"## {g}", "",
                  "| # | Fecha (UTC) | Partido | 1 | X | 2 | Pred. | Top marcadores | Real | Panel |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for r in preds[preds["group"] == g].itertuples():
            fecha = r.date_utc.strftime("%d-%m %H:%M")
            panel = "✓ panel" if r.ajustada else "—"
            lines.append(
                f"| {r.match} | {fecha} | {r.home} vs {r.away} | {r.p_home:.0%} | "
                f"{r.p_draw:.0%} | {r.p_away:.0%} | {r.score_pred} | {r.top_scores} | "
                f"{r.real or '—'} | {panel} |")
        lines.append("")
    dest = ROOT / "PREDICCIONES.md"
    dest.write_text("\n".join(lines), encoding="utf-8")
    return dest


def main() -> None:
    offline = "--offline" in sys.argv
    print("[1/4] Ingesta de datos")
    ingest(offline)

    fx = fixtures.load(DATA / "fixtures_wc2026.json")
    hist = build_training(fx)
    known = set(hist["home_team"]) | set(hist["away_team"])
    missing = fixtures.unmatched_teams(fx, known)
    if missing:
        sys.exit(f"ERROR: equipos del fixture sin match en el histórico: {missing}. "
                 "Agregar a NAME_MAP en src/ingest/fixtures.py")

    print(f"[2/4] Entrenando Dixon-Coles con {len(hist)} partidos "
          f"({hist['date'].min().date()} a {hist['date'].max().date()})")
    model = dixon_coles.fit(hist)
    top = model.strength_ranking().head(10)
    print("  Top 10 fuerza (ataque+defensa):")
    for r in top.itertuples():
        print(f"    {r.Index + 1:2d}. {r.team:<15} {r.strength:+.3f}")

    print("[3/4] Prediciendo 72 partidos de fase de grupos")
    preds = predict_groups(model, fx)

    cal_path = OUT / "calibration.json"
    if cal_path.exists():
        T = json.loads(cal_path.read_text())["temperature"]
        probs = temperature.apply(preds[["p_home", "p_draw", "p_away"]].to_numpy(), T)
        preds[["p_home", "p_draw", "p_away"]] = np.round(probs, 4)
        print(f"  Probabilidades 1X2 calibradas (T={T:.3f}, de outputs/calibration.json)")

    preds = aplicar_debates(preds)

    print("[4/4] Escribiendo salidas")
    OUT.mkdir(exist_ok=True)
    csv_path = OUT / "predicciones_grupos.csv"
    preds.to_csv(csv_path, index=False, encoding="utf-8")
    md_path = write_markdown(preds, model)
    print(f"  {csv_path}\n  {md_path}")

    jugados = preds[preds["played"]]
    if not jugados.empty:
        print(f"  Partidos ya jugados: {len(jugados)} (columna 'real' para comparar)")


if __name__ == "__main__":
    main()
