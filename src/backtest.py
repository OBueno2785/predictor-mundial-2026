"""Backtest del Dixon-Coles sobre fases de grupos de torneos recientes.

Para cada torneo: entrena solo con partidos ANTERIORES a su inicio (sin fuga
de información), predice la fase de grupos y compara contra lo ocurrido.
Ajusta temperature scaling con la muestra agregada y la guarda en
outputs/calibration.json para que src.predict la aplique.

Uso:  py -m src.backtest
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.calibration import metrics, temperature
from src.ingest import results
from src.model import dixon_coles

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"

# (nombre en dataset, inicio grupos, fin grupos, etiqueta)
TOURNAMENTS = [
    ("FIFA World Cup", "2022-11-20", "2022-12-02", "Mundial 2022"),
    ("AFC Asian Cup", "2024-01-12", "2024-01-26", "Copa Asia 2024"),
    ("African Cup of Nations", "2024-01-13", "2024-01-25", "AFCON 2024"),
    ("UEFA Euro", "2024-06-14", "2024-06-27", "Euro 2024"),
    ("Copa América", "2024-06-20", "2024-07-03", "Copa América 2024"),
    ("Gold Cup", "2025-06-14", "2025-06-25", "Copa Oro 2025"),
    ("African Cup of Nations", "2025-12-21", "2026-01-03", "AFCON 2025"),
]

TRAIN_SINCE_YEARS = 8


def backtest_tournament(full: pd.DataFrame, name: str, start: str, end: str):
    start_ts = pd.Timestamp(start)
    matches = full[(full["tournament"] == name)
                   & (full["date"] >= start) & (full["date"] <= end)
                   ].dropna(subset=["home_score", "away_score"])
    if matches.empty:
        return None, 0

    train = full[(full["date"] < start_ts)
                 & (full["date"] >= start_ts - pd.DateOffset(years=TRAIN_SINCE_YEARS))
                 ].dropna(subset=["home_score", "away_score"]).copy()
    train["home_score"] = train["home_score"].astype(int)
    train["away_score"] = train["away_score"].astype(int)
    train["neutral"] = train["neutral"].astype(bool)
    train["weight_importance"] = (train["tournament"] == "Friendly").map(
        {True: results.FRIENDLY_WEIGHT, False: 1.0})
    model = dixon_coles.fit(train, ref_date=start_ts)

    rows = []
    for m in matches.itertuples():
        if m.home_team not in model.attack or m.away_team not in model.attack:
            continue
        side = 0 if m.neutral else 1
        P = model.score_matrix(m.home_team, m.away_team, adv_side=side)
        ph, pdr, pa = model.outcome_probs(P)
        out = 0 if m.home_score > m.away_score else (1 if m.home_score == m.away_score else 2)
        rows.append({"p_home": ph, "p_draw": pdr, "p_away": pa, "outcome": out})
    return pd.DataFrame(rows), len(matches)


def main():
    full = pd.read_csv(DATA / "results.csv", parse_dates=["date"])

    all_rows = []
    print("Backtest por torneo (entrenamiento congelado al inicio de cada uno):")
    for name, start, end, label in TOURNAMENTS:
        df, n_total = backtest_tournament(full, name, start, end)
        if df is None or df.empty:
            print(f"  {label:<20} SIN PARTIDOS — revisar nombre/fechas")
            continue
        probs = df[["p_home", "p_draw", "p_away"]].to_numpy()
        outs = df["outcome"].to_numpy()
        r = metrics.rps(probs, outs)
        acc = metrics.accuracy_1x2(probs, outs)
        print(f"  {label:<20} {len(df):3d}/{n_total} partidos  RPS={r:.4f}  acc 1X2={acc:.0%}")
        df["tournament"] = label
        all_rows.append(df)

    pool = pd.concat(all_rows, ignore_index=True)
    probs = pool[["p_home", "p_draw", "p_away"]].to_numpy()
    outs = pool["outcome"].to_numpy()

    print(f"\nMuestra agregada: {len(pool)} partidos")
    raw = metrics.calibration_report(probs, outs)

    # Baseline climatológico: frecuencias base de la propia muestra
    base = np.bincount(outs, minlength=3) / len(outs)
    clim = np.tile(base, (len(outs), 1))
    print(f"  RPS baseline climatológico: {metrics.rps(clim, outs):.4f}")
    print(f"  RPS modelo crudo:           {raw['rps']:.4f}")

    T = temperature.fit(probs, outs)
    cal = metrics.calibration_report(temperature.apply(probs, T), outs)

    print(f"\nTemperature scaling: T = {T:.3f} "
          f"({'suaviza (sobreconfianza)' if T > 1 else 'agudiza (subconfianza)'})")
    print(f"  {'métrica':<12} {'crudo':>8} {'calibrado':>10}")
    for k in ["rps", "brier", "log_loss", "acc_1x2", "ece_macro",
              "ece_home", "ece_draw", "ece_away"]:
        print(f"  {k:<12} {raw[k]:>8.4f} {cal[k]:>10.4f}")
    print(f"  Spiegelhalter Z (home/draw/away) crudo: "
          f"{raw['z_home']:+.2f} / {raw['z_draw']:+.2f} / {raw['z_away']:+.2f}")
    print(f"  Spiegelhalter Z calibrado:              "
          f"{cal['z_home']:+.2f} / {cal['z_draw']:+.2f} / {cal['z_away']:+.2f}")

    OUT.mkdir(exist_ok=True)
    pool.to_csv(OUT / "backtest_predicciones.csv", index=False)
    (OUT / "calibration.json").write_text(json.dumps({
        "temperature": T,
        "n_matches": len(pool),
        "tournaments": [t[3] for t in TOURNAMENTS],
        "rps_raw": raw["rps"], "rps_cal": cal["rps"],
        "log_loss_raw": raw["log_loss"], "log_loss_cal": cal["log_loss"],
    }, indent=1), encoding="utf-8")
    print(f"\nGuardado: outputs/calibration.json (T={T:.3f}) y outputs/backtest_predicciones.csv")


if __name__ == "__main__":
    main()
