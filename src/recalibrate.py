"""Punto 3 del plan: recalibra la temperatura con datos del Mundial + backtest.

Combina los priors out-of-sample del backtest (outputs/backtest_predicciones.csv)
con los priors out-of-sample de los partidos del Mundial ya jugados (modelo
entrenado SOLO con datos previos al torneo → sin fuga) y reajusta T por log-loss
sobre el conjunto agrupado. Escribe outputs/calibration.json.

Uso:  python -m src.recalibrate
"""
import json
import sys

import numpy as np
import pandas as pd

from src.calibration import metrics, temperature
from src.ingest import fixtures, results
from src.model import dixon_coles
from src.predict import DATA, OUT, adv_side

WC_START = "2026-06-11"


def wc_oos_priors():
    fx = fixtures.load(DATA / "fixtures_wc2026.json")
    jugados = fx[fx["played"] & fx["group"].str.startswith("Group", na=False)]
    hist = results.load_training(DATA / "results.csv")
    pre = hist[hist["date"] < WC_START]
    model = dixon_coles.fit(pre, ref_date=WC_START)
    rows = []
    for r in jugados.itertuples():
        if r.home_team in model.attack and r.away_team in model.attack:
            P = model.score_matrix(r.home_team, r.away_team, adv_side=adv_side(r))
            ph, pdr, pa = model.outcome_probs(P)
            out = 0 if r.home_score > r.away_score else (1 if r.home_score == r.away_score else 2)
            rows.append([ph, pdr, pa, out])
    return np.array(rows)


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    fixtures.download(DATA / "fixtures_wc2026.json")

    bt = pd.read_csv(OUT / "backtest_predicciones.csv")
    bt_probs = bt[["p_home", "p_draw", "p_away"]].to_numpy()
    bt_outs = bt["outcome"].to_numpy()

    wc = wc_oos_priors()
    wc_probs, wc_outs = wc[:, :3], wc[:, 3].astype(int)

    probs = np.vstack([bt_probs, wc_probs])
    outs = np.concatenate([bt_outs, wc_outs])

    cur = json.loads((OUT / "calibration.json").read_text(encoding="utf-8"))
    T_old = cur["temperature"]
    T_new = temperature.fit(probs, outs)

    ll_old = metrics.log_loss(temperature.apply(probs, T_old), outs)
    ll_new = metrics.log_loss(temperature.apply(probs, T_new), outs)
    ll_wc_old = metrics.log_loss(temperature.apply(wc_probs, T_old), wc_outs)
    ll_wc_new = metrics.log_loss(temperature.apply(wc_probs, T_new), wc_outs)

    print(f"Muestra: backtest {len(bt)} + Mundial {len(wc)} = {len(outs)} partidos")
    print(f"  T: {T_old:.3f} -> {T_new:.3f}")
    print(f"  log-loss global:  {ll_old:.4f} -> {ll_new:.4f}")
    print(f"  log-loss Mundial: {ll_wc_old:.4f} -> {ll_wc_new:.4f}")

    cur.update({
        "temperature": T_new,
        "n_matches": int(len(outs)),
        "n_mundial": int(len(wc)),
        "fuente": "backtest + Mundial (out-of-sample)",
    })
    (OUT / "calibration.json").write_text(json.dumps(cur, indent=1), encoding="utf-8")
    print(f"  Escrito outputs/calibration.json (T={T_new:.3f})")


if __name__ == "__main__":
    main()
