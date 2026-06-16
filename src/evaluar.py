"""Evaluación honesta del sistema sobre los partidos del Mundial ya jugados.

Para cada partido jugado intenta recuperar la predicción SIN fuga de información:
- Si existe un debate congelado (outputs/debates/match_N_*.json): usa su prior
  y su final (ambos limpios, generados antes del partido).
- Si no, reentrena el Dixon-Coles EXCLUYENDO todos los partidos del Mundial
  (ref_date = inicio del torneo) y predice out-of-sample.

Reporta, sobre la muestra disponible: acierto 1X2, RPS, Brier, log-loss, error
de marcador, y compara prior vs final (efecto de los ajustes) vs mercado.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.calibration import metrics, temperature
from src.ingest import fixtures, results
from src.model import dixon_coles
from src.predict import DATA, OUT, adv_side, build_training

WC_START = "2026-06-11"


def outcome(hs, as_) -> int:
    return 0 if hs > as_ else (1 if hs == as_ else 2)


def cargar_debates() -> dict:
    d = {}
    deb = OUT / "debates"
    if deb.exists():
        for f in sorted(deb.glob("match_*.json")):
            try:
                d[int(f.stem.split("_")[1])] = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                continue
    return d


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    fixtures.download(DATA / "fixtures_wc2026.json")
    fx = fixtures.load(DATA / "fixtures_wc2026.json")
    jugados = fx[fx["played"] & fx["group"].str.startswith("Group", na=False)
                 ].sort_values("match_number")
    if jugados.empty:
        print("Aún no hay partidos de grupos jugados.")
        return

    # Modelo out-of-sample: SOLO datos previos al torneo (sin fuga)
    hist_full = results.load_training(DATA / "results.csv")
    pre = hist_full[hist_full["date"] < WC_START]
    model_oos = dixon_coles.fit(pre, ref_date=WC_START)
    T = json.loads((OUT / "calibration.json").read_text())["temperature"]
    debates = cargar_debates()

    filas = []
    for r in jugados.itertuples():
        real = outcome(r.home_score, r.away_score)
        side = adv_side(r)
        fuente = "out-of-sample"
        prior = final = market = None

        if r.match_number in debates and debates[r.match_number].get("prior"):
            d = debates[r.match_number]
            p = d["prior"]
            prior = np.array([p["p_home"], p["p_draw"], p["p_away"]])
            if d.get("final") and not str(
                    d.get("veredicto", {}).get("resumen", "")).startswith("Degradado"):
                f = d["final"]
                final = np.array([f["p_home"], f["p_draw"], f["p_away"]])
            o = d.get("odds")
            if o:
                market = np.array([o["p_home"], o["p_draw"], o["p_away"]])
            fuente = "debate congelado"
        else:
            if r.home_team in model_oos.attack and r.away_team in model_oos.attack:
                P = model_oos.score_matrix(r.home_team, r.away_team, adv_side=side)
                prior = np.array(model_oos.outcome_probs(P))
                prior = temperature.apply(prior, T)
            else:
                fuente = "sin datos pre-torneo"

        filas.append({
            "match": r.match_number,
            "partido": f"{r.home_team} {int(r.home_score)}-{int(r.away_score)} {r.away_team}",
            "real": real, "prior": prior, "final": final, "market": market,
            "fuente": fuente,
        })

    print(f"\n{'=' * 70}\nEVALUACIÓN — {len(filas)} partidos de grupos jugados\n{'=' * 70}\n")
    print(f"{'#':<3} {'Partido':<34} {'pred 1X2':<10} {'acierto':<8} fuente")
    etiquetas = ["1", "X", "2"]
    validos = [f for f in filas if f["prior"] is not None]
    for f in filas:
        if f["prior"] is None:
            print(f"{f['match']:<3} {f['partido']:<34} {'—':<10} {'—':<8} {f['fuente']}")
            continue
        usar = f["final"] if f["final"] is not None else f["prior"]
        pred = int(usar.argmax())
        ok = "sí" if pred == f["real"] else "no"
        probs = "/".join(f"{x:.0%}" for x in usar)
        print(f"{f['match']:<3} {f['partido']:<34} {probs:<10} "
              f"{etiquetas[pred]}->{etiquetas[f['real']]} {ok:<4} {f['fuente']}")

    if not validos:
        return

    def metricas(key):
        sub = [f for f in validos if f[key] is not None]
        if not sub:
            return None
        probs = np.array([f[key] for f in sub])
        outs = np.array([f["real"] for f in sub])
        return {
            "n": len(sub),
            "acc": metrics.accuracy_1x2(probs, outs),
            "rps": metrics.rps(probs, outs),
            "brier": metrics.brier(probs, outs),
            "logloss": metrics.log_loss(probs, outs),
        }

    base = np.array([0.40, 0.27, 0.33])  # frecuencias base 1X2 internacional
    outs_all = np.array([f["real"] for f in validos])
    clim = np.tile(base, (len(validos), 1))

    print(f"\n{'-' * 70}\nMÉTRICAS (recordar: muestra MUY pequeña, alta varianza)\n{'-' * 70}")
    mp = metricas("prior")
    print(f"\nPRIOR del modelo (n={mp['n']}):")
    print(f"  Acierto 1X2: {mp['acc']:.0%}   RPS: {mp['rps']:.4f}   "
          f"Brier: {mp['brier']:.3f}   LogLoss: {mp['logloss']:.3f}")
    print(f"  RPS baseline climatológico: {metrics.rps(clim, outs_all):.4f} "
          f"(el modelo {'gana' if mp['rps'] < metrics.rps(clim, outs_all) else 'pierde'})")

    mf = metricas("final")
    if mf:
        sub = [f for f in validos if f["final"] is not None]
        pr = np.array([f["prior"] for f in sub])
        fi = np.array([f["final"] for f in sub])
        ou = np.array([f["real"] for f in sub])
        print(f"\nEFECTO DE LOS AJUSTES (n={mf['n']} con debate):")
        print(f"  {'':16} {'prior':>8} {'final':>8}")
        print(f"  {'RPS':16} {metrics.rps(pr, ou):>8.4f} {metrics.rps(fi, ou):>8.4f}")
        print(f"  {'Brier':16} {metrics.brier(pr, ou):>8.3f} {metrics.brier(fi, ou):>8.3f}")
        print(f"  {'LogLoss':16} {metrics.log_loss(pr, ou):>8.3f} {metrics.log_loss(fi, ou):>8.3f}")
    else:
        print("\nEFECTO DE LOS AJUSTES: ningún partido jugado tenía debate congelado válido.")

    # ¿Le ganamos al mercado? Subconjunto con cuotas guardadas
    con_mkt = [f for f in validos if f.get("market") is not None and f["final"] is not None]
    if con_mkt:
        ou = np.array([f["real"] for f in con_mkt])
        pr = np.array([f["prior"] for f in con_mkt])
        fi = np.array([f["final"] for f in con_mkt])
        mk = np.array([f["market"] for f in con_mkt])
        print(f"\nVS MERCADO (n={len(con_mkt)} con cuotas) — RPS (menor = mejor):")
        print(f"  prior modelo: {metrics.rps(pr, ou):.4f}")
        print(f"  mercado:      {metrics.rps(mk, ou):.4f}")
        print(f"  FINAL mezcla: {metrics.rps(fi, ou):.4f}")


if __name__ == "__main__":
    main()
