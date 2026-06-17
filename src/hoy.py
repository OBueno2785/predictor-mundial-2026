"""Predicciones de los partidos de hoy (zona horaria America/Lima).

Toma la predicción vigente de outputs/predicciones_grupos.csv (prior calibrado,
o final ajustada por panel+mercado si el partido ya pasó por su debate T-60).

Uso:  python -m src.hoy [YYYY-MM-DD]
"""
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from src.ingest import fixtures
from src.predict import DATA, OUT

LIMA = ZoneInfo("America/Lima")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    fixtures.download(DATA / "fixtures_wc2026.json")
    fx = fixtures.load(DATA / "fixtures_wc2026.json")

    dia = sys.argv[1] if len(sys.argv) > 1 else datetime.now(LIMA).strftime("%Y-%m-%d")
    fx["fecha_lima"] = fx["date_utc"].dt.tz_convert(LIMA)
    hoy = fx[fx["fecha_lima"].dt.strftime("%Y-%m-%d") == dia].sort_values("date_utc")

    preds = pd.read_csv(OUT / "predicciones_grupos.csv")
    pmap = {int(r.match): r for r in preds.itertuples()}

    if hoy.empty:
        print(f"No hay partidos el {dia} (hora Lima).")
        return

    print(f"\nPartidos del {dia} (hora Lima) — {len(hoy)} partido(s)\n")
    for r in hoy.itertuples():
        hora = r.fecha_lima.strftime("%H:%M")
        p = pmap.get(int(r.match_number))
        cab = f"  {hora}  {r.home_team} vs {r.away_team}  [{r.group}]"
        if r.played:
            cab += f"  — YA JUGADO {int(r.home_score)}-{int(r.away_score)}"
        print(cab)
        if p is None:
            print("      (sin predicción en CSV)\n")
            continue
        panel = "  ✓ ajustado por panel+mercado" if getattr(p, "ajustada", False) else \
            "  (prior del modelo)"
        print(f"      1 {p.p_home:.0%}  ·  X {p.p_draw:.0%}  ·  2 {p.p_away:.0%}   "
              f"→ marcador probable {p.score_pred}{panel}")
        print(f"      xG {p.xg_home:.1f}-{p.xg_away:.1f}  ·  top: {p.top_scores}\n")


if __name__ == "__main__":
    main()
