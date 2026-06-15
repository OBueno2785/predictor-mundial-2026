"""¿Ganó la selección con mayor power level (ranking FIFA)?

Ranking FIFA del 11 de junio de 2026 (fuentes: ESPN, Wikipedia, Yahoo Sports).
Menor número = mejor. Cruza con los partidos de grupos ya jugados.

Uso:  python -m src.power_level
"""
import sys

from src.ingest import fixtures
from src.predict import DATA

# Ranking FIFA 11-jun-2026 (solo selecciones con partidos jugados hasta ahora)
FIFA = {
    "Brazil": 6, "Morocco": 7, "Netherlands": 8, "Germany": 10, "Mexico": 14,
    "USA": 17, "United States": 17, "Japan": 18, "Switzerland": 19, "Turkey": 22,
    "Ecuador": 23, "South Korea": 25, "Australia": 27, "Canada": 30,
    "Ivory Coast": 33, "Sweden": 38, "Czech Republic": 40, "Paraguay": 41,
    "Scotland": 42, "Tunisia": 45, "Qatar": 56, "South Africa": 60,
    "Bosnia and Herzegovina": 64, "Curaçao": 82, "Haiti": 83,
}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    fixtures.download(DATA / "fixtures_wc2026.json")
    fx = fixtures.load(DATA / "fixtures_wc2026.json")
    jugados = fx[fx["played"] & fx["group"].str.startswith("Group", na=False)
                 ].sort_values("match_number")

    gano = empate = upset = 0
    falta = set()
    print(f"\n{'#':<3} {'Partido (FIFA#)':<46} {'Resultado':<10} {'Favorito FIFA':<14} ¿ganó?")
    print("-" * 92)
    for r in jugados.itertuples():
        rh, ra = FIFA.get(r.home_team), FIFA.get(r.away_team)
        if rh is None:
            falta.add(r.home_team)
        if ra is None:
            falta.add(r.away_team)
        if rh is None or ra is None:
            continue
        hs, as_ = int(r.home_score), int(r.away_score)
        fav = r.home_team if rh < ra else r.away_team
        if hs == as_:
            estado, res = "empate", "—"
            empate += 1
        else:
            ganador = r.home_team if hs > as_ else r.away_team
            if ganador == fav:
                estado, res = "sí", "✓ favorito"
                gano += 1
            else:
                estado, res = "no (UPSET)", "✗ batacazo"
                upset += 1
        partido = f"{r.home_team}({rh}) {hs}-{as_} {r.away_team}({ra})"
        print(f"{r.match_number:<3} {partido:<46} {'':<10} {fav:<14} {estado}  {res}")

    n = gano + empate + upset
    print("-" * 92)
    print(f"\nDe {n} partidos jugados (favorito = menor número FIFA):")
    print(f"  Ganó el favorito FIFA:     {gano:>2}/{n}  ({gano/n:.0%})")
    print(f"  Empate (sin ganador):      {empate:>2}/{n}  ({empate/n:.0%})")
    print(f"  Batacazo (ganó el peor):   {upset:>2}/{n}  ({upset/n:.0%})")
    print(f"\n  Favorito NO perdió (ganó+empató): {gano+empate}/{n} ({(gano+empate)/n:.0%})")
    if falta:
        print(f"\n  (sin rank FIFA cargado: {sorted(falta)})")


if __name__ == "__main__":
    main()
