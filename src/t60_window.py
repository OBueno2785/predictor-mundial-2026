"""Partidos con kickoff dentro de la ventana T-60 (45 a 105 min desde ahora).

Pensado para el agente cloud programado cada hora: con esa ventana de 60 min
cada partido cae exactamente en una ejecución. Imprime el contexto completo
(prior, cuotas, forma) listo para el debate.

Uso:  python -m src.t60_window [--next]   (--next: el próximo partido pendiente,
                                            ignorando la ventana — para pruebas)
"""
import sys
from datetime import datetime, timedelta, timezone

from src.agents import debate
from src.debate_match import build_context
from src.ingest import fixtures
from src.predict import DATA

VENTANA_MIN = timedelta(minutes=45)
VENTANA_MAX = timedelta(minutes=105)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    proximo = "--next" in sys.argv
    fixtures.download(DATA / "fixtures_wc2026.json")
    fx = fixtures.load(DATA / "fixtures_wc2026.json")
    now = datetime.now(timezone.utc)

    grupos = fx[~fx["played"] & fx["group"].str.startswith("Group", na=False)]
    if proximo:
        grupos = grupos.sort_values("date_utc").head(1)
    else:
        delta = grupos["date_utc"] - now
        grupos = grupos[(delta > VENTANA_MIN) & (delta <= VENTANA_MAX)]

    if grupos.empty:
        print("SIN_PARTIDOS_EN_VENTANA")
        return

    print(f"PARTIDOS_EN_VENTANA: {len(grupos)}")
    for r in grupos.itertuples():
        ctx, _, _ = build_context(int(r.match_number), offline=True)
        print(f"\n{'=' * 60}\nMATCH_NUMBER: {r.match_number}\n{'=' * 60}")
        print(debate.render_contexto(ctx))


if __name__ == "__main__":
    main()
