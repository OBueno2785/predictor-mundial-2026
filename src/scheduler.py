"""Scheduler de autoajuste: un job T-60min por partido, nada más.

60 minutos antes de cada kickoff se ejecuta el pipeline completo con la
última información disponible: reingesta (resultados recientes, cuotas
frescas), reentrenamiento del Dixon-Coles y debate multiagente. El JSON
timestampeado en outputs/debates/ es la predicción congelada.

Uso:  py -m src.scheduler [--dry-run]

El proceso debe quedar abierto (consola dedicada, o Task Scheduler de
Windows con disparador "al iniciar sesión"). Al arrancar refresca el
fixture y programa los partidos pendientes.
"""
import logging
import subprocess
import sys
from datetime import datetime, timedelta, timezone

from apscheduler.executors.pool import ThreadPoolExecutor
from apscheduler.schedulers.blocking import BlockingScheduler

from src.ingest import fixtures
from src.predict import DATA, OUT, ROOT

PY = ROOT / ".venv" / "Scripts" / "python.exe"
T_MINUS = timedelta(minutes=60)
TZ = "America/Lima"

OUT.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(OUT / "scheduler.log", encoding="utf-8"),
              logging.StreamHandler()])
log = logging.getLogger("scheduler")
logging.getLogger("apscheduler").setLevel(logging.WARNING)


def _run(modulo: str, *args) -> bool:
    cmd = [str(PY), "-m", modulo, *args]
    log.info("ejecutando: %s", " ".join(cmd[2:]))
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=3600)
    for line in (r.stdout or "").splitlines():
        log.info("  | %s", line)
    if r.returncode != 0:
        log.error("FALLO %s (%d): %s", modulo, r.returncode, (r.stderr or "")[-800:])
        return False
    return True


def job_t60(match_number: int, home: str, away: str) -> None:
    log.info("=== T-60: partido %d (%s vs %s) ===", match_number, home, away)
    _run("src.debate_match", str(match_number))


def programar_t60(sched: BlockingScheduler) -> int:
    fx = fixtures.load(DATA / "fixtures_wc2026.json")
    now = datetime.now(timezone.utc)
    n = 0
    for r in fx.itertuples():
        if r.played or "Group" not in str(r.group):
            continue
        run_at = r.date_utc - T_MINUS
        if run_at <= now:
            continue
        sched.add_job(job_t60, "date", run_date=run_at,
                      args=[int(r.match_number), r.home_team, r.away_team],
                      id=f"t60_{r.match_number}", replace_existing=True,
                      misfire_grace_time=900)
        n += 1
    log.info("%d jobs T-60 programados", n)
    return n


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("El scheduler T-60 ha sido desactivado/anulado.")
    print("Los debates ahora se ejecutan dinámicamente de forma automática al correr el modelo (predict.py).")
    sys.exit(0)
    sched.start()


if __name__ == "__main__":
    main()
