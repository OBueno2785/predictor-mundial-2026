"""Scheduler de autoajuste (Fase 3).

- Job diario 06:00 (America/Lima): reingesta + reentrenamiento +
  regeneración de PREDICCIONES.md + copia histórica + reprogramación
  de los jobs T-60.
- Job T-60min por partido: pipeline completo con debate (cuotas frescas,
  alineaciones y noticias de último minuto). El JSON timestampeado en
  outputs/debates/ es la predicción congelada.

Uso:  py -m src.scheduler [--dry-run]

El proceso debe quedar abierto (consola dedicada, o Task Scheduler de
Windows con disparador "al iniciar sesión"). Si la máquina estuvo apagada,
al arrancar se reingesta todo y se reprograman los partidos pendientes.
"""
import logging
import shutil
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
HORA_DIARIA = 6

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


def job_diario(sched: BlockingScheduler) -> None:
    log.info("=== Job diario: reingesta + reentrenamiento ===")
    if _run("src.predict"):
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        hist = OUT / "history"
        hist.mkdir(exist_ok=True)
        shutil.copy2(OUT / "predicciones_grupos.csv",
                     hist / f"predicciones_{ts}.csv")
    programar_t60(sched)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    dry = "--dry-run" in sys.argv

    executors = {"default": ThreadPoolExecutor(2)}
    sched = BlockingScheduler(timezone=TZ, executors=executors,
                              job_defaults={"coalesce": True})
    sched.add_job(job_diario, "cron", hour=HORA_DIARIA, minute=0,
                  args=[sched], id="diario")

    if dry:
        programar_t60(sched)
        print("\n--- DRY RUN: jobs programados ---")
        for j in sorted(sched.get_jobs(), key=lambda j: str(j.trigger)):
            print(f"  {j.id:<12} -> {j.trigger}")
        return

    job_diario(sched)  # arranque: datos frescos + programación inicial
    log.info("Scheduler activo (diario %02d:00 %s + T-60 por partido). Ctrl+C para salir.",
             HORA_DIARIA, TZ)
    sched.start()


if __name__ == "__main__":
    main()
