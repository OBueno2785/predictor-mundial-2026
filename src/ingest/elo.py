"""Ratings Elo de selecciones (eloratings.net). Se almacena crudo para la capa de agentes (Fase 2)."""
from pathlib import Path

import pandas as pd
import requests

URL = "https://www.eloratings.net/World.tsv"


def download(dest: Path) -> None:
    r = requests.get(URL, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    dest.write_bytes(r.content)


def load(path: Path) -> pd.DataFrame:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) >= 4 and parts[0].isdigit():
            rows.append({"rank": int(parts[0]), "code": parts[2], "elo": int(parts[3])})
    return pd.DataFrame(rows)
