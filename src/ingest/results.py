"""Histórico de resultados internacionales (dataset martj42, actualizado a diario)."""
from pathlib import Path

import pandas as pd
import requests

URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"

FRIENDLY_WEIGHT = 0.5


def download(dest: Path) -> None:
    r = requests.get(URL, timeout=120)
    r.raise_for_status()
    dest.write_bytes(r.content)


def load_training(path: Path, since: str = "2018-01-01") -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df = df[df["date"] >= since].copy()
    # El Mundial 2026 se incorpora desde el feed de fixtures (fuente más fresca)
    df = df[~((df["tournament"] == "FIFA World Cup") & (df["date"] >= "2026-06-01"))]
    df = df.dropna(subset=["home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df["neutral"] = df["neutral"].astype(bool)
    df["weight_importance"] = (df["tournament"] == "Friendly").map(
        {True: FRIENDLY_WEIGHT, False: 1.0})
    return df[["date", "home_team", "away_team", "home_score", "away_score",
               "neutral", "weight_importance"]].reset_index(drop=True)
