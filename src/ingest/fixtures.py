"""Fixtures del Mundial 2026 desde fixturedownload.com (sin API key)."""
import json
from pathlib import Path

import pandas as pd
import requests

URL = "https://fixturedownload.com/feed/json/fifa-world-cup-2026"

# fixturedownload usa nombres FIFA; el histórico (martj42) usa nombres comunes
NAME_MAP = {
    "Korea Republic": "South Korea",
    "Czechia": "Czech Republic",
    "USA": "United States",
    "IR Iran": "Iran",
    "Côte d'Ivoire": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",
    "Cabo Verde": "Cape Verde",
    "China PR": "China",
    "Congo DR": "DR Congo",
    "Türkiye": "Turkey",
    "Curacao": "Curaçao",
}

MEXICO_VENUES = ("Mexico City", "Guadalajara", "Monterrey")
CANADA_VENUES = ("Toronto", "Vancouver")


def _venue_country(location: str) -> str:
    if any(v in location for v in MEXICO_VENUES):
        return "Mexico"
    if any(v in location for v in CANADA_VENUES):
        return "Canada"
    return "United States"


def download(dest: Path) -> None:
    r = requests.get(URL, timeout=60)
    r.raise_for_status()
    dest.write_text(json.dumps(r.json(), ensure_ascii=False, indent=1), encoding="utf-8")


def load(path: Path) -> pd.DataFrame:
    raw = json.loads(path.read_text(encoding="utf-8"))
    df = pd.DataFrame(raw)
    df["date_utc"] = pd.to_datetime(df["DateUtc"], utc=True)
    df["home_team"] = df["HomeTeam"].map(lambda t: NAME_MAP.get(t, t))
    df["away_team"] = df["AwayTeam"].map(lambda t: NAME_MAP.get(t, t))
    df["group"] = df["Group"]
    df["venue_country"] = df["Location"].map(_venue_country)
    df["home_score"] = pd.to_numeric(df["HomeTeamScore"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["AwayTeamScore"], errors="coerce")
    df["played"] = df["home_score"].notna() & df["away_score"].notna()
    cols = ["MatchNumber", "RoundNumber", "date_utc", "Location", "venue_country",
            "group", "home_team", "away_team", "home_score", "away_score", "played"]
    return df[cols].rename(columns={"MatchNumber": "match_number",
                                    "RoundNumber": "round", "Location": "location"})


def unmatched_teams(fixtures: pd.DataFrame, known_teams: set) -> list:
    """Valida solo fase de grupos: las eliminatorias traen placeholders (1A, 2B...)."""
    groups = fixtures[fixtures["group"].str.startswith("Group", na=False)]
    teams = set(groups["home_team"]) | set(groups["away_team"])
    return sorted(t for t in teams if t not in known_teams)
