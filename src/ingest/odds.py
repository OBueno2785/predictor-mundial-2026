"""Cuotas 1X2 de The Odds API con remoción de margen (vig).

Requiere ODDS_API_KEY en .env. Si no hay key devuelve None y el pipeline
sigue sin la señal de mercado (el Agente Mercado se omite del debate).
"""
import os

import numpy as np
import requests

SPORT_KEY = "soccer_fifa_world_cup"
URL = f"https://api.the-odds-api.com/v4/sports/{SPORT_KEY}/odds"


def fetch_all() -> list | None:
    key = os.environ.get("ODDS_API_KEY")
    if not key:
        return None
    r = requests.get(URL, params={
        "apiKey": key, "regions": "eu", "markets": "h2h", "oddsFormat": "decimal",
    }, timeout=60)
    r.raise_for_status()
    return r.json()


def _normalize(name: str) -> str:
    return name.lower().replace("south korea", "korea").strip()


def implied_probs(events: list, home: str, away: str) -> dict | None:
    """Mediana entre casas de las probabilidades sin margen (normalización
    proporcional). Devuelve {p_home, p_draw, p_away, n_bookies} o None."""
    h, a = _normalize(home), _normalize(away)
    for ev in events or []:
        eh, ea = _normalize(ev["home_team"]), _normalize(ev["away_team"])
        if {h, a} != {eh, ea}:
            continue
        flipped = (eh == a)
        per_book = []
        for bk in ev.get("bookmakers", []):
            for mk in bk.get("markets", []):
                if mk["key"] != "h2h":
                    continue
                prices = {o["name"]: o["price"] for o in mk["outcomes"]}
                if len(prices) != 3:
                    continue
                draw = prices.get("Draw")
                ph = prices.get(ev["home_team"])
                pa = prices.get(ev["away_team"])
                if not (draw and ph and pa):
                    continue
                inv = np.array([1 / ph, 1 / draw, 1 / pa])
                per_book.append(inv / inv.sum())
        if not per_book:
            return None
        med = np.median(np.array(per_book), axis=0)
        med = med / med.sum()
        if flipped:
            med = med[::-1]
        return {"p_home": float(med[0]), "p_draw": float(med[1]),
                "p_away": float(med[2]), "n_bookies": len(per_book)}
    return None
