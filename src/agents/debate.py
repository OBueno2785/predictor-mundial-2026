"""Orquestación del debate multiagente sobre el prior del Dixon-Coles.

Flujo: ronda 1 (agentes en paralelo) → ronda 2 (réplicas) → Juez (salida
estructurada con deltas acotados en log-xG) → recomputo de la matriz de
marcadores con los ritmos ajustados.
"""
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import anthropic
import numpy as np
from dotenv import load_dotenv

from src.agents import prompts
from src.agents.schemas import DELTA_MAX, VeredictoJuez

load_dotenv()

MODEL = "claude-opus-4-8"
MAX_TOKENS_AGENT = 1500
MAX_TOKENS_JUEZ = 1500
WEB_SEARCH = [{"type": "web_search_20260209", "name": "web_search", "max_uses": 4}]


@dataclass
class ResultadoDebate:
    veredicto: VeredictoJuez
    delta_home: float
    delta_away: float
    rondas: dict = field(default_factory=dict)
    usage: dict = field(default_factory=dict)


def _texto(response) -> str:
    return "\n".join(b.text for b in response.content if b.type == "text")


def _acumular_usage(usage_total: dict, response) -> None:
    u = response.usage
    usage_total["input"] = usage_total.get("input", 0) + u.input_tokens
    usage_total["output"] = usage_total.get("output", 0) + u.output_tokens
    usage_total["cache_read"] = usage_total.get("cache_read", 0) + (u.cache_read_input_tokens or 0)


def render_contexto(ctx: dict) -> str:
    """Contexto compartido del partido (bloque cacheado del prefijo)."""
    lines = [
        f"# Partido: {ctx['home']} vs {ctx['away']}",
        f"{ctx['group']} · {ctx['fecha']} · {ctx['estadio']} ({ctx['sede_pais']})",
        f"Ventaja de localía aplicada a: {ctx['ventaja'] or 'nadie (cancha neutral)'}",
        "",
        "## Prior del modelo Dixon-Coles",
        f"- xG esperado: {ctx['home']} {ctx['xg_home']:.2f} — {ctx['xg_away']:.2f} {ctx['away']}",
        f"- Probabilidades: local {ctx['p_home']:.1%} / empate {ctx['p_draw']:.1%} / visita {ctx['p_away']:.1%}",
        f"- Marcadores más probables: {ctx['top_scores']}",
        "",
        f"## Forma reciente {ctx['home']} (últimos 5)",
        ctx["form_home"],
        "",
        f"## Forma reciente {ctx['away']} (últimos 5)",
        ctx["form_away"],
    ]
    if ctx.get("odds"):
        o = ctx["odds"]
        lines += ["", "## Mercado (probabilidades implícitas sin margen, "
                  f"mediana de {o['n_bookies']} casas)",
                  f"- local {o['p_home']:.1%} / empate {o['p_draw']:.1%} / visita {o['p_away']:.1%}"]
    else:
        lines += ["", "## Mercado", "Sin datos de cuotas en esta ejecución."]
    return "\n".join(lines)


def _llamada_agente(client, system: str, contexto: str, tarea: str,
                    historial: list | None = None, tools: list | None = None):
    messages = list(historial or [])
    if not messages:
        messages.append({"role": "user", "content": [
            {"type": "text", "text": contexto, "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": tarea},
        ]})
    else:
        messages.append({"role": "user", "content": tarea})
    kwargs = {}
    if tools:
        kwargs["tools"] = tools
    return client.messages.create(
        model=MODEL, max_tokens=MAX_TOKENS_AGENT,
        thinking={"type": "adaptive"}, system=system,
        messages=messages, **kwargs)


def run_debate(ctx: dict, rounds: int = 2, client=None) -> ResultadoDebate:
    client = client or anthropic.Anthropic()
    contexto = render_contexto(ctx)
    usage_total = {}

    roles = [
        ("Estadístico", prompts.ESTADISTICO, None),
        ("Plantel/Noticias", prompts.PLANTEL, WEB_SEARCH),
        ("Sentimiento", prompts.SENTIMIENTO, WEB_SEARCH),
    ]
    if ctx.get("odds"):
        roles.insert(1, ("Mercado", prompts.MERCADO, None))

    tarea_r1 = "Emite tu análisis inicial del partido según tu rol."
    rondas = {}

    def _ronda1(rol):
        nombre, system, tools = rol
        r = _llamada_agente(client, system, contexto, tarea_r1, tools=tools)
        return nombre, r

    with ThreadPoolExecutor(max_workers=4) as ex:
        r1 = dict(ex.map(_ronda1, roles))
    for nombre, resp in r1.items():
        _acumular_usage(usage_total, resp)
    rondas["ronda1"] = {n: _texto(r) for n, r in r1.items()}

    posiciones_finales = rondas["ronda1"]
    if rounds >= 2:
        def _ronda2(rol):
            nombre, system, tools = rol
            otras = "\n\n".join(f"### {n}\n{t}" for n, t in rondas["ronda1"].items()
                                if n != nombre)
            historial = [
                {"role": "user", "content": [
                    {"type": "text", "text": contexto, "cache_control": {"type": "ephemeral"}},
                    {"type": "text", "text": tarea_r1},
                ]},
                {"role": "assistant", "content": rondas["ronda1"][nombre]},
            ]
            r = _llamada_agente(client, system, contexto,
                                prompts.REPLICA.format(posiciones=otras),
                                historial=historial, tools=tools)
            return nombre, r

        with ThreadPoolExecutor(max_workers=4) as ex:
            r2 = dict(ex.map(_ronda2, roles))
        for nombre, resp in r2.items():
            _acumular_usage(usage_total, resp)
        rondas["ronda2"] = {n: _texto(r) for n, r in r2.items()}
        posiciones_finales = rondas["ronda2"]

    panel = "\n\n".join(f"### {n}\n{t}" for n, t in posiciones_finales.items())
    juicio = client.messages.parse(
        model=MODEL, max_tokens=MAX_TOKENS_JUEZ,
        thinking={"type": "adaptive"}, system=prompts.JUEZ,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": contexto, "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": f"## Posiciones finales del panel\n\n{panel}\n\n"
                                     "Emite tu veredicto."},
        ]}],
        output_format=VeredictoJuez)
    _acumular_usage(usage_total, juicio)
    veredicto = juicio.parsed_output

    dh = float(np.clip(veredicto.delta_log_xg_home, -DELTA_MAX, DELTA_MAX))
    da = float(np.clip(veredicto.delta_log_xg_away, -DELTA_MAX, DELTA_MAX))
    return ResultadoDebate(veredicto=veredicto, delta_home=dh, delta_away=da,
                           rondas=rondas, usage=usage_total)


def guardar(resultado: ResultadoDebate, ctx: dict, dest) -> None:
    dest.write_text(json.dumps({
        "partido": f"{ctx['home']} vs {ctx['away']}",
        "fecha": str(ctx["fecha"]),
        "prior": {k: ctx[k] for k in ["xg_home", "xg_away", "p_home", "p_draw", "p_away"]},
        "veredicto": resultado.veredicto.model_dump(),
        "delta_aplicado": {"home": resultado.delta_home, "away": resultado.delta_away},
        "rondas": resultado.rondas,
        "usage": resultado.usage,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
