"""Orquestación del debate multiagente sobre el prior del Dixon-Coles utilizando Gemini.

Transporte: API de Gemini (a través del SDK google-genai). Los agentes de noticias y
sentimiento utilizan Google Search Grounding para realizar búsquedas web en tiempo real.
El Juez utiliza la API de salidas estructuradas de Gemini con el esquema Pydantic VeredictoJuez.
"""
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types

from src.agents import prompts
from src.agents.schemas import DELTA_MAX, VeredictoJuez

load_dotenv()

# Modelos por defecto: gemini-3.5-flash es rápido, económico y soporta Google Search y JSON estructurado.
MODEL = os.environ.get("DEBATE_MODEL", "gemini-3.5-flash")
MODEL_JUEZ = os.environ.get("DEBATE_MODEL_JUEZ", "gemini-3.5-flash")
TIMEOUT_S = 600
MAX_WORKERS = 4

# Cliente de Gemini (inicialización perezosa)
_client_memo = None


class CuotaAgotada(RuntimeError):
    """Límite de cuota o rate limit alcanzado en la API de Gemini (HTTP 429)."""


def get_gemini_client():
    global _client_memo
    if _client_memo is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY no encontrada en las variables de entorno. "
                "Por favor configúrala en tu archivo .env."
            )
        _client_memo = genai.Client(api_key=api_key)
    return _client_memo


@dataclass
class ResultadoDebate:
    veredicto: VeredictoJuez
    delta_home: float
    delta_away: float
    rondas: dict = field(default_factory=dict)
    usage: dict = field(default_factory=dict)


def _call_gemini(prompt: str, system_instruction: str = None, allow_web: bool = False,
                 model: str = MODEL) -> tuple[str, dict]:
    client = get_gemini_client()
    tools = [{"google_search": {}}] if allow_web else None
    
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        tools=tools,
        temperature=0.2,
    )
    
    start_time = time.time()
    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=config,
        )
    except Exception as e:
        err_msg = str(e)
        if "429" in err_msg or "Quota exceeded" in err_msg or "quota" in err_msg.lower():
            raise CuotaAgotada(f"Límite de cuota alcanzado en la API de Gemini: {e}")
        raise RuntimeError(f"Llamada a Gemini falló: {e}")
        
    duration_ms = int((time.time() - start_time) * 1000)
    
    usage = {
        "duration_ms": duration_ms,
        "num_turns": 1,
        "prompt_tokens": response.usage_metadata.prompt_token_count if response.usage_metadata else 0,
        "candidates_tokens": response.usage_metadata.candidates_token_count if response.usage_metadata else 0,
        "total_tokens": response.usage_metadata.total_token_count if response.usage_metadata else 0,
    }
    
    text = response.text or ""
    return text, usage


def _call_gemini_juez(prompt: str, system_instruction: str, response_schema,
                      model: str = MODEL_JUEZ) -> tuple[VeredictoJuez, dict]:
    client = get_gemini_client()
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        response_mime_type="application/json",
        response_schema=response_schema,
        temperature=0.1,
    )
    
    start_time = time.time()
    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=config,
        )
    except Exception as e:
        err_msg = str(e)
        if "429" in err_msg or "Quota exceeded" in err_msg or "quota" in err_msg.lower():
            raise CuotaAgotada(f"Límite de cuota alcanzado en la API de Gemini para el Juez: {e}")
        raise RuntimeError(f"Llamada al Juez de Gemini falló: {e}")
        
    duration_ms = int((time.time() - start_time) * 1000)
    
    usage = {
        "duration_ms": duration_ms,
        "num_turns": 1,
        "prompt_tokens": response.usage_metadata.prompt_token_count if response.usage_metadata else 0,
        "candidates_tokens": response.usage_metadata.candidates_token_count if response.usage_metadata else 0,
        "total_tokens": response.usage_metadata.total_token_count if response.usage_metadata else 0,
    }
    
    try:
        obj = response_schema.model_validate_json(response.text)
    except Exception as e:
        raise ValueError(
            f"El Juez de Gemini no devolvió un JSON compatible con el esquema: {response.text}. Error: {e}"
        )
        
    return obj, usage


def _acumular_usage(total: dict, meta: dict) -> None:
    total["llamadas"] = total.get("llamadas", 0) + 1
    total["duracion_ms"] = total.get("duracion_ms", 0) + meta.get("duration_ms", 0)
    total["turnos"] = total.get("turnos", 0) + meta.get("num_turns", 0)
    total["prompt_tokens"] = total.get("prompt_tokens", 0) + meta.get("prompt_tokens", 0)
    total["candidates_tokens"] = total.get("candidates_tokens", 0) + meta.get("candidates_tokens", 0)
    total["total_tokens"] = total.get("total_tokens", 0) + meta.get("total_tokens", 0)


def render_contexto(ctx: dict) -> str:
    """Contexto compartido del partido para todos los agentes."""
    lines = [
        f"# Partido: {ctx['home']} vs {ctx['away']}",
        f"{ctx['group']} (Ronda {ctx.get('round', '—')}) · {ctx['fecha']} · {ctx['estadio']} ({ctx['sede_pais']})",
        f"Ventaja de localía aplicada a: {ctx['ventaja'] or 'nadie (cancha neutral)'}",
        "",
        "## Desempeño en el Mundial 2026 (Torneo Actual)",
        f"- {ctx['home']}: {ctx.get('wc_stats_home', '—')}",
        f"- {ctx['away']}: {ctx.get('wc_stats_away', '—')}",
        "",
        "## Capacidad Goleadora de las Delanteras en el Torneo",
        f"- {ctx['home']}: {ctx.get('capacidad_goleadora_delanteras_home', '—')}",
        f"- {ctx['away']}: {ctx.get('capacidad_goleadora_delanteras_away', '—')}",
        "",
        "## Prior del modelo Dixon-Coles",
        f"- xG esperado: {ctx['home']} {ctx['xg_home']:.2f} — {ctx['xg_away']:.2f} {ctx['away']}",
        f"- Probabilidades: local {ctx['p_home']:.1%} / empate {ctx['p_draw']:.1%} / visita {ctx['p_away']:.1%}",
        f"- Marcadores más probables: {ctx['top_scores']}",
        "",
        f"## Forma reciente {ctx['home']} (últimos 5)",
        ctx["form_home"],
        f"  promedio: {ctx.get('form5_home', '—')}",
        "",
        f"## Forma reciente {ctx['away']} (últimos 5)",
        ctx["form_away"],
        f"  promedio: {ctx.get('form5_away', '—')}",
        "",
        "## Situación de grupo (incentivos)",
        f"- {ctx['home']}: posición {ctx.get('pos_home', '—')} · {ctx.get('outlook_home', '—')}",
        f"  - Probabilidad de clasificar: {ctx.get('prob_clasificar_home', 0.0):.1%}",
        f"  - Necesidad de goles: {ctx.get('necesidad_goles_home', '—')}",
        f"- {ctx['away']}: posición {ctx.get('pos_away', '—')} · {ctx.get('outlook_away', '—')}",
        f"  - Probabilidad de clasificar: {ctx.get('prob_clasificar_away', 0.0):.1%}",
        f"  - Necesidad de goles: {ctx.get('necesidad_goles_away', '—')}",
    ]
    if ctx.get("odds"):
        o = ctx["odds"]
        lines += ["", "## Mercado (probabilidades implícitas sin margen, "
                  f"mediana de {o['n_bookies']} casas)",
                  f"- local {o['p_home']:.1%} / empate {o['p_draw']:.1%} / visita {o['p_away']:.1%}"]
    else:
        lines += ["", "## Mercado", "Sin datos de cuotas en esta ejecución."]
    return "\n".join(lines)


def run_debate(ctx: dict, rounds: int = 2) -> ResultadoDebate:
    contexto = render_contexto(ctx)
    usage_total = {}
    tarea_r1 = f"Emite tu análisis inicial del partido según tu rol y basándote en este contexto:\n\n{contexto}"

    # El mercado se incorpora mecánicamente después del debate (src.blend),
    # por eso NO hay agente de Mercado aquí: sería doble conteo. El panel se
    # enfoca en información que el mercado aún no refleja.
    roles = [
        ("Estadístico", prompts.ESTADISTICO, False),
        ("Plantel/Noticias", prompts.PLANTEL, True),
        ("Sentimiento", prompts.SENTIMIENTO, True),
    ]

    rondas = {}

    def _ronda1(rol):
        nombre, system, web = rol
        texto, meta = _call_gemini(
            prompt=tarea_r1,
            system_instruction=system,
            allow_web=web,
            model=MODEL
        )
        return nombre, texto, meta

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        r1 = list(ex.map(_ronda1, roles))
    for _, _, meta in r1:
        _acumular_usage(usage_total, meta)
    rondas["ronda1"] = {n: t for n, t, _ in r1}

    posiciones_finales = rondas["ronda1"]
    if rounds >= 2:
        def _ronda2(rol):
            nombre, system, web = rol
            otras = "\n\n".join(f"### {n}\n{t}" for n, t in rondas["ronda1"].items()
                                if n != nombre)
            prompt = (f"## Contexto del Partido\n{contexto}\n\n"
                      f"## Tu análisis inicial\n{rondas['ronda1'][nombre]}\n\n"
                      + prompts.REPLICA.format(posiciones=otras))
            texto, meta = _call_gemini(
                prompt=prompt,
                system_instruction=system,
                allow_web=web,
                model=MODEL
            )
            return nombre, texto, meta

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            r2 = list(ex.map(_ronda2, roles))
        for _, _, meta in r2:
            _acumular_usage(usage_total, meta)
        rondas["ronda2"] = {n: t for n, t, _ in r2}
        posiciones_finales = rondas["ronda2"]

    panel = "\n\n".join(f"### {n}\n{t}" for n, t in posiciones_finales.items())
    
    veredicto, meta = _call_gemini_juez(
        prompt=f"## Contexto del Partido\n{contexto}\n\n## Posiciones finales del panel\n\n{panel}\n\nEmite tu veredicto.",
        system_instruction=prompts.JUEZ,
        response_schema=VeredictoJuez,
        model=MODEL_JUEZ
    )
    _acumular_usage(usage_total, meta)

    dh = float(np.clip(veredicto.delta_log_xg_home, -DELTA_MAX, DELTA_MAX))
    da = float(np.clip(veredicto.delta_log_xg_away, -DELTA_MAX, DELTA_MAX))
    return ResultadoDebate(veredicto=veredicto, delta_home=dh, delta_away=da,
                           rondas=rondas, usage=usage_total)


def guardar(resultado: ResultadoDebate, ctx: dict, dest) -> None:
    dest.write_text(json.dumps({
        "partido": f"{ctx['home']} vs {ctx['away']}",
        "fecha": str(ctx["fecha"]),
        "prior": {k: ctx[k] for k in ["xg_home", "xg_away", "p_home", "p_draw", "p_away"]},
        "odds": ctx.get("odds"),
        "model_final": ctx.get("model_final"),
        "blend_weight_market": ctx.get("blend_weight_market"),
        "final": ctx.get("final"),
        "veredicto": resultado.veredicto.model_dump(),
        "delta_aplicado": {"home": resultado.delta_home, "away": resultado.delta_away},
        "rondas": resultado.rondas,
        "usage": resultado.usage,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
