"""Combinación de pronósticos 1X2 en la probabilidad final.

Pool LINEAL (promedio ponderado): combinar dos pronósticos calibrados por
promedio es notoriamente robusto y difícil de batir. El mercado entra UNA sola
vez, aquí, de forma mecánica — por eso el debate ya NO lleva un agente de
mercado (sería doble conteo); el panel se concentra en noticias que el mercado
aún no refleja.

Por qué no se combina con el ranking de poder (FIFA/Elo): el Dixon-Coles ya se
entrena con resultados, que es la misma información de la que sale el ranking
(correlación ~0.9). Promediarlos reduce varianza marginal pero no agrega señal;
queda como mejora futura vía prior de contracción para selecciones con pocos
datos (hoy se filtran con MIN_MATCHES).

PESO DEL MERCADO (W_MARKET): provisional. El mercado es un benchmark fuerte,
pero el peso óptimo solo se fija con datos. Cada predicción registra las tres
señales (modelo / mercado / mezcla) para ajustarlo tras ~30-40 partidos.
Ajustado de 0.5 a 0.65 el 15-jun-2026: en los primeros 8 partidos con cuotas el
mercado (RPS 0.180) batió al prior (0.200) y a la mezcla 50/50 (0.191), así que
se le da más peso. Pendiente de fijar formalmente con más muestra.
"""
import numpy as np

W_MARKET = 0.65


def linear_pool(probs_list, weights) -> np.ndarray:
    P = np.array(probs_list, dtype=float)
    w = np.array(weights, dtype=float)
    w = w / w.sum()
    blended = (P * w[:, None]).sum(axis=0)
    return blended / blended.sum()


def blend_market(model_probs, market_probs, w_market: float = W_MARKET):
    """Devuelve (probs_finales, uso_mercado). Si no hay cuotas, pasa el modelo."""
    model_probs = np.array(model_probs, dtype=float)
    if market_probs is None:
        return model_probs / model_probs.sum(), False
    return linear_pool([model_probs, market_probs], [1 - w_market, w_market]), True
