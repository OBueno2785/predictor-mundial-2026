# Plan de Implementación — Predictor de Scores Mundial 2026

## 1. Objetivo y contexto

Predecir el marcador y las probabilidades (1X2, marcador exacto, over/under) de:
- Los **72 partidos de la fase de grupos** (11–27 de junio de 2026, 48 selecciones, 12 grupos).
- La **fase eliminatoria** (32avos desde el 28 de junio hasta la final del 19 de julio), generada automáticamente cuando se confirme el bracket.

**Urgencia**: el torneo empezó el 11 de junio — la fase de grupos ya está en curso. El MVP (Fase 1) debe estar operativo en 48 horas; los partidos ya jugados se usan como primeras observaciones para recalibrar.

Principio de diseño: **el modelo estadístico produce el prior; el debate multiagente lo ajusta de forma acotada; la calibración garantiza que las probabilidades finales sean confiables**. Los agentes LLM nunca inventan probabilidades desde cero.

---

## 2. Arquitectura general

```
┌────────────────────────────────────────────────────────────────┐
│ CAPA 1 — INGESTA DE DATOS                                      │
│  fixtures/resultados · cuotas casas · Elo/FIFA · lesiones/     │
│  suspensiones · noticias internas · foros/sentimiento          │
└──────────────┬─────────────────────────────────────────────────┘
               ▼
┌────────────────────────────────────────────────────────────────┐
│ CAPA 2 — MODELO BASE (Dixon-Coles / Poisson bivariado)         │
│  Salida: P(home=i, away=j) → prior 1X2 + marcador más probable │
└──────────────┬─────────────────────────────────────────────────┘
               ▼
┌────────────────────────────────────────────────────────────────┐
│ CAPA 3 — DEBATE MULTIAGENTE (Claude API, claude-opus-4-8)      │
│  A.Estadístico ↔ A.Mercado ↔ A.Plantel/Noticias ↔ A.Sentimiento│
│            └────────► Agente Juez (síntesis) ◄────────┘        │
│  Salida: ajustes acotados en log-odds sobre el prior           │
└──────────────┬─────────────────────────────────────────────────┘
               ▼
┌────────────────────────────────────────────────────────────────┐
│ CAPA 4 — CALIBRACIÓN (obligatoria)                             │
│  ECE + Spiegelhalter Z → temperature scaling / isotónica       │
└──────────────┬─────────────────────────────────────────────────┘
               ▼
┌────────────────────────────────────────────────────────────────┐
│ CAPA 5 — SCHEDULER DE AUTOAJUSTE                               │
│  Job diario (06:00) + Job T-60min por partido → predicción     │
│  final congelada y versionada                                  │
└────────────────────────────────────────────────────────────────┘
```

---

## 3. Capa 1 — Ingesta de datos

| Fuente | Datos | Herramienta |
|---|---|---|
| **API-Football** (api-sports.io) | Fixtures, resultados, alineaciones confirmadas (~T-75min), lesiones y suspensiones por selección | REST, plan gratuito 100 req/día o pago |
| **The Odds API** | Cuotas agregadas (Bet365, Pinnacle, etc.) 1X2 y totales; movimientos de línea | REST, free tier 500 req/mes — presupuestar plan pago para el torneo |
| **eloratings.net + ranking FIFA** | Fuerza base por selección | Scraping/CSV publicado |
| **Resultados históricos internacionales** | Entrenamiento del Dixon-Coles (últimos 4–6 años, ponderación temporal) | Dataset Kaggle "International football results" + actualización diaria |
| **Noticias** (Google News RSS, prensa local) | Lesiones de último minuto, conflictos internos, rotaciones | RSS + búsqueda web del agente |
| **Reddit (r/soccer + subreddits nacionales) y X** | Sentimiento, rumores de vestuario | API Reddit (PRAW) / scraping ligero |

Almacenamiento: **SQLite** + snapshots con timestamp de cada fuente. Cada predicción guarda exactamente qué datos vio (auditabilidad y backtesting honesto, sin fuga de información futura).

Conversión de cuotas a probabilidades: quitar el margen (vig) por normalización proporcional o método de Shin. Las probabilidades implícitas del mercado son el **baseline a batir** y un input del debate.

---

## 4. Capa 2 — Modelo base estadístico

**Dixon-Coles** (Poisson bivariado con corrección para marcadores bajos):

- Parámetros: ataque `α_i` y defensa `β_i` por selección, ventaja de localía `γ` (anfitriones USA/MEX/CAN + ajuste por proximidad geográfica/altitud — p. ej. Azteca a 2.200 m), corrección `ρ` para 0-0/1-0/0-1/1-1.
- Ponderación temporal exponencial: partidos recientes pesan más; amistosos pesan menos que competitivos.
- Reentrenamiento **diario** incorporando los resultados del Mundial mismo.
- Salida: matriz completa `P(home=i, away=j)` para i,j ∈ [0,6+] → de ahí se derivan 1X2, marcador más probable, over/under y distribución para Monte Carlo.

Librerías: `numpy`, `scipy.optimize` (MLE con la verosimilitud de Dixon-Coles), `pandas`. Sin dependencias exóticas.

---

## 5. Capa 3 — Debate multiagente (Claude API)

Transporte: **Claude Code en modo headless** (`claude -p`), autenticado con la suscripción del usuario — sin costo por token, consume cuota del plan. Debatientes con `sonnet` y Juez con `opus` (configurable vía `DEBATE_MODEL` / `DEBATE_MODEL_JUEZ`). Por defecto 1 ronda de debate; `--rounds 2` activa réplicas. Los agentes de noticias usan la búsqueda web integrada (WebSearch). El veredicto del Juez se emite como JSON validado con Pydantic. Si la cuota se agota a mitad de debate, el pipeline degrada al prior sin ajuste (nunca se queda sin predicción).

### Roles

| Agente | Input | Posición que defiende |
|---|---|---|
| **Estadístico** | Salida Dixon-Coles + forma reciente + Elo | El prior del modelo |
| **Mercado** | Probabilidades implícitas sin vig + movimientos de línea de las últimas 24 h | Lo que dice el dinero |
| **Plantel/Noticias** | Lesiones, suspensiones, alineación confirmada, **conflictos internos**, fatiga/viajes | Ajustes por disponibilidad real de jugadores |
| **Sentimiento** | Resumen de foros (Reddit) y prensa local | Señales blandas: moral, presión, ambiente |
| **Juez** | Las 4 posiciones + 2 rondas de réplicas | Síntesis final |

### Protocolo de debate (por partido)

1. **Ronda 1**: cada agente emite su evaluación estructurada: `{p_home, p_draw, p_away, score_mas_probable, confianza, argumentos[]}`.
2. **Ronda 2**: cada agente ve las posiciones de los demás y puede revisar la suya (réplica).
3. **Juez**: produce el ajuste final como **delta en log-odds acotado** sobre el prior del modelo base:
   - `logit(p_final) = logit(p_prior) + δ`, con `|δ| ≤ δ_max` (inicial: 0,35 ≈ máx. ±8 puntos porcentuales).
   - Debe citar qué argumento justifica cada delta (trazabilidad).
4. Total ≈ 9–10 llamadas por partido. **Prompt caching**: el contexto compartido (reglas del debate + datos del partido) va en el prefijo con `cache_control` para que las rondas 2 y el juez lean de caché.

Regla dura: si el juez no encuentra información nueva relevante (sin lesiones, mercado estable), `δ = 0` y la predicción es el prior calibrado. El debate ajusta, no reemplaza.

### Costo

Sin costo monetario por token: las ~7–9 llamadas por partido consumen cuota de la suscripción de Claude Code. Si la cuota del plan resultara insuficiente en días con muchos partidos, las opciones son reducir a 1 ronda de debate, debatir solo partidos con discrepancia modelo-mercado > 5 pp, o volver a la API de pago.

---

## 6. Capa 4 — Calibración (obligatoria, no opcional)

El problema es multiclase (1X2) + distribución de marcadores. Flujo:

1. **Medir** en conjunto de validación (backtest, nunca en datos de entrenamiento):
   - ECE por clase (umbral 0.05) + diagrama de fiabilidad.
   - Spiegelhalter Z por clase (|Z| > 1.96 ⇒ corregir).
2. **Corregir**:
   - Primera opción: **temperature scaling** sobre los log-odds finales post-debate (un solo parámetro T, ajustado por log-loss en validación; con <1000 muestras es lo más robusto).
   - Si hay datos suficientes (backtest con Qatar 2022 + Euro 2024 + Copa América 2024 + eliminatorias ≈ 400+ partidos): comparar contra isotónica one-vs-rest.
3. **Verificar** que la corrección no destruye refinamiento: comparar RPS/Brier y AUC one-vs-rest antes/después; alertar si AUC cae > 0.02.
4. **Recalibrar cada 2–3 jornadas** del torneo conforme entran resultados reales (T se reajusta con la muestra acumulada).

Se calibran **dos cosas por separado**: (a) el prior del Dixon-Coles y (b) el output post-debate. Esto permite medir si el debate multiagente **agrega o destruye** calibración — si la destruye sistemáticamente, se reduce `δ_max`.

---

## 7. Capa 5 — Autoajuste programado

| Job | Cuándo | Qué hace |
|---|---|---|
| **Diario** | 06:00 hora local | Ingesta de resultados de ayer → reentrena Dixon-Coles → refresca cuotas/Elo/noticias → regenera predicciones preliminares de los partidos de hoy y mañana → recalibra si toca |
| **T-60min** | 60 min antes de cada kickoff | Alineaciones confirmadas + últimas cuotas + noticias de última hora → re-ejecuta el debate completo → **predicción final congelada** (inmutable, con hash de los datos usados) |

- Implementación: **APScheduler** en un proceso persistente (lee el fixture y programa los jobs T-60min automáticamente). Alternativa si la máquina no está siempre encendida: Task Scheduler de Windows o un cron en una VM/Cloud Run barato.
- Cada ejecución se versiona en SQLite: `(partido, timestamp, version, prior, deltas, p_final, datos_snapshot_id)`. Permite ver cómo evolucionó la predicción y auditar al juez.
- Manejo de fallos: si una fuente no responde a T-60, se usa el último snapshot y se marca la predicción como `degraded`.

---

## 8. Fase eliminatoria

Al confirmarse el bracket (28 de junio):

1. El job diario detecta los cruces confirmados vía API de fixtures y los incorpora al pipeline sin intervención manual.
2. **Cambios de modelado**: en eliminatorias no hay empate como resultado final ⇒ se reporta `P(resultado en 90')` (sí incluye empate) **y** `P(clasifica)`, modelando prórroga (intensidad de gol reducida ~⅓ por equipo en 30') y penales (prior 50/50 ajustado por historial de tandas y especialistas, input del Agente Plantel).
3. **Monte Carlo del bracket completo** (10.000 simulaciones): P(campeón), P(llegar a semis), etc., actualizado tras cada partido.

---

## 9. Evaluación y backtesting

- **Métricas**: RPS (principal para 1X2), Brier multiclase, log-loss, % acierto de marcador exacto y de signo.
- **Baselines**: (a) probabilidades implícitas de las casas sin vig — el listón real; (b) Dixon-Coles solo, sin debate. El sistema completo debe batir (b) y acercarse o batir (a).
- **Backtest previo a confiar en los ajustes**: Qatar 2022, Euro 2024, Copa América 2024 con datos congelados a T-60min de cada partido (cuotas históricas de football-data.co.uk / OddsPortal). Aquí se calibra T y se elige `δ_max`.
- Dashboard simple (CLI o HTML estático) con predicciones vigentes, métricas acumuladas y curva de fiabilidad.

---

## 10. Stack y estructura

Python 3.11+ con `venv`. Dependencias: `anthropic`, `pandas`, `numpy`, `scipy`, `pydantic`, `APScheduler`, `requests`, `praw`, `matplotlib`, `scikit-learn` (calibración).

```
Predictor_Mundial2026/
├── PLAN.md
├── requirements.txt
├── .env.example              # ANTHROPIC_API_KEY, ODDS_API_KEY, APIFOOTBALL_KEY (nunca commitear .env)
├── data/                     # SQLite + snapshots (gitignored)
├── src/
│   ├── ingest/               # clientes de APIs, scraping, normalización
│   ├── model/                # dixon_coles.py, montecarlo.py
│   ├── agents/               # debate.py, prompts/, schemas.py (Pydantic)
│   ├── calibration/          # metrics.py (ECE, Z, RPS), temperature.py
│   ├── scheduler/            # jobs.py (diario, T-60)
│   └── storage/              # db.py, versioning.py
├── backtest/
└── tests/
```

---

## 11. Fases de implementación

| Fase | Plazo | Entregable | Hito |
|---|---|---|---|
| **1 — MVP estadístico** | Día 1–2 (13–14 jun) | Ingesta de fixtures/resultados/Elo + Dixon-Coles entrenado + predicciones de toda la fase de grupos | Tabla de predicciones publicada |
| **2 — Mercado + multiagente + calibración** | Día 3–5 (15–17 jun) | Cuotas sin vig, debate de 4 agentes + juez, calibración con backtest Qatar 2022 | Sistema completo batiendo al Dixon-Coles solo en backtest |
| **3 — Autoajuste** | Día 6–7 (18–19 jun) | APScheduler con job diario + T-60min, versionado y congelado de predicciones | Primer partido predicho 100% automático |
| **4 — Eliminatorias** | 26–28 jun | Detección automática del bracket, prórroga/penales, Monte Carlo del torneo | P(campeón) actualizada en vivo |

Al cierre de cada fase: commit descriptivo y push a GitHub (`gh repo create predictor-mundial-2026 --public --source=. --remote=origin --push` en la Fase 1; push normal después). El `.gitignore` excluye `data/` y `.env`.

---

## 12. Riesgos

| Riesgo | Mitigación |
|---|---|
| Free tiers insuficientes (Odds API 500 req/mes) | Presupuestar ~US$50–100 en planes pagos durante el torneo; cachear agresivamente |
| Pocos datos de selecciones (xG internacional escaso) | El Dixon-Coles solo necesita goles; xG queda como mejora opcional |
| El debate LLM empeora la calibración | Deltas acotados + medición separada prior vs post-debate + `δ_max` ajustable a 0 |
| Scraping de Reddit/X bloqueado | El Agente Sentimiento es el único prescindible; el sistema degrada sin él |
| Máquina apagada a la hora del job T-60 | Desplegar el scheduler en una VM/Cloud Run mínimo (~US$5/mes) durante el torneo |
| Marcador exacto es intrínsecamente difícil (~10% acierto es bueno) | Comunicar siempre probabilidades, no solo el marcador puntual |
