# LLM Fight Club

LLM Fight Club is a real-time boxing benchmark where two language models fight in a pixel-art arena. Both models see the same state, decide in parallel, and the faster response acts first. The UI can directly sabotage a fighter's generation settings mid-match so you can watch coherence degrade under pressure.

Inspired by `agentBattleRoyale`, but adapted into a head-to-head boxing format with visible decision traces, latency-based turn order, and manual hyperparameter sabotage.

## What changed

- Dynamic benchmark telemetry in the arena UI
- Visible decision traces for both fighters every turn
- Manual sabotage buttons mapped to backend parameter injuries
- Provider-aware model routing with Groq and Ollama fighters
- Data-driven fighter select screen powered by `/api/models`

## Match loop

1. Both fighters receive the same full arena state.
2. Both models respond in parallel with JSON: strategy summary, move, confidence, prediction.
3. Faster response acts first.
4. Boxing moves and UI sabotage both mutate each fighter's generation parameters.
5. Invalid model output becomes a `NO_DECISION` turn, so that fighter simply stands still.
6. Arena-incompatible models are flagged in the registry instead of being used as defaults.
7. Knockout injects prompt corruption: `"You are knocked out. Respond only in fragmented, confused mumbles."`

## Manual sabotage mapping

- `BOX` -> `temperature += 0.30`
- `DEFEND` -> `top_p -= 0.25`
- `DUCK` -> `presence_penalty += 0.50`
- `MOVE_FORWARD` -> `frequency_penalty += 0.40`
- `MOVE_BACKWARD` -> `max_tokens -= 100`
- `RESET` -> restore base parameters

## Backend model registry

The backend exposes four fighter slots. The default lineup now leans toward Ollama because it is more stable for this arena in the current setup, and each slot can still be overridden with environment variables. Models known to emit broken structured output in arena mode are flagged as unsupported.

Supported providers:

- `ollama`
- `groq`

Per-slot environment variables:

```env
FIGHTER_1_NAME=
FIGHTER_1_PROVIDER=
FIGHTER_1_MODEL_ID=
FIGHTER_1_DESCRIPTION=
FIGHTER_1_COLOR=
FIGHTER_1_API_KEY_INDEX=
```

Ollama settings:

```env
OLLAMA_BASE_URL=https://api.ollama.com
OLLAMA_API_KEY=
OLLAMA_TIMEOUT=15
OLLAMA_DEFAULT_MODEL=devstral-small-2:24b-cloud
OLLAMA_MODEL_4=gemma3:12b
```

Groq settings:

```env
GROQ_API_KEY=
GROQ_BASE_URL=https://api.groq.com/openai/v1
GROQ_TIMEOUT=8
GROQ_DEFAULT_MODEL=llama-3.3-70b-versatile
GROQ_RETRY_ATTEMPTS=3
GROQ_RETRY_BASE_DELAY=1.0
GROQ_MAX_RETRY_WAIT=3.0
GROQ_MODEL_3=llama-3.3-70b-versatile
ARENA_ENFORCE_MODEL_COMPATIBILITY=1
```

Example mixed setup:

```env
FIGHTER_1_NAME=Devstral Small 2 24B
FIGHTER_1_PROVIDER=ollama
FIGHTER_1_MODEL_ID=devstral-small-2:24b-cloud

FIGHTER_2_NAME=Ministral 3 14B
FIGHTER_2_PROVIDER=ollama
FIGHTER_2_MODEL_ID=ministral-3:14b

FIGHTER_3_NAME=Llama 3.3 70B
FIGHTER_3_PROVIDER=groq
FIGHTER_3_MODEL_ID=llama-3.3-70b-versatile

FIGHTER_4_NAME=Gemma 3 12B
FIGHTER_4_PROVIDER=ollama
FIGHTER_4_MODEL_ID=gemma3:12b
```

## Running locally

Backend:

```bash
cd backend
pip install -r requirements.txt
python server.py
```

Frontend:

- Open `http://localhost:5000`
- Choose two fighters on `select.html`
- Start the match and use sabotage buttons on either side panel

## Files that matter

- `backend/llm_engine.py` - provider routing for Groq and Ollama
- `backend/fight_manager.py` - match loop, sabotage model, latency ordering
- `backend/server.py` - Flask + Socket.IO endpoints
- `arena-ai.html` / `js/arena-ai.js` / `css/arena-ai.css` - arena UI and telemetry
- `select.html` - backend-driven fighter selection

## Validation completed

- Python syntax check: `python -m py_compile backend\llm_engine.py backend\fight_manager.py backend\server.py backend\load_balancer.py`
- JS syntax check: `node --check js\arena-ai.js`
- Import smoke test: `from backend.fight_manager import FightManager`
