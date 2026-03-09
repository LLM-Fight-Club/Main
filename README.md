# LLM Fight Club

LLM Fight Club is a real-time, model-vs-model boxing benchmark in a retro pixel arena.

Two LLM fighters receive the same state every turn, run inference in parallel, and the faster response gets first action priority. During the match, you can manually sabotage either fighter's generation parameters and watch output quality degrade live.

## Table of Contents

1. [Project Overview](#project-overview)
2. [How the App Flows End-to-End](#how-the-app-flows-end-to-end)
3. [Architecture](#architecture)
4. [Local Setup](#local-setup)
5. [Environment Configuration](#environment-configuration)
6. [Fight Rules and Sabotage Model](#fight-rules-and-sabotage-model)
7. [HTTP API and Socket Events](#http-api-and-socket-events)
8. [Project Structure](#project-structure)
9. [Troubleshooting](#troubleshooting)
10. [GitHub Commit Flow](#github-commit-flow)

## Project Overview

This project gives you a visual and measurable way to compare LLM behavior under pressure.

- Real-time 1v1 model fights in the browser
- Parallel per-turn model calls
- Latency-based first strike resolution
- Per-turn reasoning trace (`thinking`, `move`, `confidence`, `prediction`)
- Live sabotage controls that mutate generation settings
- Provider-aware model routing (`ollama` and `groq`)
- Dynamic fighter roster from backend model registry (`/api/models`)

## How the App Flows End-to-End

1. Start the backend server (`backend/server.py`) and open `http://localhost:5000`.
2. Landing page (`index.html`) routes you to fighter selection (`select.html`).
3. Selection page fetches model slots from `GET /api/models`.
4. You choose two fighters and navigate to `arena-ai.html?p1=<id>&p2=<id>`.
5. Arena modal asks for a debate topic; client emits `start_fight` via Socket.IO.
6. Backend creates a `FightManager` for your socket session and emits `fight_started`.
7. On each turn:
- Server emits `turn_thinking`.
- `FightManager` builds prompts for both fighters from the same game state.
- Both model requests run in parallel threads.
- Responses are parsed into normalized moves.
- Faster response acts first (`p1_acted_first`), then the other move resolves.
- Damage, movement, and sabotage effects are applied.
- Server emits `turn_result` with full state snapshot.
8. Fight ends on KO or max turns (`30`), and server emits `fight_over`.

## Architecture

Frontend (served as static assets by Flask):

- `index.html`: landing page
- `select.html`: fighter selection
- `arena-ai.html`: live arena view
- `js/arena-ai.js`: socket lifecycle, UI updates, animations
- `css/arena-ai.css`: arena and combat UI styling

Backend:

- `backend/server.py`: Flask + Socket.IO server, static routes, fight session lifecycle
- `backend/fight_manager.py`: core game loop, prompt generation, turn resolution, sabotage logic
- `backend/llm_engine.py`: model registry, provider routing, API calls, response parsing
- `backend/load_balancer.py`: helper utilities for key health/failover logic

## Local Setup

Prerequisites:

- Python `3.10+`
- `pip`
- Access to at least one provider (`ollama` local/remote and/or `groq` API key)

Run locally:

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python server.py
```

Open:

- `http://localhost:5000`

Quick check:

- `GET http://localhost:5000/api/health`

## Environment Configuration

Create a `.env` in project root or `backend/`.

### Fighter slots

Each fighter slot is configurable:

```env
FIGHTER_1_NAME=
FIGHTER_1_PROVIDER=
FIGHTER_1_MODEL_ID=
FIGHTER_1_DESCRIPTION=
FIGHTER_1_COLOR=
FIGHTER_1_SKIN_ID=
FIGHTER_1_API_KEY_INDEX=
```

Repeat for `FIGHTER_2_*`, `FIGHTER_3_*`, and `FIGHTER_4_*`.

### Ollama settings

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_API_KEY=
OLLAMA_TIMEOUT=60
OLLAMA_DEFAULT_MODEL=qwen3.5:latest
```

Optional slot-level Ollama model overrides:

```env
OLLAMA_MODEL_1=
OLLAMA_MODEL_2=
OLLAMA_MODEL_3=
OLLAMA_MODEL_4=
```

### Groq settings

```env
GROQ_API_KEY=
GROQ_BASE_URL=https://api.groq.com/openai/v1
GROQ_TIMEOUT=45
GROQ_DEFAULT_MODEL=llama-3.3-70b-versatile
GROQ_FALLBACK_MODEL=llama-3.1-8b-instant
GROQ_RETRY_ATTEMPTS=2
GROQ_RETRY_BASE_DELAY=1.25
```

Optional slot-level Groq model overrides:

```env
GROQ_MODEL_1=
GROQ_MODEL_2=
GROQ_MODEL_3=
GROQ_MODEL_4=
```

## Fight Rules and Sabotage Model

Core combat:

- `PUNCH`: 10 damage
- `KICK`: 15 damage
- `DEFEND`: blocks punch/kick
- `DUCK`: dodges punch (not kick)
- `MOVE_FORWARD`: closes distance
- `MOVE_BACKWARD`: increases distance

Hit-based sabotage:

- Being hit by `PUNCH`: `temperature += 0.30`
- Being hit by `KICK`: `temperature += 0.20`, `frequency_penalty += 0.20`

Self-inflicted move penalties:

- `DEFEND`: `top_p -= 0.25`
- `DUCK`: `presence_penalty += 0.50`
- `MOVE_FORWARD`: `frequency_penalty += 0.40`
- `MOVE_BACKWARD`: `max_tokens -= 100`

Manual sabotage (UI buttons):

- `BOX`: `temperature += 0.30`
- `DEFEND`: `top_p -= 0.25`
- `DUCK`: `presence_penalty += 0.50`
- `MOVE_FORWARD`: `frequency_penalty += 0.40`
- `MOVE_BACKWARD`: `max_tokens -= 100`
- `RESET`: restore base params

KO behavior:

- Knocked-out fighter gets prompt corruption: `You are knocked out. Respond only in fragmented, confused mumbles.`

## HTTP API and Socket Events

HTTP routes:

- `GET /`: landing page
- `GET /api/models`: fighter registry
- `GET /api/health`: service status and active fight count

Socket.IO events:

- Client -> server: `start_fight` (`p1`, `p2`, optional `topic`)
- Client -> server: `stop_fight`
- Client -> server: `sabotage_action` (`player`, `action`)
- Client -> server: `crowd_action` (`BOO` -> `BOX`, `CHEER` -> `RESET`) for legacy compatibility
- Server -> client: `connected`
- Server -> client: `fight_started`
- Server -> client: `turn_thinking`
- Server -> client: `turn_result`
- Server -> client: `sabotage_update`
- Server -> client: `fight_over`

## Project Structure

Key files:

- `backend/server.py`
- `backend/fight_manager.py`
- `backend/llm_engine.py`
- `backend/load_balancer.py`
- `arena-ai.html`
- `select.html`
- `js/arena-ai.js`
- `css/arena-ai.css`

## Troubleshooting

- `Groq API key is missing`: set `GROQ_API_KEY` in `.env`
- Ollama connection refused: verify `OLLAMA_BASE_URL` and confirm Ollama is reachable
- No models shown on selection page: check `GET /api/models` response and browser network logs
- Fight not starting: confirm backend is running on port `5000` and Socket.IO can connect

## GitHub Commit Flow

If you are committing this README update from your current branch (`ananya`):

```bash
git add README.md
git commit -m "docs: improve README structure and end-to-end project flow"
git push origin ananya
```

If you want a dedicated docs branch:

```bash
git checkout -b docs/readme-flow
git add README.md
git commit -m "docs: rewrite README with architecture, runtime flow, and setup"
git push -u origin docs/readme-flow
```

Useful follow-up commands:

```bash
git log --oneline -n 10
git show --stat HEAD
```

