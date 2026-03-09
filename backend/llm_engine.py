"""
LLM engine for the fight arena.

Supports mixed-model benchmarking with provider-aware routing across Ollama
and Groq.
"""

import json
import os
import re
import time

import requests
from dotenv import load_dotenv

load_dotenv()


def _first_non_empty(*values):
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _clamp(value, lower, upper):
    return max(lower, min(upper, value))


def _to_int(value, default):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_float(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_bool(value, default=False):
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


DEFAULT_MODEL_SLOTS = {
    "1": {
        "name": "Qwen 3.5",
        "model_id": "qwen3.5:latest",
        "provider": "ollama",
        "skin_id": "1",
        "description": "Ollama Cloud multimodal generalist with strong overall utility.",
        "color": "#ffffff",
    },
    "2": {
        "name": "Groq Llama 3.3 70B",
        "model_id": "llama-3.3-70b-versatile",
        "provider": "groq",
        "skin_id": "2",
        "description": "Groq production model optimized for quality with solid reasoning range.",
        "color": "#f55036",
    },
    "3": {
        "name": "GPT OSS 20B",
        "model_id": "openai/gpt-oss-20b",
        "provider": "groq",
        "skin_id": "3",
        "description": "OpenAI open-weight 20B model hosted on Groq.",
        "color": "#6ef2ff",
    },
    "4": {
        "name": "Groq Llama 3.1 8B",
        "model_id": "llama-3.1-8b-instant",
        "provider": "groq",
        "skin_id": "4",
        "description": "Fast Groq production model tuned for low latency.",
        "color": "#ffb347",
    },
}

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "").strip()
OLLAMA_API_KEYS = [
    k for k in [
        OLLAMA_API_KEY,
        os.getenv("OLLAMA_API_KEY_2", "").strip(),
        os.getenv("OLLAMA_API_KEY_3", "").strip(),
    ]
    if k
]
OLLAMA_TIMEOUT = _to_int(os.getenv("OLLAMA_TIMEOUT"), 10)
OLLAMA_RETRY_ATTEMPTS = max(1, _to_int(os.getenv("OLLAMA_RETRY_ATTEMPTS"), 2))
OLLAMA_RETRY_BASE_DELAY = _to_float(os.getenv("OLLAMA_RETRY_BASE_DELAY"), 0.5)
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_TIMEOUT = _to_int(os.getenv("GROQ_TIMEOUT"), 8)
GROQ_RETRY_ATTEMPTS = max(1, _to_int(os.getenv("GROQ_RETRY_ATTEMPTS"), 2))
GROQ_RETRY_BASE_DELAY = _to_float(os.getenv("GROQ_RETRY_BASE_DELAY"), 0.35)
GROQ_MAX_RETRY_WAIT = _to_float(os.getenv("GROQ_MAX_RETRY_WAIT"), 3.0)
GROQ_RATE_LIMIT_COOLDOWN = _to_float(os.getenv("GROQ_RATE_LIMIT_COOLDOWN"), 8.0)
ARENA_ENFORCE_MODEL_COMPATIBILITY = _to_bool(os.getenv("ARENA_ENFORCE_MODEL_COMPATIBILITY"), True)

# Per-model rate-limit cooldown tracking.
_groq_rate_limited_until: dict[str, float] = {}

# Per-fighter call counter for round-robin key rotation.
_ollama_call_counters: dict[str, int] = {}

KNOWN_ARENA_INCOMPATIBLE_MODELS = {
    ("ollama", "gemini-3-flash-preview:cloud"): (
        "Gemini 3 Flash Preview on Ollama cloud returns truncated arena JSON and is disabled by default."
    ),
}


def _arena_support_metadata(provider, model_id):
    normalized_provider = str(provider or "").strip().lower()
    normalized_model = str(model_id or "").strip().lower()
    for (blocked_provider, blocked_model), reason in KNOWN_ARENA_INCOMPATIBLE_MODELS.items():
        if normalized_provider == blocked_provider and normalized_model == blocked_model:
            return False, reason
    return True, ""


def _build_model_registry():
    models = {}
    for slot_id, defaults in DEFAULT_MODEL_SLOTS.items():
        provider = _first_non_empty(
            os.getenv(f"FIGHTER_{slot_id}_PROVIDER"),
            defaults["provider"],
        ).lower()

        if provider == "ollama":
            model_id = _first_non_empty(
                os.getenv(f"FIGHTER_{slot_id}_MODEL_ID"),
                os.getenv(f"OLLAMA_MODEL_{slot_id}"),
                os.getenv("OLLAMA_DEFAULT_MODEL"),
                defaults["model_id"],
            )
        elif provider == "groq":
            model_id = _first_non_empty(
                os.getenv(f"FIGHTER_{slot_id}_MODEL_ID"),
                os.getenv(f"GROQ_MODEL_{slot_id}"),
                os.getenv("GROQ_DEFAULT_MODEL"),
                defaults["model_id"],
            )
        else:
            model_id = _first_non_empty(
                os.getenv(f"FIGHTER_{slot_id}_MODEL_ID"),
                defaults["model_id"],
            )

        arena_supported, arena_warning = _arena_support_metadata(provider, model_id)

        models[slot_id] = {
            "fighter_id": slot_id,
            "skin_id": _first_non_empty(
                os.getenv(f"FIGHTER_{slot_id}_SKIN_ID"),
                defaults.get("skin_id"),
                slot_id,
            ),
            "name": _first_non_empty(
                os.getenv(f"FIGHTER_{slot_id}_NAME"),
                defaults["name"],
            ),
            "model_id": model_id,
            "provider": provider,
            "api_key_index": _to_int(
                os.getenv(f"FIGHTER_{slot_id}_API_KEY_INDEX"),
                defaults.get("api_key_index", 0),
            ),
            "rotate_keys": _to_bool(
                os.getenv(f"FIGHTER_{slot_id}_ROTATE_KEYS"),
                defaults.get("rotate_keys", False),
            ),
            "description": _first_non_empty(
                os.getenv(f"FIGHTER_{slot_id}_DESCRIPTION"),
                defaults["description"],
            ),
            "color": _first_non_empty(
                os.getenv(f"FIGHTER_{slot_id}_COLOR"),
                defaults["color"],
            ),
            "arena_supported": arena_supported,
            "arena_warning": arena_warning,
        }
    return models


MODELS = _build_model_registry()

BASE_PARAMS = {
    "temperature": 0.7,
    "top_p": 1.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "max_tokens": 350,
}

FIGHT_SYSTEM = (
    "You are an LLM boxer in a live benchmark arena. "
    "Return ONLY a single valid JSON object — no prose, no markdown, no text outside the JSON. "
    "Keys: "
    '"debate" (1-2 sentences taking a specific, original stance on the topic — make a concrete argument with real reasoning, never repeat what you said before), '
    '"thinking" (1-2 tactical sentences: read opponent\'s pattern and justify your move choice), '
    '"move" (exactly one of PUNCH, KICK, DEFEND, DUCK, MOVE_FORWARD, MOVE_BACKWARD), '
    '"confidence" (0.0-1.0), '
    '"prediction" (exactly one of PUNCH, KICK, DEFEND, DUCK, MOVE_FORWARD, MOVE_BACKWARD). '
    'Example: {"debate":"Automation historically creates more jobs than it eliminates, but the transition period devastates workers without retraining support.","thinking":"Opponent has punched 3 turns straight, they will punch again so I duck to avoid it.","move":"DUCK","confidence":0.78,"prediction":"PUNCH"}'
)


def get_lb_dashboard():
    return []


def _base_result(elapsed=0.0, text="", error=None, error_type=None, key_used="n/a"):
    return {
        "text": text,
        "error": error,
        "error_type": error_type,
        "response_time": elapsed,
        "key_used": key_used,
    }


def call_ollama(model_id, prompt, params, api_key_index: int = 0):
    """Call Ollama's HTTP API for local or remote models.

    On 5xx errors, rotates through all available API keys before giving up.
    """
    url = f"{OLLAMA_BASE_URL}/api/generate"

    repeat_penalty = 1.0
    repeat_penalty += _clamp(_to_float(params.get("frequency_penalty"), 0.0), 0.0, 2.0) * 0.35
    repeat_penalty += _clamp(_to_float(params.get("presence_penalty"), 0.0), 0.0, 2.0) * 0.2

    payload = {
        "model": model_id,
        "prompt": f"{FIGHT_SYSTEM}\n\n{prompt}",
        "format": "json",
        "stream": False,
        "options": {
            "temperature": _clamp(_to_float(params.get("temperature"), 0.7), 0.0, 2.0),
            "top_p": _clamp(_to_float(params.get("top_p"), 1.0), 0.1, 1.0),
            "num_predict": max(80, _to_int(params.get("max_tokens"), 500)),
            "repeat_penalty": _clamp(repeat_penalty, 1.0, 2.0),
        },
    }

    # Build ordered key rotation: start from preferred index, then cycle through others
    if OLLAMA_API_KEYS:
        num_keys = len(OLLAMA_API_KEYS)
        key_order = [OLLAMA_API_KEYS[(api_key_index + i) % num_keys] for i in range(num_keys)]
    else:
        key_order = [""]

    total_attempts = max(OLLAMA_RETRY_ATTEMPTS, len(key_order))
    call_started = time.time()
    last_result = None

    for attempt in range(total_attempts):
        # Rotate key: use next key in order on each retry
        current_key = key_order[attempt % len(key_order)]
        headers = {}
        if current_key:
            headers["Authorization"] = f"Bearer {current_key}"
        key_label = f"ollama-cloud-key{(api_key_index + attempt) % max(len(key_order), 1)}" if current_key else "ollama-local"

        try:
            response = requests.post(url, json=payload, headers=headers, timeout=OLLAMA_TIMEOUT)
        except Exception as exc:
            last_result = _base_result(
                elapsed=time.time() - call_started,
                error=str(exc),
                error_type="network",
                key_used=key_label,
            )
            if attempt + 1 < total_attempts:
                time.sleep(OLLAMA_RETRY_BASE_DELAY * (attempt + 1))
                continue
            break

        if response.status_code == 200:
            data = response.json()
            return _base_result(
                elapsed=time.time() - call_started,
                text=data.get("response", ""),
                key_used=key_label,
            )

        last_result = _base_result(
            elapsed=time.time() - call_started,
            error=f"{response.status_code}: {response.text[:400]}",
            error_type="api",
            key_used=key_label,
        )
        if response.status_code >= 500 and attempt + 1 < total_attempts:
            time.sleep(OLLAMA_RETRY_BASE_DELAY * (attempt + 1))
            continue
        break

    return last_result or _base_result(
        error="Ollama request failed",
        error_type="api",
        key_used="ollama",
    )


def _normalize_model_id(model_id):
    return str(model_id or "").strip()


def _groq_supports_json_object_response_format(model_id):
    normalized = _normalize_model_id(model_id).lower()
    unsupported_prefixes = (
        "openai/gpt-oss",
        "qwen/qwen3",
    )
    return not any(normalized.startswith(prefix) for prefix in unsupported_prefixes)


def _coerce_groq_text(value):
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(item))
        return "".join(parts)
    if value is None:
        return ""
    return str(value)


def _is_groq_cooling(model_key: str) -> bool:
    """Return True if this Groq model is in a rate-limit cooldown."""
    until = _groq_rate_limited_until.get(model_key, 0.0)
    return time.time() < until


def _set_groq_cooldown(model_key: str, retry_after: float | None = None) -> None:
    """Set a cooldown for a Groq model after hitting a 429."""
    duration = retry_after if (retry_after and 0 < retry_after < 120) else GROQ_RATE_LIMIT_COOLDOWN
    _groq_rate_limited_until[model_key] = time.time() + duration


def _bounded_retry_wait(retry_after, attempt):
    base_wait = GROQ_RETRY_BASE_DELAY * (attempt + 1)
    hinted_wait = retry_after if (retry_after and retry_after > 0) else 0.0
    wait = max(base_wait, hinted_wait)
    return max(0.0, min(wait, GROQ_MAX_RETRY_WAIT))


def call_groq(model_id, prompt, params, fighter_key: str = ""):
    """Call Groq's OpenAI-compatible chat completions API."""
    if not GROQ_API_KEY:
        return _base_result(error="Groq API key is missing", error_type="config", key_used="missing")

    call_started = time.time()
    url = f"{GROQ_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    model_id = _normalize_model_id(model_id)
    if _is_groq_cooling(model_id):
        until = _groq_rate_limited_until[model_id]
        wait = max(0.0, until - time.time())
        bounded_wait = min(wait, GROQ_MAX_RETRY_WAIT)
        if bounded_wait > 0 and GROQ_RETRY_ATTEMPTS > 1:
            time.sleep(bounded_wait)
        elif wait > 0:
            return _base_result(
                elapsed=time.time() - call_started,
                error=f"429: rate limited on {model_id} (cooling {wait:.1f}s)",
                error_type="rate_limit",
                key_used=f"groq:{model_id}",
            )

    if _is_groq_cooling(model_id):
        wait = max(0.0, _groq_rate_limited_until[model_id] - time.time())
        return _base_result(
            elapsed=time.time() - call_started,
            error=f"429: rate limited on {model_id} (cooling {wait:.1f}s)",
            error_type="rate_limit",
            key_used=f"groq:{model_id}",
        )

    payload = {
        "model": model_id,
        "temperature": _clamp(_to_float(params.get("temperature"), 0.7), 0.0, 2.0),
        "top_p": _clamp(_to_float(params.get("top_p"), 1.0), 0.1, 1.0),
        "max_tokens": max(180, _to_int(params.get("max_tokens"), 220)),
        "presence_penalty": _clamp(_to_float(params.get("presence_penalty"), 0.0), -2.0, 2.0),
        "frequency_penalty": _clamp(_to_float(params.get("frequency_penalty"), 0.0), -2.0, 2.0),
        "messages": [
            {"role": "system", "content": FIGHT_SYSTEM},
            {"role": "user", "content": prompt},
        ],
    }
    if _groq_supports_json_object_response_format(model_id):
        payload["response_format"] = {"type": "json_object"}

    last_result = None
    for attempt in range(GROQ_RETRY_ATTEMPTS):
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=GROQ_TIMEOUT)
        except Exception as exc:
            last_result = _base_result(
                elapsed=time.time() - call_started,
                error=str(exc),
                error_type="network",
                key_used=f"groq:{model_id}",
            )
            break

        if response.status_code == 429:
            retry_after = None
            try:
                retry_after = float(response.headers.get("Retry-After", 0) or 0)
            except (ValueError, TypeError):
                pass
            _set_groq_cooldown(model_id, retry_after)
            last_result = _base_result(
                elapsed=time.time() - call_started,
                error=f"429: rate limited on {model_id}",
                error_type="rate_limit",
                key_used=f"groq:{model_id}",
            )
            if attempt + 1 < GROQ_RETRY_ATTEMPTS:
                wait = _bounded_retry_wait(retry_after, attempt)
                if wait > 0:
                    time.sleep(wait)
                continue
            break

        if response.status_code != 200:
            error_body = response.text[:400]
            last_result = _base_result(
                elapsed=time.time() - call_started,
                error=f"{response.status_code}: {error_body}",
                error_type="api",
                key_used=f"groq:{model_id}",
            )
            should_retry = (
                response.status_code >= 500
                or (response.status_code == 400 and "json_validate_fai" in error_body)
            )
            if should_retry and attempt + 1 < GROQ_RETRY_ATTEMPTS:
                time.sleep(GROQ_RETRY_BASE_DELAY * (attempt + 1))
                continue
            break

        data = response.json()
        choices = data.get("choices", [])
        if not choices:
            last_result = _base_result(
                elapsed=time.time() - call_started,
                error="Empty response from Groq",
                error_type="empty",
                key_used=f"groq:{model_id}",
            )
            break

        message = choices[0].get("message", {})
        content = _coerce_groq_text(message.get("content", ""))
        if not content.strip():
            last_result = _base_result(
                elapsed=time.time() - call_started,
                error=f"Empty content from Groq model {model_id}",
                error_type="empty",
                key_used=f"groq:{model_id}",
            )
            break
        return _base_result(
            elapsed=time.time() - call_started,
            text=content,
            key_used=f"groq:{model_id}",
        )

    return last_result or _base_result(
        error="Groq request failed",
        error_type="api",
        key_used="groq",
    )


def call_model(fighter_id, prompt, sabotage_params):
    """Route a fighter request to its configured provider."""
    info = MODELS.get(str(fighter_id))
    if not info:
        return _base_result(error=f"Unknown fighter: {fighter_id}", error_type="config", key_used="n/a")

    if ARENA_ENFORCE_MODEL_COMPATIBILITY and not info.get("arena_supported", True):
        provider = info.get("provider", "unknown")
        model_id = info.get("model_id", "")
        warning = info.get("arena_warning") or "Model is disabled for arena use."
        return _base_result(
            error=warning,
            error_type="config",
            key_used=f"{provider}:{model_id}",
        )

    params = {**BASE_PARAMS, **(sabotage_params or {})}
    provider = info.get("provider", "ollama").lower()

    if provider == "ollama":
        if info.get("rotate_keys") and len(OLLAMA_API_KEYS) > 1:
            fid = str(fighter_id)
            count = _ollama_call_counters.get(fid, 0)
            _ollama_call_counters[fid] = count + 1
            key_idx = count % len(OLLAMA_API_KEYS)
        else:
            key_idx = info.get("api_key_index", 0)
        return call_ollama(info["model_id"], prompt, params, api_key_index=key_idx)

    if provider == "groq":
        return call_groq(info["model_id"], prompt, params, fighter_key=str(fighter_id))

    return _base_result(
        error=f"Unsupported provider: {provider}",
        error_type="config",
        key_used="n/a",
    )


MOVE_ALIASES = {
    "BOX": "PUNCH",
    "MOVE FORWARD": "MOVE_FORWARD",
    "MOVE BACKWARD": "MOVE_BACKWARD",
    "MOVE BACK": "MOVE_BACKWARD",
    "FORWARD": "MOVE_FORWARD",
    "BACKWARD": "MOVE_BACKWARD",
}


def _extract_thinking(data):
    debate = str(data.get("debate", "")).strip()
    strat = (
        str(data.get("thinking", "")).strip()
        or str(data.get("reasoning", "")).strip()
        or str(data.get("analysis", "")).strip()
        or "No tactical reasoning."
    )
    if debate:
        return f"[DEBATE] {debate} [TACTICS] {strat}"[:1000]
    return strat[:500]


def _extract_tactics(data):
    return (
        str(data.get("thinking", "")).strip()
        or str(data.get("reasoning", "")).strip()
        or str(data.get("analysis", "")).strip()
        or "No tactical reasoning."
    )[:500]


def _extract_debate(data):
    return str(data.get("debate", "")).strip()[:1000]


def _has_meaningful_text(value, min_words=3, min_chars=12):
    text = " ".join(str(value or "").split()).strip()
    if len(text) < min_chars:
        return False
    words = [word for word in re.split(r"\s+", text) if word]
    return len(words) >= min_words


def _normalize_move(move, default=""):
    raw = str(move if move is not None else default).upper().strip().replace("-", "_")
    if not raw:
        raw = str(default or "").upper().strip().replace("-", "_")
    raw = raw.replace("  ", " ")
    raw = MOVE_ALIASES.get(raw, raw)
    return raw.replace(" ", "_")


def parse_llm_response(text):
    """Parse a model response into a normalized fight move payload."""
    valid_moves = ["PUNCH", "KICK", "DEFEND", "DUCK", "MOVE_FORWARD", "MOVE_BACKWARD"]

    if not text or not text.strip():
        return _invalid("No response from model", text)

    clean = text.strip()
    clean = re.sub(r"<think>.*?</think>", "", clean, flags=re.DOTALL).strip()
    clean = re.sub(r"```(?:json)?\s*", "", clean).strip()

    start = clean.find("{")
    if start == -1:
        return _invalid("Model did not return a JSON object", text)

    json_blob = clean[start:]

    try:
        data = json.loads(json_blob)
        return _from_data(data, valid_moves, text)
    except json.JSONDecodeError:
        pass

    for suffix in ['"}', '"}', "}"]:
        try:
            data = json.loads(json_blob + suffix)
            return _from_data(data, valid_moves, text)
        except json.JSONDecodeError:
            continue

    move_match = re.search(r'"(?:move|action)"\s*:\s*"([^"]+)"', json_blob)
    think_match = re.search(r'"(?:thinking|reasoning|analysis)"\s*:\s*"([^"]*)', json_blob)
    debate_match = re.search(r'"debate"\s*:\s*"([^"]*)', json_blob)
    confidence_match = re.search(r'"confidence"\s*:\s*([\d.]+)', json_blob)
    prediction_match = re.search(r'"prediction"\s*:\s*"([^"]*)', json_blob)

    if move_match:
        move = _normalize_move(move_match.group(1))
        odebate = debate_match.group(1) if debate_match else ""
        ostrat = think_match.group(1) if think_match else ""
        if move in valid_moves and _has_meaningful_text(odebate) and _has_meaningful_text(ostrat):
            if odebate:
                final_think = f"[DEBATE] {odebate} [TACTICS] {ostrat}"[:1000]
            else:
                final_think = ostrat[:500]

            return {
                "thinking": final_think,
                "tactics": ostrat[:500],
                "debate": odebate[:1000],
                "move": move,
                "confidence": float(confidence_match.group(1)) if confidence_match else 0.5,
                "prediction": prediction_match.group(1) if prediction_match else "Unknown",
                "raw": text,
                "valid": True,
                "parse_error": None,
            }

    return _invalid("Model returned malformed or incomplete JSON", text)

def _from_data(data, valid_moves, raw):
    move = _normalize_move(data.get("move", data.get("action", "")))
    if move not in valid_moves:
        bad_move = data.get("move", data.get("action", ""))
        return _invalid(f"Invalid or missing move: {bad_move}", raw)
    debate = _extract_debate(data)
    tactics = _extract_tactics(data)
    if not _has_meaningful_text(debate):
        return _invalid("Missing or too-short debate field", raw)
    if not _has_meaningful_text(tactics):
        return _invalid("Missing or too-short thinking field", raw)
    raw_pred = _normalize_move(str(data.get("prediction", "")))
    prediction = raw_pred if raw_pred in valid_moves else "Unknown"
    return {
        "thinking": _extract_thinking(data),
        "tactics": tactics,
        "debate": debate,
        "move": move,
        "confidence": _clamp(_to_float(data.get("confidence"), 0.5), 0.0, 1.0),
        "prediction": prediction,
        "raw": raw,
        "valid": True,
        "parse_error": None,
    }


def _invalid(message, raw=""):
    return {
        "thinking": message,
        "tactics": message,
        "debate": "",
        "move": "NO_DECISION",
        "confidence": 0.0,
        "prediction": "Unknown",
        "raw": raw,
        "valid": False,
        "parse_error": message,
    }
