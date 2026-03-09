"""
Optional live commentary generation for the arena.

Uses Sarvam AI when configured. The rest of the app should continue to work
without the SDK or API key.
"""

import os

from dotenv import load_dotenv
import requests

load_dotenv()

try:
    from sarvamai import SarvamAI
except ImportError:
    SarvamAI = None


SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "").strip()
SARVAM_MODEL = os.getenv("SARVAM_MODEL", "sarvam-30b") or "sarvam-30b"
SARVAM_REASONING_EFFORT = os.getenv("SARVAM_REASONING_EFFORT", "low").strip() or "low"
SARVAM_TIMEOUT = max(3, int(os.getenv("SARVAM_TIMEOUT", "8") or "8"))
SARVAM_TTS_URL = os.getenv("SARVAM_TTS_URL", "https://api.sarvam.ai/text-to-speech").strip()
SARVAM_TTS_MODEL = os.getenv("SARVAM_TTS_MODEL", "bulbul:v3").strip() or "bulbul:v3"
SARVAM_TTS_LANGUAGE_CODE = os.getenv("SARVAM_TTS_LANGUAGE_CODE", "en-IN").strip() or "en-IN"
SARVAM_TTS_SPEAKER = os.getenv("SARVAM_TTS_SPEAKER", "Rahul").strip() or "Rahul"
SARVAM_TTS_PACE = max(0.5, min(2.0, float(os.getenv("SARVAM_TTS_PACE", "1.08") or "1.08")))
SARVAM_TTS_SAMPLE_RATE = int(os.getenv("SARVAM_TTS_SAMPLE_RATE", "24000") or "24000")
SARVAM_TTS_TEMPERATURE = max(0.01, min(2.0, float(os.getenv("SARVAM_TTS_TEMPERATURE", "0.45") or "0.45")))
SARVAM_TTS_CODEC = os.getenv("SARVAM_TTS_CODEC", "mp3").strip().lower() or "mp3"

TTS_MODEL_SPEAKERS = {
    "bulbul:v3": {
        "shubh", "aditya", "ritu", "priya", "neha", "rahul", "pooja", "rohan", "simran", "kavya",
        "amit", "dev", "ishita", "shreya", "ratan", "varun", "manan", "sumit", "roopa", "kabir",
        "aayan", "ashutosh", "advait", "amelia", "sophia", "anand", "tanya", "tarun", "sunny",
        "mani", "gokul", "vijay", "shruti", "suhani", "mohit", "kavitha", "rehan", "soham", "rupali",
    },
    "bulbul:v2": {
        "anushka", "manisha", "vidya", "arya", "abhilash", "karun", "hitesh",
    },
}

TTS_DEFAULT_SPEAKER = {
    "bulbul:v3": "shubh",
    "bulbul:v2": "anushka",
}

CODEC_MIME = {
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "aac": "audio/aac",
    "opus": "audio/ogg",
    "flac": "audio/flac",
    "pcm": "audio/wav",
    "mulaw": "audio/basic",
    "alaw": "audio/basic",
}

_CLIENT = None


def commentary_available():
    return bool(SARVAM_API_KEY and SarvamAI is not None)


def _get_client():
    global _CLIENT
    if _CLIENT is None and commentary_available():
        _CLIENT = SarvamAI(api_subscription_key=SARVAM_API_KEY)
    return _CLIENT


def _clip(text, limit):
    value = " ".join(str(text or "").split())
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip() + "..."


def _event_digest(turn_events):
    highlights = []
    for event in turn_events or []:
        text = _clip(event.get("text", ""), 120)
        if text:
            highlights.append(text)
    return " | ".join(highlights[:3]) or "No clean exchanges yet."


def _normalize_tts_model(model_name):
    model = str(model_name or "bulbul:v3").strip().lower()
    return model if model in TTS_MODEL_SPEAKERS else "bulbul:v3"


def _normalize_tts_speaker(model_name, speaker_name):
    model = _normalize_tts_model(model_name)
    speaker = str(speaker_name or "").strip().lower()
    if speaker in TTS_MODEL_SPEAKERS[model]:
        return speaker
    return TTS_DEFAULT_SPEAKER[model]


_CAPS_WORD = __import__('re').compile(r'\b([A-Z]{2,})\b')


def _normalize_tts_text(text):
    """Convert all-caps words like OH, WOW, YES to sentence-case so TTS doesn't spell them out."""
    def _to_word(m):
        w = m.group(1)
        # Keep known acronyms/abbreviations as-is
        keep = {'AI', 'LLM', 'OK', 'KO', 'RL', 'TTS', 'API'}
        if w in keep:
            return w
        return w.capitalize()
    return _CAPS_WORD.sub(_to_word, text)


def _synthesize_commentary_audio(text):
    if not text:
        return {"audio_base64": None, "audio_mime": None, "audio_error": "No commentary text"}
    text = _normalize_tts_text(text)

    tts_model = _normalize_tts_model(SARVAM_TTS_MODEL)
    tts_speaker = _normalize_tts_speaker(tts_model, SARVAM_TTS_SPEAKER)
    codec = SARVAM_TTS_CODEC if SARVAM_TTS_CODEC in CODEC_MIME else "mp3"

    payload = {
        "text": _clip(text, 400),
        "target_language_code": SARVAM_TTS_LANGUAGE_CODE,
        "speaker": tts_speaker,
        "pace": SARVAM_TTS_PACE,
        "speech_sample_rate": SARVAM_TTS_SAMPLE_RATE,
        "model": tts_model,
        "output_audio_codec": codec,
        "temperature": SARVAM_TTS_TEMPERATURE,
    }
    headers = {
        "api-subscription-key": SARVAM_API_KEY,
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(SARVAM_TTS_URL, json=payload, headers=headers, timeout=SARVAM_TIMEOUT)
    except Exception as exc:
        return {
            "audio_base64": None,
            "audio_mime": None,
            "audio_error": _clip(str(exc), 180),
        }

    if response.status_code != 200:
        return {
            "audio_base64": None,
            "audio_mime": None,
            "audio_error": _clip(f"{response.status_code}: {response.text}", 180),
        }

    data = response.json()
    audios = data.get("audios") or []
    audio_base64 = audios[0] if audios else None
    return {
        "audio_base64": audio_base64,
        "audio_mime": CODEC_MIME.get(codec, "audio/mpeg"),
        "audio_error": None if audio_base64 else "Sarvam TTS returned no audio",
    }


def generate_live_commentary(topic, turn_number, p1, p2, turn_events, audience):
    if not commentary_available():
        return None

    client = _get_client()
    if client is None:
        return None

    prompt = f"""You are the HYPE ringside commentator for an AI boxing match where fighters also debate.
Deliver ONE explosive commentary line under 55 words — use energy, exclamation, dramatic pauses (use commas or dashes for rhythm).
Vary your tone: sometimes a shout, sometimes a breathless rush, sometimes a slow build.
Mention the debate clash AND the combat momentum. No markdown, no bullets.
Examples of ideal tone: "OH! {p1['name']} LANDS it — and that argument cuts DEEP!" or "wait — {p2['name']} ducks, pivots, and drops a bomb on the topic!"

Turn: {turn_number}
Topic: {_clip(topic or 'No explicit topic', 180)}
Audience cheers: {p1['name']}={audience['p1']['cheers']} | {p2['name']}={audience['p2']['cheers']}

Fighter 1: {p1['name']}
Debate: {_clip(p1.get('debate') or p1.get('thinking') or 'No argument delivered.', 240)}
Move: {p1.get('move', 'UNKNOWN')}

Fighter 2: {p2['name']}
Debate: {_clip(p2.get('debate') or p2.get('thinking') or 'No argument delivered.', 240)}
Move: {p2.get('move', 'UNKNOWN')}

Fight events: {_event_digest(turn_events)}
"""

    try:
        response = client.chat.completions(
            model=SARVAM_MODEL,
            reasoning_effort=SARVAM_REASONING_EFFORT,
            messages=[
                {
                    "role": "system",
                    "content": "You are an electrifying, high-energy sports commentator. Speak with raw excitement — shout, gasp, rush your words. Be unpredictable: sometimes a roar, sometimes a whisper that builds. Use dramatic punctuation for rhythm and intonation.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )
        choices = getattr(response, "choices", []) or []
        if not choices:
            return {
                "provider": "sarvam",
                "model": SARVAM_MODEL,
                "text": "",
                "error": "Empty commentary response",
            }
        message = getattr(choices[0], "message", None)
        text = getattr(message, "content", "") if message is not None else ""
        tts = _synthesize_commentary_audio(text)
        return {
            "provider": "sarvam",
            "model": SARVAM_MODEL,
            "text": _clip(text, 280),
            "audio_base64": tts.get("audio_base64"),
            "audio_mime": tts.get("audio_mime"),
            "audio_error": tts.get("audio_error"),
            "error": None,
        }
    except Exception as exc:
        return {
            "provider": "sarvam",
            "model": SARVAM_MODEL,
            "text": "",
            "audio_base64": None,
            "audio_mime": None,
            "audio_error": None,
            "error": _clip(str(exc), 180),
        }