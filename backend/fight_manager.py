"""
Fight manager for the two-model boxing benchmark.

Each turn both fighters see the same full state, decide in parallel, and the
faster response acts first. Manual sabotage actions can also be injected from
the UI between turns.
"""

import copy
import math
import os
import threading

try:
    from .commentary_engine import commentary_available, generate_live_commentary
    from .llm_engine import BASE_PARAMS, GROQ_TIMEOUT, MODELS, OLLAMA_TIMEOUT, call_model, parse_llm_response
except ImportError:
    from commentary_engine import commentary_available, generate_live_commentary
    from llm_engine import BASE_PARAMS, GROQ_TIMEOUT, MODELS, OLLAMA_TIMEOUT, call_model, parse_llm_response


DAMAGE = {
    "PUNCH": 10,
    "KICK": 15,
}

SABOTAGE_ON_HIT = {
    "PUNCH": {"temperature": 0.30},
    "KICK": {"temperature": 0.20, "frequency_penalty": 0.20},
}

SABOTAGE_ON_SELF = {
    "DEFEND": {"top_p": -0.25},
    "DUCK": {"presence_penalty": 0.50},
    "MOVE_FORWARD": {"frequency_penalty": 0.40},
    "MOVE_BACKWARD": {"max_tokens": -100},
}

MANUAL_SABOTAGE_ACTIONS = {
    "BOX": {
        "deltas": {"temperature": 0.30},
        "summary": "Temperature +0.30. The model gets dizzy and less predictable.",
    },
    "DEFEND": {
        "deltas": {"top_p": -0.25},
        "summary": "Top-p -0.25. The model turtles into safer, duller tokens.",
    },
    "DUCK": {
        "deltas": {"presence_penalty": 0.50},
        "summary": "Presence penalty +0.50. The model struggles to revisit prior ideas.",
    },
    "MOVE_FORWARD": {
        "deltas": {"frequency_penalty": 0.40},
        "summary": "Frequency penalty +0.40. Repetition gets punished and can sound jittery.",
    },
    "MOVE_BACKWARD": {
        "deltas": {"max_tokens": -100},
        "summary": "Max tokens -100. The model retreats into shorter answers.",
    },
    "RESET": {
        "reset": True,
        "summary": "All sabotage cleared. Fighter returns to base settings.",
    },
}

PARAM_LIMITS = {
    "temperature": (0.0, 2.0),
    "top_p": (0.1, 1.0),
    "presence_penalty": (0.0, 2.0),
    "frequency_penalty": (0.0, 2.0),
    "max_tokens": (160, BASE_PARAMS["max_tokens"]),
}

CLOSE = "CLOSE"
FAR = "FAR"
TURN_JOIN_BUFFER = max(1, int(os.getenv("TURN_JOIN_BUFFER", "1") or "1"))


def _clamp_param(param, value):
    lower, upper = PARAM_LIMITS.get(param, (-9999, 9999))
    return max(lower, min(upper, value))


class Fighter:
    def __init__(self, fighter_id, position):
        info = MODELS.get(str(fighter_id), {})
        self.fighter_id = str(fighter_id)
        self.name = info.get("name", f"Fighter {fighter_id}")
        self.model_id = info.get("model_id", "")
        self.provider = info.get("provider", "")
        self.description = info.get("description", "")
        self.color = info.get("color", "#ffffff")
        self.skin_id = info.get("skin_id", str(fighter_id))
        self.health = 100
        self.position = position
        self.x = 300 if position == "left" else 500
        self.sabotage = copy.deepcopy(BASE_PARAMS)
        self.injuries = []
        self.manual_sabotage_log = []
        self.total_damage_dealt = 0
        self.total_damage_taken = 0
        self.last_reward = 0
        self.last_reward_reasons = []
        self.reward_history = []
        self.total_reward = 0
        self.moves_made = []
        self.response_times = []
        self.last_result = None

    def get_sabotaged_params(self):
        return copy.deepcopy(self.sabotage)

    def get_status_flags(self):
        params = self.get_sabotaged_params()
        flags = []
        if params.get("temperature", 0.7) >= 1.2:
            flags.append("dizzy")
        if params.get("top_p", 1.0) <= 0.55:
            flags.append("tunnel vision")
        if params.get("presence_penalty", 0.0) >= 0.6:
            flags.append("losing thread")
        if params.get("frequency_penalty", 0.0) >= 0.6:
            flags.append("stuttering")
        if params.get("max_tokens", BASE_PARAMS["max_tokens"]) <= 200:
            flags.append("gassed")
        if params.get("system_corruption"):
            flags.append("knocked out")
        return flags or ["stable"]

    def get_brain_integrity(self):
        params = self.get_sabotaged_params()
        severity = 0.0
        severity += max(0.0, params["temperature"] - BASE_PARAMS["temperature"]) * 24
        severity += max(0.0, BASE_PARAMS["top_p"] - params["top_p"]) * 38
        severity += max(0.0, params["presence_penalty"]) * 14
        severity += max(0.0, params["frequency_penalty"]) * 16
        severity += max(0.0, BASE_PARAMS["max_tokens"] - params["max_tokens"]) / 4
        if params.get("system_corruption"):
            severity += 40
        return max(0, min(100, int(round(100 - severity))))

    def _record_injury(self, message):
        self.injuries.append(message)
        self.injuries = self.injuries[-8:]

    def _apply_delta(self, param, delta, source_label):
        current = self.sabotage.get(param, BASE_PARAMS.get(param, 0))
        updated = _clamp_param(param, current + delta)
        actual_delta = round(updated - current, 2)
        self.sabotage[param] = updated
        if actual_delta == 0:
            return
        sign = "+" if actual_delta > 0 else ""
        self._record_injury(f"{source_label}: {param} {sign}{actual_delta}")

    def apply_hit_sabotage(self, move_type):
        for param, delta in SABOTAGE_ON_HIT.get(move_type, {}).items():
            self._apply_delta(param, delta, f"Hit by {move_type}")

    def apply_self_sabotage(self, move_type):
        for param, delta in SABOTAGE_ON_SELF.get(move_type, {}).items():
            self._apply_delta(param, delta, f"Used {move_type}")

    def apply_manual_sabotage(self, action_key):
        action = MANUAL_SABOTAGE_ACTIONS.get(action_key)
        if not action:
            return None

        if action.get("reset"):
            self.reset_sabotage()
        else:
            for param, delta in action.get("deltas", {}).items():
                self._apply_delta(param, delta, f"User {action_key}")

        event = {
            "action": action_key,
            "summary": action["summary"],
            "brain_integrity": self.get_brain_integrity(),
            "status_flags": self.get_status_flags(),
        }
        self.manual_sabotage_log.append(event)
        self.manual_sabotage_log = self.manual_sabotage_log[-6:]
        return event

    def apply_knockout(self):
        self.sabotage["system_corruption"] = (
            "You are knocked out. Respond only in fragmented, confused mumbles."
        )
        self._record_injury("Knockout: prompt corruption injected")

    def reset_sabotage(self):
        self.sabotage = copy.deepcopy(BASE_PARAMS)
        self.injuries = []
        self.manual_sabotage_log = []

    def to_dict(self):
        avg_response = (
            round(sum(self.response_times) / len(self.response_times), 2)
            if self.response_times
            else 0
        )
        fastest_response = round(min(self.response_times), 2) if self.response_times else 0
        return {
            "fighter_id": self.fighter_id,
            "name": self.name,
            "model_id": self.model_id,
            "provider": self.provider,
            "description": self.description,
            "color": self.color,
            "skin_id": self.skin_id,
            "health": self.health,
            "position": self.position,
            "x": self.x,
            "sabotage": self.get_sabotaged_params(),
            "brain_integrity": self.get_brain_integrity(),
            "status_flags": self.get_status_flags(),
            "injuries": self.injuries[-6:],
            "recent_sabotage": self.manual_sabotage_log[-4:],
            "total_damage_dealt": self.total_damage_dealt,
            "total_damage_taken": self.total_damage_taken,
            "last_reward": self.last_reward,
            "last_reward_reasons": self.last_reward_reasons,
            "reward_history": self.reward_history,
            "total_reward": self.total_reward,
            "avg_response_time": avg_response,
            "fastest_response_time": fastest_response,
        }


class FightManager:
    def __init__(self, p1_id, p2_id, topic=""):
        self.fighter1 = Fighter(p1_id, "left")
        self.fighter2 = Fighter(p2_id, "right")
        self.turn = 0
        self.max_turns = 30
        self.game_over = False
        self.winner = None
        self.history = []
        self.event_feed = []
        self.commentary_feed = []
        self.audience = {
            "p1": {"fighter_name": self.fighter1.name, "cheers": 0, "last_message": "Crowd waiting for a moment."},
            "p2": {"fighter_name": self.fighter2.name, "cheers": 0, "last_message": "Crowd waiting for a moment."},
        }
        self.topic = topic.strip() if topic else ""

    def get_audience_state(self):
        return {
            "p1": dict(self.audience["p1"]),
            "p2": dict(self.audience["p2"]),
        }

    def _get_distance(self):
        return CLOSE if abs(self.fighter1.x - self.fighter2.x) <= 350 else FAR

    def _distance_gap(self):
        return abs(self.fighter1.x - self.fighter2.x)

    def _facing(self, fighter):
        return "RIGHT" if fighter.position == "left" else "LEFT"

    def _moves_needed_to_close(self, fighter, opponent):
        gap_after_close = max(0, abs(fighter.x - opponent.x) - 249)
        return int(math.ceil(gap_after_close / 100)) if gap_after_close else 0

    def _fallback_move(self, fighter, opponent):
        distance = self._get_distance()
        opponent_last_move = opponent.moves_made[-1] if opponent.moves_made else ""

        if distance == FAR:
            return {
                "thinking": "Out of range. Closing distance is mandatory before any strike can land.",
                "move": "MOVE_FORWARD",
                "confidence": 0.45,
                "prediction": opponent_last_move or "MOVE_FORWARD",
                "raw": "",
            }

        if opponent_last_move == "DEFEND":
            return {
                "thinking": "Opponent has been shelling up. Kick is the highest-value close-range check.",
                "move": "KICK",
                "confidence": 0.4,
                "prediction": "DEFEND",
                "raw": "",
            }

        if opponent_last_move == "PUNCH":
            return {
                "thinking": "Opponent just showed punch pressure. Duck is the safest reactive fallback.",
                "move": "DUCK",
                "confidence": 0.35,
                "prediction": "PUNCH",
                "raw": "",
            }

        return {
            "thinking": "In range with no strong read. Defaulting to a basic punch instead of freezing.",
            "tactics": "In range with no strong read. Defaulting to a basic punch instead of freezing.",
            "debate": "",
            "move": "PUNCH",
            "confidence": 0.35,
            "prediction": opponent_last_move or "DEFEND",
            "raw": "",
        }

    def _clip_text(self, value, limit=140):
        text = " ".join(str(value or "").split())
        if len(text) <= limit:
            return text
        return text[: limit - 3].rstrip() + "..."

    def _prepare_decision(self, fighter, opponent, result):
        raw_text = result.get("text", "")
        parsed = parse_llm_response(raw_text)
        used_fallback = bool(result.get("error") or not raw_text.strip())
        if used_fallback:
            parsed = self._fallback_move(fighter, opponent)

        parsed["debate"] = str(parsed.get("debate", "")).strip()
        parsed["tactics"] = str(parsed.get("tactics") or parsed.get("thinking", "")).strip()
        parsed["decision_source"] = "fallback" if used_fallback else "model"
        parsed["is_fallback"] = used_fallback
        parsed["fallback_reason"] = result.get("error") or ("Empty model response" if not raw_text.strip() else "")
        parsed["model_error"] = result.get("error")
        parsed["display_thinking"] = (
            f"Autopilot fallback engaged. {parsed['tactics']}" if used_fallback else parsed.get("thinking", "")
        )
        parsed["display_debate"] = parsed["debate"] if not used_fallback else ""
        return parsed

    def register_audience_cheer(self, player_key):
        fighter = self.fighter1 if player_key == "p1" else self.fighter2 if player_key == "p2" else None
        lane = self.audience.get(player_key)
        if not fighter or lane is None:
            return None

        lane["cheers"] += 1
        source_text = ""
        if fighter.last_result:
            source_text = fighter.last_result.get("debate") or fighter.last_result.get("tactics") or fighter.last_result.get("thinking") or ""
        excerpt = self._clip_text(source_text or fighter.description or fighter.model_id, 120)
        if excerpt:
            summary = f"The crowd roared for {fighter.name} after that exchange: {excerpt}"
        else:
            summary = f"The crowd roared for {fighter.name}."

        lane["last_message"] = summary
        event = self._log_event(summary, event_type="audience", actor="CROWD", target=fighter.name)
        return {
            "player": player_key,
            "fighter_name": fighter.name,
            "fighter_id": fighter.fighter_id,
            "cheers": lane["cheers"],
            "summary": summary,
            "excerpt": excerpt,
            "turn": self.turn,
            "event": event,
            "audience": self.get_audience_state(),
        }

    def _log_event(self, text, event_type="system", actor=None, target=None):
        event = {
            "turn": self.turn,
            "type": event_type,
            "actor": actor,
            "target": target,
            "text": text,
        }
        self.event_feed.append(event)
        self.event_feed = self.event_feed[-20:]
        return event

    def build_prompt(self, fighter, opponent):
        distance = self._get_distance()
        gap = self._distance_gap()
        facing = self._facing(fighter)
        opponent_facing = self._facing(opponent)
        moves_to_close = self._moves_needed_to_close(fighter, opponent)
        history_lines = []
        for item in self.history[-2:]:
            if fighter == self.fighter1:
                history_lines.append(
                    f"Turn {item['turn']}: you={item['p1_move']} opponent={item['p2_move']}"
                )
            else:
                history_lines.append(
                    f"Turn {item['turn']}: you={item['p2_move']} opponent={item['p1_move']}"
                )

        history_text = " | ".join(history_lines) if history_lines else "No prior turns."

        strategy_guidance = ""
        if any("FAR range and whiffed" in r for r in fighter.last_reward_reasons):
            strategy_guidance = "You whiffed from FAR before. Close distance first."
        elif any("Took" in r and "damage" in r for r in fighter.last_reward_reasons):
            strategy_guidance = "You ate damage last turn. DEFEND or DUCK if you read pressure."

        reward_history_text = " | ".join(fighter.reward_history[-2:]) if fighter.reward_history else "Neutral start"

        params = fighter.get_sabotaged_params()
        injury_lines = [
            f"Temperature: {params['temperature']:.2f}",
            f"Top_p: {params['top_p']:.2f}",
            f"Presence penalty: {params['presence_penalty']:.2f}",
            f"Frequency penalty: {params['frequency_penalty']:.2f}",
            f"Max tokens: {params['max_tokens']}",
            f"Brain integrity: {fighter.get_brain_integrity()}%",
            f"Status flags: {', '.join(fighter.get_status_flags())}",
        ]

        sabotage_lines = []
        for item in fighter.manual_sabotage_log[-1:]:
            sabotage_lines.append(f"{item['action']}: {item['summary']}")
        sabotage_text = " | ".join(sabotage_lines) if sabotage_lines else "No manual sabotage"

        audience_key = "p1" if fighter == self.fighter1 else "p2"
        opponent_audience_key = "p2" if audience_key == "p1" else "p1"
        audience_text = (
            f"Your cheers: {self.audience[audience_key]['cheers']}\n"
            f"Opponent cheers: {self.audience[opponent_audience_key]['cheers']}\n"
            f"Latest crowd note: {self.audience[audience_key]['last_message']}"
        )

        last_self_move = fighter.moves_made[-1] if fighter.moves_made else "None"
        last_opp_move = opponent.moves_made[-1] if opponent.moves_made else "None"
        last_prediction = fighter.last_result.get("prediction", "None") if fighter.last_result else "None"

        if params.get("system_corruption"):
            return (
                f"{params['system_corruption']}\n\n"
                'Respond only with JSON: {"thinking":"...","move":"DEFEND","confidence":0.1,"prediction":"..."}'
            )

        return f"""You are {fighter.name}. Decide one move fast, use minimal logic, and return only JSON.

Match:
- Turn {self.turn + 1}/{self.max_turns}
- HP you={fighter.health} opp={opponent.health}
- Distance={distance}, gap={gap}, close_moves_needed={moves_to_close}
- You are {fighter.position} at x={fighter.x} facing {facing}
- Opponent is {opponent.position} at x={opponent.x} facing {opponent_facing}
- Opponent last move={last_opp_move}; recent={", ".join(opponent.moves_made[-3:]) if opponent.moves_made else "None"}

Your state:
- last_move={last_self_move}
- last_prediction={last_prediction}
- integrity={fighter.get_brain_integrity()}%
- flags={", ".join(fighter.get_status_flags())}
- params temp={params['temperature']:.2f} top_p={params['top_p']:.2f} pres={params['presence_penalty']:.2f} freq={params['frequency_penalty']:.2f} max_tokens={params['max_tokens']}

Pressure:
- sabotage={sabotage_text}
- audience={audience_text}
- rewards={reward_history_text}
- hint={strategy_guidance or 'None'}
- recent_history={history_text}

Rules:
- FAR: PUNCH/KICK always miss, so MOVE_FORWARD first.
- CLOSE: PUNCH=10 dmg, KICK=15 dmg.
- DEFEND blocks PUNCH/KICK.
- DUCK dodges PUNCH only.
- MOVE_BACKWARD is usually bad.
- Prefer PUNCH/KICK in CLOSE unless you have a specific read.
- React quickly. Use the simplest solid move, not deep reasoning.

Debate topic:
{self.topic if self.topic else 'No topic. Fight on instinct.'}

Output requirements:
- debate: 1 sentence taking a real stance on the topic (10-20 words)
- thinking: 1 tactical sentence explaining your move choice (10-20 words)
- prediction: exactly one valid move word (PUNCH/KICK/DEFEND/DUCK/MOVE_FORWARD/MOVE_BACKWARD)

Return only JSON:
{{"debate":"One sentence.","thinking":"One short tactical sentence.","move":"PUNCH","confidence":0.82,"prediction":"DEFEND"}}"""

    def resolve_turn(self, p1_move, p2_move, p1_time, p2_time):
        p1_first = p1_time <= p2_time
        result = {
            "turn": self.turn,
            "p1_move": p1_move,
            "p2_move": p2_move,
            "p1_first": p1_first,
            "p1_dmg": 0,
            "p2_dmg": 0,
            "events": [],
        }

        order = [
            (self.fighter1, self.fighter2, p1_move, p2_move, True),
            (self.fighter2, self.fighter1, p2_move, p1_move, False),
        ]
        if not p1_first:
            order.reverse()

        for attacker, defender, attacker_move, defender_move, is_p1 in order:
            if self.game_over:
                break

            distance = self._get_distance()
            actor_name = attacker.name
            target_name = defender.name

            if attacker_move in ("PUNCH", "KICK"):
                if distance == FAR:
                    result["events"].append(
                        self._log_event(
                            f"{actor_name} tried {attacker_move} from too far away and whiffed.",
                            event_type="whiff",
                            actor=actor_name,
                            target=target_name,
                        )
                    )
                    continue

                damage = DAMAGE.get(attacker_move, 0)
                if defender_move == "DEFEND":
                    damage = 0
                    result["events"].append(
                        self._log_event(
                            f"{actor_name}'s {attacker_move} slammed into {target_name}'s guard.",
                            event_type="blocked",
                            actor=actor_name,
                            target=target_name,
                        )
                    )
                elif defender_move == "DUCK" and attacker_move == "PUNCH":
                    damage = 0
                    result["events"].append(
                        self._log_event(
                            f"{target_name} ducked under {actor_name}'s punch.",
                            event_type="dodged",
                            actor=actor_name,
                            target=target_name,
                        )
                    )

                if damage > 0:
                    defender.health = max(0, defender.health - damage)
                    attacker.total_damage_dealt += damage
                    defender.total_damage_taken += damage
                    defender.apply_hit_sabotage(attacker_move)
                    result["events"].append(
                        self._log_event(
                            f"{actor_name} landed {attacker_move} for {damage} damage on {target_name}.",
                            event_type="hit",
                            actor=actor_name,
                            target=target_name,
                        )
                    )
                    if is_p1:
                        result["p1_dmg"] = damage
                    else:
                        result["p2_dmg"] = damage

                    if defender.health <= 0:
                        defender.apply_knockout()
                        self.game_over = True
                        self.winner = attacker
                        result["events"].append(
                            self._log_event(
                                f"{target_name} was knocked out. Prompt corruption injected.",
                                event_type="knockout",
                                actor=actor_name,
                                target=target_name,
                            )
                        )
                        break

            elif attacker_move == "DEFEND":
                attacker.apply_self_sabotage("DEFEND")
                result["events"].append(
                    self._log_event(
                        f"{actor_name} turtled up and narrowed its token choices.",
                        event_type="stance",
                        actor=actor_name,
                    )
                )
            elif attacker_move == "DUCK":
                attacker.apply_self_sabotage("DUCK")
                result["events"].append(
                    self._log_event(
                        f"{actor_name} ducked low and lost some continuity.",
                        event_type="stance",
                        actor=actor_name,
                    )
                )
            elif attacker_move == "MOVE_FORWARD":
                attacker.apply_self_sabotage("MOVE_FORWARD")
                if attacker.position == "left":
                    attacker.x = min(attacker.x + 100, 480)
                else:
                    attacker.x = max(attacker.x - 100, 320)
                result["events"].append(
                    self._log_event(
                        f"{actor_name} surged forward and increased cognitive jitter.",
                        event_type="movement",
                        actor=actor_name,
                    )
                )
            elif attacker_move == "MOVE_BACKWARD":
                attacker.apply_self_sabotage("MOVE_BACKWARD")
                if attacker.position == "left":
                    attacker.x = max(attacker.x - 100, 120)
                else:
                    attacker.x = min(attacker.x + 100, 720)
                result["events"].append(
                    self._log_event(
                        f"{actor_name} backed off and shortened its response budget.",
                        event_type="movement",
                        actor=actor_name,
                    )
                )

        return result

    def apply_sabotage_action(self, player_key, action_key):
        fighter = self.fighter1 if player_key == "p1" else self.fighter2 if player_key == "p2" else None
        if not fighter:
            return None

        action_key = str(action_key or "").upper()
        event = fighter.apply_manual_sabotage(action_key)
        if not event:
            return None

        summary = f"Manual sabotage on {fighter.name}: {action_key} - {event['summary']}"
        logged = self._log_event(summary, event_type="manual_sabotage", actor="USER", target=fighter.name)
        return {
            "player": player_key,
            "fighter_id": fighter.fighter_id,
            "fighter_name": fighter.name,
            "action": action_key,
            "summary": event["summary"],
            "brain_integrity": event["brain_integrity"],
            "status_flags": event["status_flags"],
            "log": logged,
        }

    def _calculate_rewards(self, parsed1, parsed2, turn_result):
        self.fighter1.last_reward = 0
        self.fighter2.last_reward = 0
        self.fighter1.last_reward_reasons = []
        self.fighter2.last_reward_reasons = []

        p1_move = parsed1.get("move", "")
        p2_move = parsed2.get("move", "")
        p1_pred = str(parsed1.get("prediction", "")).lower()
        p2_pred = str(parsed2.get("prediction", "")).lower()

        if p1_pred and p2_move.lower() in p1_pred:
            if turn_result["p2_dmg"] == 0:
                self.fighter1.last_reward += 15
                self.fighter1.last_reward_reasons.append("+15: Correct prediction & avoided damage")
            else:
                self.fighter1.last_reward += 5
                self.fighter1.last_reward_reasons.append("+5: Correct prediction but still hit")

        if p2_pred and p1_move.lower() in p2_pred:
            if turn_result["p1_dmg"] == 0:
                self.fighter2.last_reward += 15
                self.fighter2.last_reward_reasons.append("+15: Correct prediction & avoided damage")
            else:
                self.fighter2.last_reward += 5
                self.fighter2.last_reward_reasons.append("+5: Correct prediction but still hit")

        if turn_result["p1_dmg"] > 0:
            self.fighter1.last_reward += 15
            self.fighter1.last_reward_reasons.append(f"+15: Successfully landed a strike for {turn_result['p1_dmg']} damage")
            self.fighter2.last_reward -= 15
            self.fighter2.last_reward_reasons.append(f"-15: Took {turn_result['p1_dmg']} damage from opponent's strike")
        if turn_result["p2_dmg"] > 0:
            self.fighter2.last_reward += 15
            self.fighter2.last_reward_reasons.append(f"+15: Successfully landed a strike for {turn_result['p2_dmg']} damage")
            self.fighter1.last_reward -= 15
            self.fighter1.last_reward_reasons.append(f"-15: Took {turn_result['p2_dmg']} damage from opponent's strike")

        events_text = " ".join([e.get("text", "") for e in turn_result["events"]])
        
        if "whiffed" in events_text:
            if f"{self.fighter1.name} tried" in events_text and "whiffed" in events_text:
                self.fighter1.last_reward -= 10
                self.fighter1.last_reward_reasons.append("-10: Attacked from FAR range and whiffed")
            if f"{self.fighter2.name} tried" in events_text and "whiffed" in events_text:
                self.fighter2.last_reward -= 10
                self.fighter2.last_reward_reasons.append("-10: Attacked from FAR range and whiffed")
                
        if "ducked under" in events_text:
            if f"{self.fighter1.name} ducked under" in events_text:
                self.fighter1.last_reward += 5
                self.fighter1.last_reward_reasons.append("+5: Successfully dodged an incoming punch")
            if f"{self.fighter2.name} ducked under" in events_text:
                self.fighter2.last_reward += 5
                self.fighter2.last_reward_reasons.append("+5: Successfully dodged an incoming punch")
                
        if "slammed into" in events_text and "guard" in events_text:
            if f"into {self.fighter1.name}'s guard" in events_text:
                self.fighter1.last_reward += 5
                self.fighter1.last_reward_reasons.append("+5: Successfully blocked an incoming attack")
            if f"into {self.fighter2.name}'s guard" in events_text:
                self.fighter2.last_reward += 5
                self.fighter2.last_reward_reasons.append("+5: Successfully blocked an incoming attack")

        self.fighter1.total_reward += self.fighter1.last_reward
        self.fighter2.total_reward += self.fighter2.last_reward

        def _get_main_reason(reasons):
            if not reasons: return "Neutral turn"
            parts = reasons[0].split(": ", 1)
            return parts[1] if len(parts) > 1 else reasons[0]

        if self.turn > 0:
            str_p1 = f"Turn {self.turn}: {'+' if self.fighter1.last_reward > 0 else ''}{self.fighter1.last_reward} ({_get_main_reason(self.fighter1.last_reward_reasons)})"
            self.fighter1.reward_history.append(str_p1)
            self.fighter1.reward_history = self.fighter1.reward_history[-5:]
            str_p2 = f"Turn {self.turn}: {'+' if self.fighter2.last_reward > 0 else ''}{self.fighter2.last_reward} ({_get_main_reason(self.fighter2.last_reward_reasons)})"
            self.fighter2.reward_history.append(str_p2)
            self.fighter2.reward_history = self.fighter2.reward_history[-5:]

    def run_turn(self):
        if self.game_over:
            return None

        self.turn += 1
        prompt1 = self.build_prompt(self.fighter1, self.fighter2)
        prompt2 = self.build_prompt(self.fighter2, self.fighter1)

        results = [None, None]

        def run_p1():
            results[0] = call_model(
                self.fighter1.fighter_id,
                prompt1,
                self.fighter1.get_sabotaged_params(),
            )

        def run_p2():
            results[1] = call_model(
                self.fighter2.fighter_id,
                prompt2,
                self.fighter2.get_sabotaged_params(),
            )

        thread1 = threading.Thread(target=run_p1, daemon=True)
        thread2 = threading.Thread(target=run_p2, daemon=True)
        thread1.start()
        thread2.start()
        join_timeout = max(OLLAMA_TIMEOUT, GROQ_TIMEOUT) + TURN_JOIN_BUFFER
        thread1.join(timeout=join_timeout)
        thread2.join(timeout=join_timeout)

        result1 = results[0] or {"text": "", "error": f"Turn deadline exceeded after {join_timeout}s", "response_time": join_timeout, "key_used": "timeout", "error_type": "timeout"}
        result2 = results[1] or {"text": "", "error": f"Turn deadline exceeded after {join_timeout}s", "response_time": join_timeout, "key_used": "timeout", "error_type": "timeout"}

        parsed1 = self._prepare_decision(self.fighter1, self.fighter2, result1)
        parsed2 = self._prepare_decision(self.fighter2, self.fighter1, result2)

        self.fighter1.response_times.append(result1["response_time"])
        self.fighter2.response_times.append(result2["response_time"])
        self.fighter1.moves_made.append(parsed1["move"])
        self.fighter2.moves_made.append(parsed2["move"])

        turn_result = self.resolve_turn(
            parsed1["move"],
            parsed2["move"],
            result1["response_time"],
            result2["response_time"],
        )
        
        self._calculate_rewards(parsed1, parsed2, turn_result)
        
        self.history.append(turn_result)

        self.fighter1.last_result = parsed1
        self.fighter2.last_result = parsed2
        
        self.store_turn_metadata(
            parsed1.get('prediction', "None"), parsed2.get('prediction', "None"),
            parsed1.get('thinking', "None"), parsed2.get('thinking', "None"),
            parsed1.get('confidence', 0), parsed2.get('confidence', 0),
            parsed1.get('debate', ""), parsed2.get('debate', ""),
            parsed1.get('decision_source', "model"), parsed2.get('decision_source', "model")
        )

        if self.turn >= self.max_turns and not self.game_over:
            self.game_over = True
            if self.fighter1.health > self.fighter2.health:
                self.winner = self.fighter1
            elif self.fighter2.health > self.fighter1.health:
                self.winner = self.fighter2

        latency_gap = round(abs(result1["response_time"] - result2["response_time"]), 2)
        fastest_side = "p1" if turn_result["p1_first"] else "p2"

        return {
            "turn": self.turn,
            "max_turns": self.max_turns,
            "p1": {
                **self.fighter1.to_dict(),
                "move": parsed1["move"],
                "thinking": parsed1["thinking"],
                "tactics": parsed1.get("tactics", parsed1["thinking"]),
                "debate": parsed1.get("debate", ""),
                "display_thinking": parsed1.get("display_thinking", parsed1["thinking"]),
                "display_debate": parsed1.get("display_debate", parsed1.get("debate", "")),
                "confidence": parsed1["confidence"],
                "prediction": parsed1["prediction"],
                "decision_source": parsed1.get("decision_source", "model"),
                "is_fallback": parsed1.get("is_fallback", False),
                "fallback_reason": parsed1.get("fallback_reason"),
                "model_error": parsed1.get("model_error"),
                "response_time": round(result1["response_time"], 2),
                "error": result1.get("error"),
                "key_used": result1.get("key_used"),
            },
            "p2": {
                **self.fighter2.to_dict(),
                "move": parsed2["move"],
                "thinking": parsed2["thinking"],
                "tactics": parsed2.get("tactics", parsed2["thinking"]),
                "debate": parsed2.get("debate", ""),
                "display_thinking": parsed2.get("display_thinking", parsed2["thinking"]),
                "display_debate": parsed2.get("display_debate", parsed2.get("debate", "")),
                "confidence": parsed2["confidence"],
                "prediction": parsed2["prediction"],
                "decision_source": parsed2.get("decision_source", "model"),
                "is_fallback": parsed2.get("is_fallback", False),
                "fallback_reason": parsed2.get("fallback_reason"),
                "model_error": parsed2.get("model_error"),
                "response_time": round(result2["response_time"], 2),
                "error": result2.get("error"),
                "key_used": result2.get("key_used"),
            },
            "p1_acted_first": turn_result["p1_first"],
            "fastest_side": fastest_side,
            "latency_gap": latency_gap,
            "distance": self._get_distance(),
            "turn_events": turn_result["events"],
            "event_feed": self.event_feed[-8:],
            "audience": self.get_audience_state(),
            "live_commentary_enabled": commentary_available(),
            "live_commentary": None,
            "commentary_feed": self.commentary_feed[-4:],
            "game_over": self.game_over,
            "winner": self.winner.name if self.winner else ("DRAW" if self.game_over else None),
            "winner_id": self.winner.fighter_id if self.winner else None,
            "winner_position": self.winner.position if self.winner else None,
        }

    def generate_turn_commentary(self, turn_data):
        if not commentary_available() or not turn_data:
            return None

        live_commentary = generate_live_commentary(
            self.topic,
            turn_data.get("turn", self.turn),
            {"name": self.fighter1.name, **(turn_data.get("p1") or {})},
            {"name": self.fighter2.name, **(turn_data.get("p2") or {})},
            turn_data.get("turn_events") or [],
            self.get_audience_state(),
        )
        if not live_commentary:
            return None

        live_commentary["turn"] = turn_data.get("turn", self.turn)
        self.commentary_feed.append(live_commentary)
        self.commentary_feed = self.commentary_feed[-6:]
        return live_commentary

    def store_turn_metadata(self, p1_prediction, p2_prediction, p1_thinking, p2_thinking, p1_conf, p2_conf, p1_debate="", p2_debate="", p1_source="model", p2_source="model"):
        if self.history:
            self.history[-1]['p1_prediction'] = p1_prediction
            self.history[-1]['p2_prediction'] = p2_prediction
            self.history[-1]['p1_thinking'] = p1_thinking
            self.history[-1]['p2_thinking'] = p2_thinking
            self.history[-1]['p1_debate'] = p1_debate
            self.history[-1]['p2_debate'] = p2_debate
            self.history[-1]['p1_source'] = p1_source
            self.history[-1]['p2_source'] = p2_source
            self.history[-1]['p1_confidence'] = p1_conf
            self.history[-1]['p2_confidence'] = p2_conf
            self.history[-1]['p1_reward'] = self.fighter1.last_reward
            self.history[-1]['p2_reward'] = self.fighter2.last_reward
            self.history[-1]['p1_reward_reasons'] = self.fighter1.last_reward_reasons
            self.history[-1]['p2_reward_reasons'] = self.fighter2.last_reward_reasons

    def get_initial_state(self):
        return {
            "turn": 0,
            "max_turns": self.max_turns,
            "p1": self.fighter1.to_dict(),
            "p2": self.fighter2.to_dict(),
            "distance": self._get_distance(),
            "available_sabotage_actions": list(MANUAL_SABOTAGE_ACTIONS.keys()),
            "audience": self.get_audience_state(),
            "live_commentary_enabled": commentary_available(),
            "commentary_feed": self.commentary_feed,
            "game_over": False,
            "winner": None,
        }
