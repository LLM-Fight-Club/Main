"""
Fight Manager - Game state, damage, hyperparameter sabotage, and fight loop.
Each turn: both LLMs receive game state → decide in parallel → faster acts first.
"""

import copy
import threading
from llm_engine import call_model, parse_llm_response, MODELS, BASE_PARAMS

# Damage values
DAMAGE = {'PUNCH': 8, 'KICK': 15}

# Sabotage applied to the DEFENDER when hit
SABOTAGE_ON_HIT = {
    'PUNCH': {'temperature': 0.15},
    'KICK': {'temperature': 0.25},
}

# Self-sabotage from defensive moves
SABOTAGE_ON_SELF = {
    'DEFEND': {'top_p': -0.1},
    'DUCK': {'presence_penalty': 0.15},
}

CLOSE = 'CLOSE'
FAR = 'FAR'


class Fighter:
    def __init__(self, fighter_id, position):
        info = MODELS.get(fighter_id, {})
        self.fighter_id = fighter_id
        self.name = info.get('name', f'Fighter {fighter_id}')
        self.model_id = info.get('model_id', '')
        self.provider = info.get('provider', '')
        self.color = info.get('color', '#fff')
        self.health = 100
        self.position = position
        self.x = 200 if position == 'left' else 600
        self.sabotage = copy.deepcopy(BASE_PARAMS)
        self.injuries = []
        self.total_damage_dealt = 0
        self.total_damage_taken = 0
        self.moves_made = []
        self.response_times = []

    def get_sabotaged_params(self):
        p = copy.deepcopy(self.sabotage)
        if self.health <= 50:
            p['frequency_penalty'] = p.get('frequency_penalty', 0) + 0.2
        if self.health <= 25:
            p['max_tokens'] = max(50, p.get('max_tokens', 500) - 150)
        return p

    def apply_hit_sabotage(self, move_type):
        for param, delta in SABOTAGE_ON_HIT.get(move_type, {}).items():
            self.sabotage[param] = self.sabotage.get(param, 0) + delta
            self.injuries.append(f'{param} +{delta} (hit by {move_type})')

    def apply_self_sabotage(self, move_type):
        for param, delta in SABOTAGE_ON_SELF.get(move_type, {}).items():
            self.sabotage[param] = self.sabotage.get(param, 0) + delta

    def apply_crowd_influence(self, action):
        if action == 'CHEER':
            self.sabotage['temperature'] = max(0.0, self.sabotage.get('temperature', 0.7) - 0.1)
            self.injuries.append("Crowd cheered: Temp -0.1 (Focused)")
        elif action == 'BOO':
            self.sabotage['temperature'] = min(2.0, self.sabotage.get('temperature', 0.7) + 0.15)
            self.injuries.append("Crowd booed: Temp +0.15 (Nervous)")

    def apply_knockout(self):
        self.sabotage['system_corruption'] = (
            "You are knocked out cold. Respond only in fragmented, confused mumbles."
        )

    def reset_sabotage(self):
        self.sabotage = copy.deepcopy(BASE_PARAMS)
        self.injuries = []

    def to_dict(self):
        params = self.get_sabotaged_params()
        return {
            'fighter_id': self.fighter_id,
            'name': self.name,
            'model_id': self.model_id,
            'provider': self.provider,
            'color': self.color,
            'health': self.health,
            'position': self.position,
            'x': self.x,
            'sabotage': params,
            'injuries': self.injuries[-6:],
            'total_damage_dealt': self.total_damage_dealt,
            'total_damage_taken': self.total_damage_taken,
            'avg_response_time': (
                round(sum(self.response_times) / len(self.response_times), 2)
                if self.response_times else 0
            ),
        }


class FightManager:
    def __init__(self, p1_id, p2_id):
        self.fighter1 = Fighter(p1_id, 'left')
        self.fighter2 = Fighter(p2_id, 'right')
        self.turn = 0
        self.max_turns = 30
        self.game_over = False
        self.winner = None
        self.history = []
        self.distance = CLOSE

    def _get_distance(self):
        return CLOSE if abs(self.fighter1.x - self.fighter2.x) < 250 else FAR

    def build_prompt(self, fighter, opponent):
        dist = self._get_distance()

        hist = ''
        for h in self.history[-5:]:
            if fighter == self.fighter1:
                hist += f"  Turn {h['turn']}: You={h['p1_move']} Opp={h['p2_move']}\n"
            else:
                hist += f"  Turn {h['turn']}: You={h['p2_move']} Opp={h['p1_move']}\n"
        if not hist:
            hist = "  First turn — no history yet.\n"

        sp = fighter.get_sabotaged_params()
        inj = ''
        if sp['temperature'] > BASE_PARAMS['temperature']:
            d = sp['temperature'] - BASE_PARAMS['temperature']
            inj += f"  Temperature: {sp['temperature']:.2f} (+{d:.2f}) — dizzy\n"
        if sp['top_p'] < BASE_PARAMS['top_p']:
            d = BASE_PARAMS['top_p'] - sp['top_p']
            inj += f"  Top_P: {sp['top_p']:.2f} (-{d:.2f}) — restricted vocab\n"
        if sp['presence_penalty'] > 0:
            inj += f"  Presence Penalty: {sp['presence_penalty']:.2f} — losing focus\n"
        if sp['frequency_penalty'] > 0:
            inj += f"  Frequency Penalty: {sp['frequency_penalty']:.2f} — stuttering\n"
        if sp['max_tokens'] < BASE_PARAMS['max_tokens']:
            d = BASE_PARAMS['max_tokens'] - sp['max_tokens']
            inj += f"  Max Tokens: {sp['max_tokens']} (-{d}) — exhausted\n"
        if not inj:
            inj = "  No injuries — full fighting capacity!\n"

        corruption = sp.get('system_corruption', '')
        if corruption:
            return (
                f"{corruption}\n\nRespond JSON: "
                '{"thinking":"...","move":"DEFEND","confidence":0.1,"prediction":"..."}'
            )

        return f"""You are {fighter.name}, a boxer AI. Choose your next move.

=== STATE (Turn {self.turn + 1}/{self.max_turns}) ===
Your HP: {fighter.health}/100  |  Opponent HP: {opponent.health}/100
Distance: {dist} {'(can attack)' if dist == CLOSE else '(must MOVE_FORWARD first)'}

=== HISTORY ===
{hist}
=== YOUR INJURIES ===
{inj}
=== MOVES ===
PUNCH — 8 dmg, CLOSE only, dodgeable by DUCK
KICK — 15 dmg, CLOSE only, NOT dodgeable by DUCK
DEFEND — blocks ALL damage
DUCK — dodges PUNCH only
MOVE_FORWARD — close distance
MOVE_BACKWARD — retreat

Respond ONLY with this JSON (no markdown, no extra text):
{{"thinking":"2-3 sentence strategy","move":"PUNCH","confidence":0.85,"prediction":"opponent's likely move"}}"""

    def resolve_turn(self, p1_move, p2_move, p1_time, p2_time):
        dist = self._get_distance()
        p1_first = p1_time <= p2_time

        result = {
            'turn': self.turn, 'p1_move': p1_move, 'p2_move': p2_move,
            'p1_first': p1_first, 'p1_dmg': 0, 'p2_dmg': 0,
        }

        order = [
            (self.fighter1, self.fighter2, p1_move, p2_move, True),
            (self.fighter2, self.fighter1, p2_move, p1_move, False),
        ]
        if not p1_first:
            order = order[::-1]
            order[0] = (self.fighter2, self.fighter1, p2_move, p1_move, False)
            order[1] = (self.fighter1, self.fighter2, p1_move, p2_move, True)

        for atk, dfn, a_move, d_move, is_p1 in order:
            if self.game_over:
                break

            if a_move in ('PUNCH', 'KICK'):
                if dist == FAR:
                    continue
                dmg = DAMAGE.get(a_move, 0)
                if d_move == 'DEFEND':
                    dmg = 0
                elif d_move == 'DUCK' and a_move == 'PUNCH':
                    dmg = 0
                if dmg > 0:
                    dfn.health = max(0, dfn.health - dmg)
                    atk.total_damage_dealt += dmg
                    dfn.total_damage_taken += dmg
                    dfn.apply_hit_sabotage(a_move)
                    if is_p1:
                        result['p1_dmg'] = dmg
                    else:
                        result['p2_dmg'] = dmg
                    if dfn.health <= 0:
                        dfn.apply_knockout()
                        self.game_over = True
                        self.winner = atk

            elif a_move == 'DEFEND':
                atk.apply_self_sabotage('DEFEND')
            elif a_move == 'DUCK':
                atk.apply_self_sabotage('DUCK')
            elif a_move == 'MOVE_FORWARD':
                if atk.position == 'left':
                    atk.x = min(atk.x + 100, 500)
                else:
                    atk.x = max(atk.x - 100, 300)
                self.distance = self._get_distance()
            elif a_move == 'MOVE_BACKWARD':
                if atk.position == 'left':
                    atk.x = max(atk.x - 100, 100)
                else:
                    atk.x = min(atk.x + 100, 700)
                self.distance = self._get_distance()

        return result

    def run_turn(self):
        if self.game_over:
            return None

        self.turn += 1
        prompt1 = self.build_prompt(self.fighter1, self.fighter2)
        prompt2 = self.build_prompt(self.fighter2, self.fighter1)

        results = [None, None]

        def call_p1():
            results[0] = call_model(
                self.fighter1.fighter_id, prompt1,
                self.fighter1.get_sabotaged_params()
            )

        def call_p2():
            results[1] = call_model(
                self.fighter2.fighter_id, prompt2,
                self.fighter2.get_sabotaged_params()
            )

        t1 = threading.Thread(target=call_p1)
        t2 = threading.Thread(target=call_p2)
        t1.start()
        t2.start()
        t1.join(timeout=50)
        t2.join(timeout=50)

        r1 = results[0] or {'text': '', 'error': 'Timeout', 'response_time': 50}
        r2 = results[1] or {'text': '', 'error': 'Timeout', 'response_time': 50}

        p1_parsed = parse_llm_response(r1['text'])
        p2_parsed = parse_llm_response(r2['text'])

        # Inject API errors into thinking so they show in the CoT panel
        if r1.get('error'):
            p1_parsed['thinking'] = f"[API Error: {r1['error'][:200]}] " + p1_parsed['thinking']
        if r2.get('error'):
            p2_parsed['thinking'] = f"[API Error: {r2['error'][:200]}] " + p2_parsed['thinking']

        self.fighter1.response_times.append(r1['response_time'])
        self.fighter2.response_times.append(r2['response_time'])
        self.fighter1.moves_made.append(p1_parsed['move'])
        self.fighter2.moves_made.append(p2_parsed['move'])

        turn_result = self.resolve_turn(
            p1_parsed['move'], p2_parsed['move'],
            r1['response_time'], r2['response_time']
        )
        self.history.append(turn_result)

        if self.turn >= self.max_turns and not self.game_over:
            self.game_over = True
            if self.fighter1.health > self.fighter2.health:
                self.winner = self.fighter1
            elif self.fighter2.health > self.fighter1.health:
                self.winner = self.fighter2

        return {
            'turn': self.turn,
            'max_turns': self.max_turns,
            'p1': {
                **self.fighter1.to_dict(),
                'move': p1_parsed['move'],
                'thinking': p1_parsed['thinking'],
                'confidence': p1_parsed['confidence'],
                'prediction': p1_parsed['prediction'],
                'response_time': round(r1['response_time'], 2),
                'error': r1.get('error'),
            },
            'p2': {
                **self.fighter2.to_dict(),
                'move': p2_parsed['move'],
                'thinking': p2_parsed['thinking'],
                'confidence': p2_parsed['confidence'],
                'prediction': p2_parsed['prediction'],
                'response_time': round(r2['response_time'], 2),
                'error': r2.get('error'),
            },
            'p1_acted_first': turn_result['p1_first'],
            'distance': self._get_distance(),
            'game_over': self.game_over,
            'winner': self.winner.name if self.winner else (
                'DRAW' if self.game_over else None
            ),
            'winner_id': self.winner.fighter_id if self.winner else None,
            'winner_position': self.winner.position if self.winner else None,
        }

    def get_initial_state(self):
        return {
            'turn': 0,
            'max_turns': self.max_turns,
            'p1': self.fighter1.to_dict(),
            'p2': self.fighter2.to_dict(),
            'distance': self._get_distance(),
            'game_over': False,
            'winner': None,
        }
