/**
 * AI Arena — Game Logic
 * Connects via WebSocket to Python backend, receives LLM decisions,
 * animates fighters, displays chain of thought, and tracks sabotage.
 */

// === URL Params ===
const urlParams = new URLSearchParams(window.location.search);
const p1Selection = urlParams.get('p1') || '1';
const p2Selection = urlParams.get('p2') || '2';

// === DOM Elements ===
const connectOverlay = document.getElementById('connect-overlay');
const fighter1 = document.getElementById('fighter1');
const fighter2 = document.getElementById('fighter2');
const fighter1Wrapper = document.getElementById('fighter1-wrapper');
const fighter2Wrapper = document.getElementById('fighter2-wrapper');
const p1Health = document.getElementById('p1-health');
const p2Health = document.getElementById('p2-health');
const timerEl = document.getElementById('timer');
const turnCounter = document.getElementById('turn-counter');
const roundIndicator = document.getElementById('round-indicator');
const victoryOverlay = document.getElementById('victory-overlay');
const winnerText = document.getElementById('winner-text');
const winnerModel = document.getElementById('winner-model');
const statsGrid = document.getElementById('stats-grid');
const distIndicator = document.getElementById('distance-indicator');
const speedCompare = document.getElementById('speed-compare');
const distanceText = document.getElementById('distance-text');
const cotLogP1 = document.getElementById('cot-log-p1');
const cotLogP2 = document.getElementById('cot-log-p2');

// === Sounds ===
function playSound(id) {
    const s = document.getElementById(id);
    if (s) { s.currentTime = 0; s.play().catch(() => { }); }
}

// === Fighter State Setup ===
function setFighterClass(el, playerNum, state) {
    const facingLeft = el.classList.contains('facing-left');
    el.className = `fighter player${playerNum} ${state}`;
    if (facingLeft) el.classList.add('facing-left');
}

fighter1.className = `fighter player${p1Selection} idle`;
fighter2.className = `fighter player${p2Selection} idle facing-left`;

// === Sabotage UI Updater ===
function updateSabotageUI(prefix, sabotage) {
    const base = { temperature: 0.7, top_p: 1.0, presence_penalty: 0, frequency_penalty: 0, max_tokens: 500 };

    const temp = sabotage.temperature || base.temperature;
    const tp = sabotage.top_p || base.top_p;
    const pp = sabotage.presence_penalty || base.presence_penalty;
    const fp = sabotage.frequency_penalty || base.frequency_penalty;
    const mt = sabotage.max_tokens || base.max_tokens;

    setStatVal(prefix + '-temp', temp.toFixed(2), temp > 1.2 ? 'danger' : temp > 0.9 ? 'warn' : '');
    setStatBar(prefix + '-temp-bar', (temp / 2.0) * 100, temp > 1.2 ? 'danger' : temp > 0.9 ? 'warn' : '');

    setStatVal(prefix + '-topp', tp.toFixed(2), tp < 0.5 ? 'danger' : tp < 0.8 ? 'warn' : '');
    setStatBar(prefix + '-topp-bar', tp * 100, tp < 0.5 ? 'danger' : tp < 0.8 ? 'warn' : '');

    setStatVal(prefix + '-pres', pp.toFixed(2), pp > 0.8 ? 'danger' : pp > 0.3 ? 'warn' : '');
    setStatBar(prefix + '-pres-bar', Math.min(pp / 2, 1) * 100, pp > 0.8 ? 'danger' : pp > 0.3 ? 'warn' : '');

    setStatVal(prefix + '-freq', fp.toFixed(2), fp > 0.8 ? 'danger' : fp > 0.3 ? 'warn' : '');
    setStatBar(prefix + '-freq-bar', Math.min(fp / 2, 1) * 100, fp > 0.8 ? 'danger' : fp > 0.3 ? 'warn' : '');

    setStatVal(prefix + '-tokens', mt, mt < 200 ? 'danger' : mt < 350 ? 'warn' : '');
    setStatBar(prefix + '-tokens-bar', (mt / 500) * 100, mt < 200 ? 'danger' : mt < 350 ? 'warn' : '');
}

function setStatVal(id, val, cls) {
    const el = document.getElementById(id);
    if (el) { el.textContent = val; el.className = 'sab-value' + (cls ? ' ' + cls : ''); }
}
function setStatBar(id, pct, cls) {
    const el = document.getElementById(id);
    if (el) { el.style.width = Math.min(100, pct) + '%'; el.className = 'sab-bar-fill' + (cls ? ' ' + cls : ''); }
}

// === Health Bar Updater ===
function updateHealth(id, hp) {
    const el = document.getElementById(id);
    if (!el) return;
    el.style.width = hp + '%';
    el.classList.remove('low', 'medium');
    if (hp <= 25) el.classList.add('low');
    else if (hp <= 50) el.classList.add('medium');
}

// === Chain of Thought ===
function addCotEntry(logEl, turnNum, move, thinking, confidence, prediction, responseTime) {
    // Remove "latest" from previous entries
    logEl.querySelectorAll('.cot-entry.latest').forEach(e => e.classList.remove('latest'));

    const moveClass = ['DEFEND'].includes(move) ? 'defend' : ['DUCK'].includes(move) ? 'duck' :
        ['MOVE_FORWARD', 'MOVE_BACKWARD'].includes(move) ? 'move' : '';

    const entry = document.createElement('div');
    entry.className = 'cot-entry latest';
    entry.innerHTML = `
        <div class="cot-turn">Turn ${turnNum} · ${responseTime}s</div>
        <div class="cot-move ${moveClass}">▶ ${move}</div>
        <div class="cot-thinking">${escHtml(thinking)}</div>
        <div class="cot-prediction">Prediction: ${escHtml(prediction)}</div>
        <div class="cot-confidence">Confidence: ${(confidence * 100).toFixed(0)}%</div>
    `;
    logEl.prepend(entry);
}

function showThinking(logEl) {
    const existing = logEl.querySelector('.thinking-indicator');
    if (existing) return;
    const div = document.createElement('div');
    div.className = 'thinking-indicator';
    div.textContent = '⏳ Thinking...';
    logEl.prepend(div);
}
function hideThinking(logEl) {
    const el = logEl.querySelector('.thinking-indicator');
    if (el) el.remove();
}

function escHtml(s) {
    const d = document.createElement('div');
    d.textContent = s || '';
    return d.innerHTML;
}

// === Move Animation Map ===
const MOVE_TO_STATE = {
    'PUNCH': 'boxing', 'KICK': 'kicking', 'DEFEND': 'defend',
    'DUCK': 'duck', 'MOVE_FORWARD': 'move-forward', 'MOVE_BACKWARD': 'move-backward',
};
const MOVE_DURATION = {
    'PUNCH': 500, 'KICK': 600, 'DEFEND': 800,
    'DUCK': 600, 'MOVE_FORWARD': 500, 'MOVE_BACKWARD': 500,
};

function animateMove(fighterEl, playerNum, move, isP1, dmgDealt) {
    const state = MOVE_TO_STATE[move] || 'idle';
    setFighterClass(fighterEl, playerNum, state);

    if (move === 'PUNCH' || move === 'KICK') {
        playSound(move === 'KICK' ? 'kick-sound' : 'hit-sound');
    }

    const dur = MOVE_DURATION[move] || 500;
    setTimeout(() => {
        setFighterClass(fighterEl, playerNum, 'idle');
    }, dur);
}

// === Hit Effect ===
function showHitEffect(wrapper) {
    const eff = document.createElement('div');
    eff.className = 'hit-effect';
    eff.style.left = '20px'; eff.style.top = '50px';
    wrapper.appendChild(eff);
    setTimeout(() => eff.remove(), 350);
}

// === Victory Sparkles ===
const SPARKLES = ['✨', '⭐', '💫', '🌟', '✦', '★'];
let sparkleInterval = null;
function spawnSparkle(wrapper) {
    const el = document.createElement('span');
    el.className = 'sparkle';
    el.textContent = SPARKLES[Math.floor(Math.random() * SPARKLES.length)];
    const x = Math.random() * 80 - 20, y = Math.random() * 180;
    const tx = (Math.random() - 0.5) * 120 + 'px', ty = -(30 + Math.random() * 100) + 'px';
    const rot = Math.random() * 360 + 'deg', dur = (0.8 + Math.random() * 0.8) + 's';
    el.style.cssText = `left:${x}px;top:${y}px;--tx:${tx};--ty:${ty};--rot:${rot};--dur:${dur};`;
    wrapper.appendChild(el);
    setTimeout(() => el.remove(), parseFloat(dur) * 1000 + 100);
}
function startSparkles(wrapper) {
    for (let i = 0; i < 10; i++) setTimeout(() => spawnSparkle(wrapper), i * 70);
    sparkleInterval = setInterval(() => spawnSparkle(wrapper), 150);
}
function stopSparkles() {
    if (sparkleInterval) { clearInterval(sparkleInterval); sparkleInterval = null; }
    document.querySelectorAll('.sparkle').forEach(s => s.remove());
}

// ============================================================
//                    SOCKET.IO CONNECTION
// ============================================================

const socket = io('http://localhost:5000', {
    transports: ['websocket', 'polling'],
    reconnection: true,
    reconnectionDelay: 2000,
});

socket.on('connect', () => {
    console.log('[WS] Connected:', socket.id);
    connectOverlay.style.display = 'none';

    // Start the fight
    socket.emit('start_fight', { p1: p1Selection, p2: p2Selection });
});

socket.on('connect_error', (err) => {
    console.error('[WS] Connection error:', err.message);
    connectOverlay.querySelector('p').textContent = 'Connection failed — is the Python server running on port 5000?';
});

socket.on('disconnect', () => {
    console.log('[WS] Disconnected');
});

// === Fight Started ===
socket.on('fight_started', (data) => {
    console.log('[FIGHT] Started:', data);

    // Set names
    document.getElementById('p1-name').textContent = data.p1.name;
    document.getElementById('p2-name').textContent = data.p2.name;
    document.getElementById('p1-model-name').textContent = data.p1.model_id;
    document.getElementById('p2-model-name').textContent = data.p2.model_id;
    document.getElementById('p1-dot').style.background = data.p1.color;
    document.getElementById('p2-dot').style.background = data.p2.color;

    // Set fighter skins
    fighter1.className = `fighter player${p1Selection} idle`;
    fighter2.className = `fighter player${p2Selection} idle facing-left`;

    // Show FIGHT!
    roundIndicator.style.display = 'block';
    roundIndicator.textContent = 'FIGHT!';
    playSound('bell-sound');
    setTimeout(() => { roundIndicator.style.display = 'none'; }, 2500);

    timerEl.textContent = data.max_turns;
    turnCounter.textContent = `TURN 0/${data.max_turns}`;
});

// === Thinking ===
socket.on('turn_thinking', (data) => {
    showThinking(cotLogP1);
    showThinking(cotLogP2);
    turnCounter.textContent = `TURN ${data.turn}/${30} ⏳`;
});

// === Turn Result ===
socket.on('turn_result', (data) => {
    console.log(`[TURN ${data.turn}] P1: ${data.p1.move} (${data.p1.response_time}s) | P2: ${data.p2.move} (${data.p2.response_time}s)`);

    hideThinking(cotLogP1);
    hideThinking(cotLogP2);

    // Update turn counter
    timerEl.textContent = data.max_turns - data.turn;
    turnCounter.textContent = `TURN ${data.turn}/${data.max_turns}`;

    // Animate the faster fighter first
    const p1Num = p1Selection, p2Num = p2Selection;

    if (data.p1_acted_first) {
        animateMove(fighter1, p1Num, data.p1.move, true, data.p1.total_damage_dealt);
        setTimeout(() => animateMove(fighter2, p2Num, data.p2.move, false, data.p2.total_damage_dealt), 300);
    } else {
        animateMove(fighter2, p2Num, data.p2.move, false, data.p2.total_damage_dealt);
        setTimeout(() => animateMove(fighter1, p1Num, data.p1.move, true, data.p1.total_damage_dealt), 300);
    }

    // Hit effects
    if (['PUNCH', 'KICK'].includes(data.p1.move) && data.p2.health < (data.p2.health + (data.p1.move === 'PUNCH' ? 8 : 15))) {
        setTimeout(() => showHitEffect(fighter2Wrapper), 200);
    }
    if (['PUNCH', 'KICK'].includes(data.p2.move) && data.p1.health < (data.p1.health + (data.p2.move === 'PUNCH' ? 8 : 15))) {
        setTimeout(() => showHitEffect(fighter1Wrapper), data.p1_acted_first ? 500 : 200);
    }

    // Update health
    setTimeout(() => {
        updateHealth('p1-health', data.p1.health);
        updateHealth('p2-health', data.p2.health);
    }, 400);

    // Update CoT
    addCotEntry(cotLogP1, data.turn, data.p1.move, data.p1.thinking, data.p1.confidence, data.p1.prediction, data.p1.response_time);
    addCotEntry(cotLogP2, data.turn, data.p2.move, data.p2.thinking, data.p2.confidence, data.p2.prediction, data.p2.response_time);

    // Response times
    document.getElementById('p1-resp-time').textContent = data.p1.response_time + 's';
    document.getElementById('p1-resp-time').className = 'time-val' + (data.p1.response_time > 5 ? ' slow' : '');
    document.getElementById('p2-resp-time').textContent = data.p2.response_time + 's';
    document.getElementById('p2-resp-time').className = 'time-val' + (data.p2.response_time > 5 ? ' slow' : '');

    // Speed compare
    const faster = data.p1_acted_first ? data.p1.name : data.p2.name;
    speedCompare.innerHTML = `⚡ <span class="faster">${faster}</span> acted first`;

    // Distance
    distIndicator.textContent = data.distance;
    distanceText.textContent = `Distance: ${data.distance}`;

    // Sabotage stats
    updateSabotageUI('p1', data.p1.sabotage);
    updateSabotageUI('p2', data.p2.sabotage);
});

// === Fight Over ===
socket.on('fight_over', (data) => {
    console.log('[FIGHT OVER]', data);

    const winPos = data.winner_position;
    const winWrap = winPos === 'left' ? fighter1Wrapper : fighter2Wrapper;
    const loseWrap = winPos === 'left' ? fighter2Wrapper : fighter1Wrapper;
    const winFighter = winPos === 'left' ? fighter1 : fighter2;
    const loseFighter = winPos === 'left' ? fighter2 : fighter1;
    const winNum = winPos === 'left' ? p1Selection : p2Selection;
    const loseNum = winPos === 'left' ? p2Selection : p1Selection;

    if (data.winner && data.winner !== 'DRAW') {
        setFighterClass(winFighter, winNum, 'victory');
        setFighterClass(loseFighter, loseNum, 'defeated');
        playSound('win-sound');
        setTimeout(() => playSound('thud-sound'), 400);
        startSparkles(winWrap);

        setTimeout(() => {
            stopSparkles();
            winnerText.textContent = `${data.winner} WINS!`;
            winnerModel.textContent = `in ${data.turns} turns`;
            statsGrid.innerHTML = `
                <div class="stat-card"><div class="stat-label">P1 Damage Dealt</div><div class="stat-val">${data.p1_final.total_damage_dealt}</div></div>
                <div class="stat-card"><div class="stat-label">P2 Damage Dealt</div><div class="stat-val">${data.p2_final.total_damage_dealt}</div></div>
                <div class="stat-card"><div class="stat-label">P1 Avg Response</div><div class="stat-val">${data.p1_final.avg_response_time}s</div></div>
                <div class="stat-card"><div class="stat-label">P2 Avg Response</div><div class="stat-val">${data.p2_final.avg_response_time}s</div></div>
            `;
            victoryOverlay.style.display = 'flex';
        }, 5000);
    } else {
        winnerText.textContent = 'DRAW!';
        winnerModel.textContent = `${data.turns} turns — no clear winner`;
        victoryOverlay.style.display = 'flex';
    }
});

// Cleanup on page leave
window.addEventListener('beforeunload', () => {
    socket.emit('stop_fight');
    socket.disconnect();
});

// === Crowd Action ===
function sendCrowdAction(player, action) {
    socket.emit('crowd_action', { player, action });
}

socket.on('sabotage_update', (data) => {
    updateSabotageUI('p1', data.p1.sabotage);
    updateSabotageUI('p2', data.p2.sabotage);
});
