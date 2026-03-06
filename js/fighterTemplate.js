/**
 * Fighter Template Module
 * Generates fighter HTML structure
 */

/**
 * Fighter template configurations
 */
const fighterTemplates = {
    default: {
        bodyColor: '#ff6b6b',
        headColor: '#ffd93d',
        gloveColor: '#c0392b',
        pantsColor: '#3498db'
    },
    gpt4: {
        bodyColor: '#00a67d',
        headColor: '#ffd93d',
        gloveColor: '#006644',
        pantsColor: '#004d33'
    },
    claude: {
        bodyColor: '#d97706',
        headColor: '#ffd93d',
        gloveColor: '#92400e',
        pantsColor: '#78350f'
    },
    qwenCoder: {
        bodyColor: '#6ef2ff',
        headColor: '#ffd93d',
        gloveColor: '#16b8c7',
        pantsColor: '#0d7e8a'
    },
    groqFast: {
        bodyColor: '#ffb347',
        headColor: '#ffd93d',
        gloveColor: '#f08a24',
        pantsColor: '#b85e11'
    }
};

/**
 * Get template based on fighter ID
 * @param {string} fighterId - The fighter ID
 * @returns {object} - The template configuration
 */
function getTemplateForFighter(fighterId) {
    const templateMap = {
        '1': 'gpt4',
        '2': 'claude',
        '3': 'qwenCoder',
        '4': 'groqFast'
    };
    return fighterTemplates[templateMap[fighterId] || 'default'];
}

/**
 * Create fighter HTML structure
 * @param {string} fighterId - The fighter ID
 * @param {number} playerNum - Player number (1 or 2)
 * @returns {string} - HTML string for the fighter
 */
function createFighterHTML(fighterId, playerNum) {
    const template = getTemplateForFighter(fighterId);

    return `
        <div class="boxer" id="boxer-p${playerNum}" data-player="${playerNum}">
            <div class="boxer-head" style="background: ${template.headColor}"></div>
            <div class="boxer-body" style="background: ${template.bodyColor}">
                <div class="boxer-arm left" style="background: ${template.bodyColor}">
                    <div class="boxer-glove" style="background: ${template.gloveColor}"></div>
                </div>
                <div class="boxer-arm right" style="background: ${template.bodyColor}">
                    <div class="boxer-glove" style="background: ${template.gloveColor}"></div>
                </div>
            </div>
            <div class="boxer-leg left" style="background: ${template.pantsColor}"></div>
            <div class="boxer-leg right" style="background: ${template.pantsColor}"></div>
        </div>
    `;
}

/**
 * Inject fighter into container
 * @param {string} containerId - The container element ID
 * @param {string} fighterId - The fighter ID
 * @param {number} playerNum - Player number (1 or 2)
 */
function injectFighter(containerId, fighterId, playerNum) {
    const container = document.getElementById(containerId);
    if (container) {
        container.innerHTML = createFighterHTML(fighterId, playerNum);
    }
}

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { createFighterHTML, injectFighter, fighterTemplates };
}
