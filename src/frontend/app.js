/**
 * WABI Chat — Client-side application logic
 * Manages WebSocket connections, user sessions, and UI updates.
 */

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

const state = {
    users: [],
    activeUserId: null,
    ws: null,
    pendingImage: null, // { base64, mimeType }
    userLocation: { lat: null, lng: null }, // Cached user geolocation
    userTimezone: Intl.DateTimeFormat().resolvedOptions().timeZone || null,
    pendingAssistantMessageEl: null,
};

// ---------------------------------------------------------------------------
// DOM Refs
// ---------------------------------------------------------------------------

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const dom = {
    sidebar: $('#sidebar'),
    userList: $('#user-list'),
    emptyState: $('#empty-state'),
    welcomeScreen: $('#welcome-screen'),
    chatView: $('#chat-view'),
    chatAvatar: $('#chat-avatar'),
    chatUserName: $('#chat-user-name'),
    chatStatus: $('#chat-status'),
    messagesScroll: $('#messages-scroll'),
    messageInput: $('#message-input'),
    imageInput: $('#image-input'),
    imagePreview: $('#image-preview'),
    previewImg: $('#preview-img'),
    btnSend: $('#btn-send'),
    btnNewChat: $('#btn-new-chat'),
    btnWelcomeNewChat: $('#btn-welcome-new-chat'),
    btnBack: $('#btn-back'),
    btnDeleteChat: $('#btn-delete-chat'),
    btnRemoveImg: $('#btn-remove-img'),
    searchInput: $('#search-input'),
    modalOverlay: $('#modal-overlay'),
    modalNameInput: $('#modal-name-input'),
    modalCancel: $('#modal-cancel'),
    modalConfirm: $('#modal-confirm'),
    btnProfile: $('#btn-profile'),
    profilePanel: $('#profile-panel'),
    drawerBackdrop: $('#drawer-backdrop'),
    btnCloseProfile: $('#btn-close-profile'),
    btnSaveProfile: $('#btn-save-profile'),
    profInputs: {
        age: $('#prof-age'),
        height_cm: $('#prof-height'),
        weight_kg: $('#prof-weight'),
        gender: $('#prof-gender'),
        health_conditions: $('#prof-health'),
        dietary_preferences: $('#prof-diet'),
        allergies: $('#prof-allergies'),
        fitness_goals: $('#prof-goals'),
    },
};

// ---------------------------------------------------------------------------
// API helpers
// ---------------------------------------------------------------------------

async function apiPost(url, body = {}) {
    const res = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
    });
    return res.json();
}

async function apiGet(url) {
    const res = await fetch(url);
    return res.json();
}

async function apiDelete(url) {
    const res = await fetch(url, { method: 'DELETE' });
    return res.json();
}

// ---------------------------------------------------------------------------
// User Management
// ---------------------------------------------------------------------------

async function loadUsers() {
    state.users = await apiGet('/api/users');
    renderUserList();
}

function renderUserList(filter = '') {
    // Remove old items (keep empty state)
    dom.userList.querySelectorAll('.user-item').forEach((el) => el.remove());

    const filtered = filter
        ? state.users.filter(
              (u) =>
                  u.name.toLowerCase().includes(filter.toLowerCase()) ||
                  u.user_id.toLowerCase().includes(filter.toLowerCase())
          )
        : state.users;

    if (filtered.length === 0) {
        dom.emptyState.style.display = 'flex';
    } else {
        dom.emptyState.style.display = 'none';
    }

    filtered.forEach((user) => {
        const el = document.createElement('div');
        el.className = `user-item${user.user_id === state.activeUserId ? ' active' : ''}`;
        el.dataset.userId = user.user_id;

        const initial = user.name.charAt(0).toUpperCase();
        const timeStr = formatTime(user.last_active);

        el.innerHTML = `
            <div class="avatar">${initial}</div>
            <div class="user-item-info">
                <div class="user-item-name">${escapeHtml(user.name)}</div>
                <div class="user-item-preview">${user.message_count} messages</div>
            </div>
            <div class="user-item-meta">
                <span class="user-item-time">${timeStr}</span>
                <span class="user-item-id">${user.user_id.slice(-8)}</span>
            </div>
        `;

        el.addEventListener('click', () => selectUser(user.user_id));
        dom.userList.insertBefore(el, dom.emptyState);
    });
}

function showNewChatModal() {
    dom.modalOverlay.style.display = 'flex';
    dom.modalNameInput.value = '';
    dom.modalNameInput.focus();
}

function hideModal() {
    dom.modalOverlay.style.display = 'none';
}

async function createUser() {
    const name = dom.modalNameInput.value.trim();
    hideModal();
    const user = await apiPost('/api/users', { name: name || null });
    state.users.push(user);
    renderUserList();
    selectUser(user.user_id);
}

async function deleteCurrentUser() {
    if (!state.activeUserId) return;
    const confirmed = confirm('Delete this conversation? This cannot be undone.');
    if (!confirmed) return;

    await apiDelete(`/api/users/${state.activeUserId}`);
    disconnectWS();
    state.users = state.users.filter((u) => u.user_id !== state.activeUserId);
    state.activeUserId = null;
    renderUserList();
    showWelcome();
}

// ---------------------------------------------------------------------------
// Chat Selection
// ---------------------------------------------------------------------------

async function selectUser(userId) {
    state.activeUserId = userId;
    const user = state.users.find((u) => u.user_id === userId);
    if (!user) return;

    // Update header
    dom.chatAvatar.textContent = user.name.charAt(0).toUpperCase();
    dom.chatUserName.textContent = user.name;
    dom.chatStatus.textContent = 'online';

    // Show chat view
    dom.welcomeScreen.style.display = 'none';
    dom.chatView.style.display = 'flex';

    // Highlight in sidebar
    renderUserList();

    // Mobile: hide sidebar
    dom.sidebar.classList.add('hidden');

    // Load history
    await loadHistory(userId);

    // Connect WebSocket
    connectWS(userId);

    // Focus input
    dom.messageInput.focus();
}

function showWelcome() {
    dom.chatView.style.display = 'none';
    dom.welcomeScreen.style.display = 'flex';
    dom.sidebar.classList.remove('hidden');
}

async function loadHistory(userId) {
    dom.messagesScroll.innerHTML = '';
    clearPendingAssistantMessage();
    const history = await apiGet(`/api/users/${userId}/history`);
    history.forEach((msg) =>
        appendMessage(msg.role, msg.content, msg.timestamp, msg.image_refs || [])
    );
    scrollToBottom();
}

// ---------------------------------------------------------------------------
// WebSocket
// ---------------------------------------------------------------------------

function connectWS(userId) {
    disconnectWS();

    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${location.host}/ws/${userId}`;

    state.ws = new WebSocket(wsUrl);

    state.ws.onopen = () => {
        dom.chatStatus.textContent = 'online';
    };

    state.ws.onmessage = (event) => {
        const data = JSON.parse(event.data);

        if (data.type === 'typing') {
            showThinkingPlaceholder();
        } else if (data.type === 'thinking') {
            removeTypingIndicator();
            updateThinkingIndicator(data);
            scrollToBottom();
        } else if (data.type === 'message') {
            removeTypingIndicator();
            appendOrFinalizeAssistantMessage(data.content, data.timestamp);
            scrollToBottom();
            // Update user's message count
            const user = state.users.find((u) => u.user_id === userId);
            if (user) {
                user.message_count += 1;
                user.last_active = data.timestamp;
                renderUserList();
            }
        } else if (data.type === 'error') {
            removeTypingIndicator();
            clearPendingAssistantMessage();
            appendMessage('assistant', `⚠️ ${data.content}`, data.timestamp);
            scrollToBottom();
        }
    };

    state.ws.onclose = () => {
        dom.chatStatus.textContent = 'disconnected';
    };

    state.ws.onerror = () => {
        dom.chatStatus.textContent = 'error';
    };
}

function disconnectWS() {
    if (state.ws) {
        state.ws.close();
        state.ws = null;
    }
}

// ---------------------------------------------------------------------------
// Sending Messages
// ---------------------------------------------------------------------------

async function sendMessage() {
    const text = dom.messageInput.value.trim();
    if (!text && !state.pendingImage) return;
    if (!state.ws || state.ws.readyState !== WebSocket.OPEN) return;

    clearPendingAssistantMessage();

    // Disable input while waiting
    dom.messageInput.disabled = true;
    dom.btnSend.disabled = true;
    const oldPlaceholder = dom.messageInput.placeholder;
    dom.messageInput.placeholder = "Waiting for location permission...";

    // Wait for the initial location prompt to resolve (allow, deny, or timeout)
    // Or fetch a fresh location if already resolved
    let locationPromise = requestLocation();
    await locationPromise;

    // Re-enable input
    dom.messageInput.disabled = false;
    dom.btnSend.disabled = false;
    dom.messageInput.placeholder = oldPlaceholder;

    const now = new Date().toISOString();

    if (state.pendingImage) {
        // Send image + text
        state.ws.send(
            JSON.stringify({
                type: 'image',
                timezone: state.userTimezone,
                content: state.pendingImage.base64,
                text: text,
                mime_type: state.pendingImage.mimeType,
                lat: state.userLocation.lat,
                lng: state.userLocation.lng,
            })
        );
        // Show in UI via the same figure.chat-image wrapper used on DB reload,
        // so CSS size constraints apply consistently.
        const dataUrl = `data:${state.pendingImage.mimeType};base64,${state.pendingImage.base64}`;
        appendMessage('user', text, now, [{ src: dataUrl }]);
        clearImagePreview();
    } else {
        // Send text only
        state.ws.send(
            JSON.stringify({
                type: 'message',
                content: text,
                lat: state.userLocation.lat,
                lng: state.userLocation.lng,
                timezone: state.userTimezone,
            })
        );
        appendMessage('user', text, now);
    }

    dom.messageInput.value = '';
    dom.messageInput.style.height = 'auto';
    dom.messageInput.focus();
    scrollToBottom();

    // Show thinking container immediately (no three-dot typing state)
    showThinkingPlaceholder();

    // Update user meta
    const user = state.users.find((u) => u.user_id === state.activeUserId);
    if (user) {
        user.message_count += 1;
        user.last_active = now;
        renderUserList();
    }
}

// ---------------------------------------------------------------------------
// Image Handling
// ---------------------------------------------------------------------------

function handleImageSelect(event) {
    const file = event.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
        const dataUrl = e.target.result;
        const base64 = dataUrl.split(',')[1];
        const mimeType = file.type || 'image/jpeg';

        state.pendingImage = { base64, mimeType };

        dom.previewImg.src = dataUrl;
        dom.imagePreview.style.display = 'flex';
        dom.messageInput.focus();
    };
    reader.readAsDataURL(file);
    event.target.value = ''; // reset for re-select
}

function clearImagePreview() {
    state.pendingImage = null;
    dom.imagePreview.style.display = 'none';
    dom.previewImg.src = '';
}

// ---------------------------------------------------------------------------
// Profile Management
// ---------------------------------------------------------------------------

async function openProfilePanel() {
    dom.profilePanel.classList.remove('hidden');
    dom.drawerBackdrop.classList.remove('hidden');
    await loadProfile();
}

function closeProfilePanel() {
    dom.profilePanel.classList.add('hidden');
    dom.drawerBackdrop.classList.add('hidden');
}

async function toggleProfilePanel() {
    if (dom.profilePanel.classList.contains('hidden')) {
        await openProfilePanel();
    } else {
        closeProfilePanel();
    }
}

async function loadProfile() {
    if (!state.activeUserId) return;
    try {
        const profile = await apiGet(`/api/users/${state.activeUserId}/profile`);
        dom.profInputs.age.value = profile.age || '';
        dom.profInputs.height_cm.value = profile.height_cm || '';
        dom.profInputs.weight_kg.value = profile.weight_kg || '';
        dom.profInputs.gender.value = profile.gender || '';
        dom.profInputs.health_conditions.value = profile.health_conditions || '';
        dom.profInputs.dietary_preferences.value = profile.dietary_preferences || '';
        dom.profInputs.allergies.value = profile.allergies || '';
        dom.profInputs.fitness_goals.value = profile.fitness_goals || '';
    } catch (e) {
        console.error('Failed to load profile', e);
    }
}

async function saveProfile() {
    if (!state.activeUserId) return;
    const btn = dom.btnSaveProfile;
    const oldText = btn.textContent;
    btn.textContent = 'Saving...';
    btn.disabled = true;

    const data = {
        age: parseInt(dom.profInputs.age.value) || null,
        height_cm: parseFloat(dom.profInputs.height_cm.value) || null,
        weight_kg: parseFloat(dom.profInputs.weight_kg.value) || null,
        gender: dom.profInputs.gender.value || null,
        health_conditions: dom.profInputs.health_conditions.value || null,
        dietary_preferences: dom.profInputs.dietary_preferences.value || null,
        allergies: dom.profInputs.allergies.value || null,
        fitness_goals: dom.profInputs.fitness_goals.value || null,
    };

    try {
        await fetch(`/api/users/${state.activeUserId}/profile`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data),
        });
        btn.textContent = 'Saved!';
        refreshAllNutritionViz();
        setTimeout(() => {
            btn.textContent = oldText;
            btn.disabled = false;
        }, 1500);
    } catch (e) {
        console.error('Failed to save profile', e);
        btn.textContent = 'Error';
        setTimeout(() => {
            btn.textContent = oldText;
            btn.disabled = false;
        }, 1500);
    }
}

// ---------------------------------------------------------------------------
// Message Rendering
// ---------------------------------------------------------------------------

function renderImageRefs(imageRefs) {
    if (!imageRefs || !imageRefs.length) return '';
    const uid = state.activeUserId;
    return imageRefs.map((ref) => {
        // `src` is set for fresh uploads (inline data URL); otherwise build the
        // API URL from the server-assigned uuid.
        let src = ref.src;
        if (!src) {
            if (!uid || !ref.uuid) return '';
            src = `/api/images/${encodeURIComponent(uid)}/${ref.uuid}`;
        }
        const desc = (ref.description || '').trim();
        const caption = desc ? `<div class="img-caption">${escapeHtml(desc)}</div>` : '';
        return `<figure class="chat-image"><img src="${src}" alt="image" loading="lazy"/>${caption}</figure>`;
    }).join('');
}

function appendMessage(role, content, timestamp, imageRefs) {
    const el = document.createElement('div');
    el.className = `message ${role}`;

    const rendered = renderMarkdown(content);
    const images = renderImageRefs(imageRefs);
    const timeStr = formatTime(timestamp);

    el.innerHTML = `
        <div class="message-content">${rendered}${images}</div>
        <span class="message-time">${timeStr}</span>
    `;

    dom.messagesScroll.appendChild(el);
}

function ensurePendingAssistantMessage() {
    if (state.pendingAssistantMessageEl && state.pendingAssistantMessageEl.isConnected) {
        return state.pendingAssistantMessageEl;
    }

    const el = document.createElement('div');
    el.className = 'message assistant';
    el.innerHTML = `
        <div class="message-content"></div>
        <span class="message-time"></span>
    `;
    dom.messagesScroll.appendChild(el);
    state.pendingAssistantMessageEl = el;
    return el;
}

function appendOrFinalizeAssistantMessage(content, timestamp) {
    const pending = state.pendingAssistantMessageEl;
    if (pending && pending.isConnected) {
        finalizeThinkingContainer(pending);
        const contentEl = pending.querySelector('.message-content');
        contentEl.insertAdjacentHTML('beforeend', renderMarkdown(content));
        pending.querySelector('.message-time').textContent = formatTime(timestamp);
        state.pendingAssistantMessageEl = null;
        return;
    }
    appendMessage('assistant', content, timestamp);
}

function showTypingIndicator() {
    removeTypingIndicator();
    const el = document.createElement('div');
    el.className = 'typing-indicator';
    el.id = 'typing-indicator';
    el.innerHTML = `
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
    `;
    dom.messagesScroll.appendChild(el);
    scrollToBottom();
}

function updateThinkingIndicator(data) {
    const nodeName = data.content || data.node || '';
    const thinkingStep = extractThinkingStep(data, nodeName);
    
    const pendingMessage = ensurePendingAssistantMessage();
    let container = pendingMessage.querySelector('.thinking-container');

    if (!container) {
        container = createThinkingContainer(pendingMessage);
    }
    
    const logs = container.querySelector('.thinking-logs');
    const panelMeta = container.querySelector('.thinking-panel-meta');
    const stage = container.querySelector('.thinking-stage');

    // --- Logic for Appending vs Replacing ---

    // Always clear previous logs when the router provides a new intent.
    // This signifies the start of a new, distinct logical flow.
    if (nodeName === 'intent_router' || nodeName === 'router') {
        logs.innerHTML = '';
    }

    // For the recognition node, handle step-by-step logging.
    if (nodeName === 'recognition') {
        const isFirstStep = thinkingStep.body.includes('1/4');
        // If this is the first step, clear any previous recognition logs from a prior run.
        if (isFirstStep) {
            const oldRecLogs = logs.querySelectorAll('li[data-stream-item^="recognition"]');
            oldRecLogs.forEach(el => el.remove());
        }
        
        // Create a unique key for each step to ensure it's always appended.
        const stepMatch = thinkingStep.body.match(/Step (\d\/\d)/);
        const stepKey = stepMatch ? `recognition-step-${stepMatch[1].replace('/', '-')}` : `recognition-other-${Date.now()}`;

        let li = logs.querySelector(`li[data-stream-item="${stepKey}"]`);
        let isNew = false;
        if (!li) {
            li = document.createElement('li');
            li.dataset.streamItem = stepKey;
            logs.appendChild(li);
            isNew = true;
        }
        li.innerHTML = buildThinkingLogMarkup(thinkingStep);
        li.dataset.status = 'active';
        if (isNew) {
            li.classList.add('thinking-log-enter');
        }

    } else {
        // For all other nodes (router, chitchat, etc.), replace their specific log item.
        let li = logs.querySelector(`li[data-stream-item="${nodeName || 'thinking'}"]`);
        let isNew = false;
        if (!li) {
            li = document.createElement('li');
            li.dataset.streamItem = nodeName || 'thinking';
            logs.appendChild(li);
            isNew = true;
        }
        const markup = buildThinkingLogMarkup(thinkingStep);
        if (li.innerHTML !== markup) {
            li.innerHTML = markup;
        }
        li.dataset.status = 'active';
        if (isNew) {
            li.classList.add('thinking-log-enter');
        }
    }

    syncThinkingLogStates(logs, nodeName === 'recognition'
        ? logs.lastElementChild?.dataset.streamItem
        : (nodeName || 'thinking'));

    // --- Update Summary and Layout ---
    const summary = container.querySelector('.thinking-summary');
    summary.textContent = buildThinkingSummary(thinkingStep.body || thinkingStep.title);
    stage.textContent = '思考中';
    panelMeta.textContent = `已记录 ${logs.children.length} 步`;

    const content = container.querySelector('.thinking-content');
    if (container.classList.contains('expanded')) {
        content.style.maxHeight = `${content.scrollHeight}px`;
    }
}

function createThinkingContainer(parentMessageEl) {
    const container = document.createElement('div');
    container.className = 'thinking-container';
    container.innerHTML = `
        <button type="button" class="thinking-chip" aria-expanded="false">
            <span class="thinking-icon" aria-hidden="true">
                <span class="thinking-icon-core"></span>
            </span>
            <span class="thinking-head">
                <span class="thinking-title">思考过程</span>
                <span class="thinking-summary">正在整理过程...</span>
            </span>
            <span class="thinking-stage">思考中</span>
            <span class="thinking-toggle">⌄</span>
        </button>
        <div class="thinking-content">
            <div class="thinking-panel">
                <div class="thinking-panel-header">
                    <span class="thinking-panel-title">处理步骤</span>
                    <span class="thinking-panel-meta">等待更新</span>
                </div>
                <ul class="thinking-logs"></ul>
            </div>
        </div>
    `;
    parentMessageEl.querySelector('.message-content').prepend(container);

    const chip = container.querySelector('.thinking-chip');
    const content = container.querySelector('.thinking-content');
    chip.addEventListener('click', () => {
        const expanded = container.classList.toggle('expanded');
        chip.setAttribute('aria-expanded', String(expanded));
        content.style.maxHeight = expanded ? `${content.scrollHeight}px` : '0px';
    });
    return container;
}

function showThinkingPlaceholder() {
    const pendingMessage = ensurePendingAssistantMessage();
    let container = pendingMessage.querySelector('.thinking-container');
    if (container) return;
    
    container = createThinkingContainer(pendingMessage);
    const summary = container.querySelector('.thinking-summary');
    summary.textContent = '正在整理过程...';
}

function buildThinkingSummary(text) {
    const plain = stripHtml(text).replace(/\s+/g, ' ').trim();
    const combined = plain || '...';
    const maxChars = window.innerWidth <= 768 ? 18 : 32;
    if (combined.length <= maxChars) return combined;
    return `${combined.slice(0, maxChars - 1)}…`;
}

function extractThinkingStep(data, nodeName = '') {
    const analysis = data.analysis || {};
    const title = (analysis.title && typeof analysis.title === 'string')
        ? analysis.title.trim()
        : getThinkingTitle(nodeName);
    const tone = (analysis.tone && typeof analysis.tone === 'string')
        ? analysis.tone
        : inferThinkingTone(nodeName);
    // For router, build a detailed summary string.
    if ((nodeName === 'intent_router' || nodeName === 'router') && analysis.intent && typeof analysis.intent === 'string') {
        const confidencePart = typeof analysis.confidence === 'number'
            ? ` (confidence ${Math.round(analysis.confidence * 100)}%)`
            : '';
        const reason = (analysis.reasoning && typeof analysis.reasoning === 'string')
            ? analysis.reasoning.trim()
            : '';
        const body = reason
            ? `${analysis.intent}${confidencePart}. ${reason}`
            : `${analysis.intent}${confidencePart}`;
        return { title, body, tone: analysis.tone || 'intent' };
    }
    if (analysis.reasoning && typeof analysis.reasoning === 'string') {
        return { title, body: analysis.reasoning.trim(), tone };
    }
    if (typeof analysis === 'string' && analysis.trim()) {
        return { title, body: analysis.trim(), tone };
    }
    const kv = Object.entries(analysis).filter(([, value]) => (
        typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean'
    ));
    if (kv.length) {
        return {
            title,
            body: kv.map(([key, value]) => `${key}: ${String(value)}`).join(' | '),
            tone,
        };
    }
    return { title, body: '...', tone: 'neutral' };
}

function getThinkingTitle(nodeName = '') {
    if (nodeName === 'intent_router' || nodeName === 'router') return 'Intent';
    if (nodeName.includes('tool_call_analyze_food_image')) return 'Analyzing image';
    if (nodeName.includes('tool_result_analyze_food_image')) return 'Recognition result';
    if (nodeName.includes('tool_call_search_restaurants')) return 'Searching places';
    if (nodeName.includes('tool_result_search_restaurants')) return 'Restaurant results';
    if (nodeName === 'supervisor_reply') return 'Writing response';
    if (nodeName === 'chitchat') return 'Drafting reply';
    if (nodeName === 'recognition') return 'Recognition step';
    if (nodeName === 'recommendation') return 'Recommendations';
    return 'Live update';
}

function inferThinkingTone(nodeName = '') {
    if (nodeName.includes('tool_call')) return 'active';
    if (nodeName.includes('tool_result')) return 'success';
    if (nodeName === 'intent_router' || nodeName === 'router') return 'intent';
    if (nodeName === 'supervisor_reply' || nodeName === 'chitchat') return 'compose';
    return 'neutral';
}

function buildThinkingLogMarkup(step) {
    const safeTitle = escapeHtml(step.title || 'Live update');
    const safeBody = escapeHtml(step.body || '...');
    return `
        <div class="thinking-log-rail" aria-hidden="true"></div>
        <div class="thinking-log-main">
            <div class="thinking-log-top">
                <span class="thinking-log-title">${safeTitle}</span>
                <span class="thinking-log-badge thinking-log-badge-${escapeHtml(step.tone || 'neutral')}">Active</span>
            </div>
            <div class="thinking-log-body">${safeBody}</div>
        </div>
    `;
}

function removeTrailingComposeLog(logs) {
    if (!logs) return;
    const composeKeys = ['supervisor_reply', 'chitchat'];
    const items = Array.from(logs.querySelectorAll('li'));
    for (let i = items.length - 1; i >= 0; i -= 1) {
        const key = items[i].dataset.streamItem || '';
        if (composeKeys.includes(key)) {
            items[i].remove();
            return;
        }
        if (items[i].dataset.status === 'active') {
            break;
        }
    }
}

function syncThinkingLogStates(logs, activeKey) {
    const items = Array.from(logs.querySelectorAll('li'));
    items.forEach((li) => {
        const badge = li.querySelector('.thinking-log-badge');
        li.classList.remove('is-active', 'is-complete');
        li.classList.remove('thinking-log-enter');
        if (li.dataset.streamItem === activeKey) {
            li.classList.add('is-active');
            li.dataset.status = 'active';
            if (badge) badge.textContent = '进行中';
        } else {
            li.classList.add('is-complete');
            li.dataset.status = 'complete';
            if (badge) badge.textContent = '完成';
        }
    });
}

function finalizeThinkingContainer(messageEl) {
    const container = messageEl.querySelector('.thinking-container');
    if (!container) return;
    const chip = container.querySelector('.thinking-chip');
    const title = container.querySelector('.thinking-title');
    const stage = container.querySelector('.thinking-stage');
    const panelMeta = container.querySelector('.thinking-panel-meta');
    const content = container.querySelector('.thinking-content');
    const logs = container.querySelector('.thinking-logs');
    removeTrailingComposeLog(logs);
    container.classList.remove('expanded');
    container.classList.add('is-finished');
    chip.setAttribute('aria-expanded', 'false');
    title.textContent = '思考过程';
    stage.textContent = '已完成';
    panelMeta.textContent = `共完成 ${logs.children.length} 步`;
    syncThinkingLogStates(logs, null);
    const summary = container.querySelector('.thinking-summary');
    const lastLogBody = logs.lastElementChild?.querySelector('.thinking-log-body')?.textContent?.trim();
    summary.textContent = lastLogBody || '已完成';
    content.style.maxHeight = '0px';
}

function stripHtml(html) {
    const div = document.createElement('div');
    div.innerHTML = html || '';
    return div.textContent || '';
}

function clearPendingAssistantMessage() {
    if (!state.pendingAssistantMessageEl) return;
    if (state.pendingAssistantMessageEl.isConnected) {
        state.pendingAssistantMessageEl.remove();
    }
    state.pendingAssistantMessageEl = null;
}

function removeTypingIndicator() {
    const el = document.getElementById('typing-indicator');
    if (el) el.remove();
}

function scrollToBottom() {
    requestAnimationFrame(() => {
        dom.messagesScroll.scrollTop = dom.messagesScroll.scrollHeight;
    });
}

// ---------------------------------------------------------------------------
// Simple Markdown Renderer
// ---------------------------------------------------------------------------

function getFoodIcon(name) {
    const n = name.toLowerCase();
    const icons = [
        [/rice|饭|粥/, '\u{1F35A}'],
        [/noodle|面|粉|pasta|spaghetti/, '\u{1F35C}'],
        [/bread|toast|面包|馒头|包子/, '\u{1F35E}'],
        [/burger|汉堡/, '\u{1F354}'],
        [/pizza|披萨/, '\u{1F355}'],
        [/chicken|鸡/, '\u{1F357}'],
        [/meat|肉|beef|pork|牛|猪|羊|steak/, '\u{1F969}'],
        [/fish|鱼|sushi|寿司|海鲜|shrimp|虾/, '\u{1F363}'],
        [/egg|蛋|卵/, '\u{1F95A}'],
        [/salad|沙拉|蔬菜|vegetable|菜/, '\u{1F957}'],
        [/fruit|水果|apple|banana|苹果|香蕉|橙/, '\u{1F34E}'],
        [/soup|汤/, '\u{1F372}'],
        [/cake|蛋糕|dessert|甜点|cookie|饼/, '\u{1F370}'],
        [/drink|饮|coffee|咖啡|tea|茶|juice|奶/, '\u{2615}'],
        [/dumpling|饺|馄饨/, '\u{1F95F}'],
        [/taco|burrito|卷/, '\u{1F32E}'],
        [/fry|炸|薯条/, '\u{1F35F}'],
        [/hot.?dog/, '\u{1F32D}'],
        [/sandwich|三明治/, '\u{1F96A}'],
        [/tofu|豆腐|豆/, '\u{1F962}'],
    ];
    for (const [re, icon] of icons) {
        if (re.test(n)) return icon;
    }
    return '\u{1F37D}\uFE0F'; // plate with cutlery as default
}

// ---------------------------------------------------------------------------
// Nutrition Chart Helpers
// ---------------------------------------------------------------------------

/** Unique ID counter for chart containers */
let _chartIdCounter = 0;

/** Solarized-inspired chart palette */
const CHART_COLORS = [
    '#2aa198', // cyan
    '#268bd2', // blue
    '#b58900', // yellow
    '#cb4b16', // orange
    '#d33682', // magenta
    '#6c71c4', // violet
    '#859900', // green
    '#dc322f', // red
];

const MACRO_COLORS = {
    fat:     '#cb4b16',
    carbs:   '#b58900',
    protein: '#2aa198',
};

/**
 * Determines the current meal type (Breakfast, Lunch, Dinner) based on
 * the browser's local time (respects the user's actual timezone).
 */
function getCurrentMealType(isZh = true) {
    const localHours = new Date().getHours();
    if (localHours >= 5 && localHours < 11) {
        return isZh ? '早餐' : 'Breakfast';
    } else if (localHours >= 11 && localHours < 17) {
        return isZh ? '午餐' : 'Lunch';
    } else if (localHours >= 17 && localHours < 22) {
        return isZh ? '晚餐' : 'Dinner';
    } else {
        return isZh ? '加餐' : 'Snack';
    }
}

/**
 * Get recommended per-meal nutrition based on user profile or defaults.
 * Uses Mifflin-St Jeor if profile has weight/height/age/gender.
 */
function getRecommendedMeal() {
    const prof = getCurrentProfileValues();
    
    // Set defaults based on gender or lack thereof
    let gender = prof.gender || 'unknown';
    let height_cm = prof.height_cm;
    let weight_kg = prof.weight_kg;
    let age = prof.age || 30;

    if (!height_cm) {
        if (gender === 'female') height_cm = 160;
        else if (gender === 'male') height_cm = 170;
        else height_cm = 165;
    }
    if (!weight_kg) {
        if (gender === 'female') weight_kg = 55;
        else if (gender === 'male') weight_kg = 65;
        else weight_kg = 60;
    }
    
    // PAL (Physical Activity Level) - default to 1.2, or 1.5 for high activity
    const pal = (prof.fitness_goals && prof.fitness_goals.toLowerCase().includes('high intensity')) ? 1.5 : 1.2;
    
    // Step 1: Calculate BMR using Mifflin-St Jeor formula
    let bmr;
    if (gender === 'male') {
        bmr = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) + 5;
    } else if (gender === 'female') {
        bmr = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) - 161;
    } else { // 'unknown' gender
        bmr = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) - 78;
    }

    // Step 2: Calculate TDEE
    const tdee = bmr * pal;

    // Step 3: Calculate Meal Targets (1/3 of TDEE)
    const mealCal = tdee / 3;

    return {
        calories: Math.round(mealCal),
        fat:      Math.round((mealCal * 0.25) / 9),  // 25% calories from fat
        carbs:    Math.round((mealCal * 0.55) / 4),   // 55% calories from carbs
        protein:  Math.round((mealCal * 0.20) / 4),   // 20% calories from protein
    };
}

/** Read current profile values from the DOM inputs */
function getCurrentProfileValues() {
    return {
        age:       parseInt(dom.profInputs.age?.value) || 0,
        height_cm: parseFloat(dom.profInputs.height_cm?.value) || 0,
        weight_kg: parseFloat(dom.profInputs.weight_kg?.value) || 0,
        gender:    dom.profInputs.gender?.value || '',
        fitness_goals: dom.profInputs.fitness_goals?.value || '',
    };
}

/** Build an SVG pie chart. Returns SVG string. */
function buildPieChartSVG(slices, size = 140) {
    const cx = size / 2, cy = size / 2, r = size / 2 - 4;
    const total = slices.reduce((s, sl) => s + sl.value, 0);
    if (total === 0) return '';

    // Handle single slice edge case - render as full circle
    if (slices.length === 1) {
        const sl = slices[0];
        return `<svg viewBox="0 0 ${size} ${size}" width="${size}" height="${size}" class="nc-pie-svg"><circle cx="${cx}" cy="${cy}" r="${r}" fill="${sl.color}" opacity="0.85" data-chart-slice="0" style="transition:opacity 0.2s,transform 0.2s;transform-origin:${cx}px ${cy}px;"/><text x="${cx}" y="${cy}" class="nc-pie-label" data-chart-label="0" text-anchor="middle" dominant-baseline="middle">100%</text></svg>`;
    }
    let paths = '';
    let labels = '';
    let startAngle = -Math.PI / 2;

    slices.forEach((sl, i) => {
        const fraction = sl.value / total;
        const angle = fraction * 2 * Math.PI;
        const endAngle = startAngle + angle;
        const largeArc = angle > Math.PI ? 1 : 0;

        const x1 = cx + r * Math.cos(startAngle);
        const y1 = cy + r * Math.sin(startAngle);
        const x2 = cx + r * Math.cos(endAngle);
        const y2 = cy + r * Math.sin(endAngle);

        paths += `<path d="M${cx},${cy} L${x1},${y1} A${r},${r} 0 ${largeArc},1 ${x2},${y2} Z" fill="${sl.color}" data-chart-slice="${i}" opacity="0.85" style="transition:opacity 0.2s,transform 0.2s;transform-origin:${cx}px ${cy}px"/>`;

        // Add percentage label if the slice is large enough
        if (fraction > 0.05) {
            const midAngle = startAngle + angle / 2;
            const textX = cx + (r * 0.7) * Math.cos(midAngle);
            const textY = cy + (r * 0.7) * Math.sin(midAngle);
            const pct = Math.round(fraction * 100);
            labels += `<text x="${textX}" y="${textY}" class="nc-pie-label" data-chart-label="${i}" text-anchor="middle" dominant-baseline="middle">${pct}%</text>`;
        }

        startAngle = endAngle;
    });

    return `<svg viewBox="0 0 ${size} ${size}" width="${size}" height="${size}" class="nc-pie-svg">${paths}${labels}</svg>`;
}

/** Build an SVG horizontal bar chart comparing actual vs recommended. */
function buildBarChartSVG(items, recommended) {
    const barH = 24, gap = 8, labelW = 50, barAreaW = 200, padding = 4;
    const totalH = items.length * (barH + gap) + padding * 2;
    const totalW = labelW + barAreaW + 60;

    // --- Robust Dual-scaling logic ---
    const maxCalVal = items
        .filter(it => it.label.includes('Cal'))
        .reduce((max, it) => Math.max(max, it.actual, it.recommended), 0);

    const maxGramVal = items
        .filter(it => !it.label.includes('Cal'))
        .reduce((max, it) => Math.max(max, it.actual, it.recommended), 0);

    let bars = '';
    items.forEach((it, i) => {
        const y = padding + i * (barH + gap);
        
        const isCalorie = it.label.includes('Cal');
        const maxVal = Math.max(1, isCalorie ? maxCalVal : maxGramVal);

        const actualW = (it.actual / maxVal) * barAreaW;
        const recW = (it.recommended / maxVal) * barAreaW;
        const pct = it.recommended > 0 ? Math.round((it.actual / it.recommended) * 100) : 0;
        const overLimit = it.actual > it.recommended;
        const barColor = overLimit ? 'var(--sol-red)' : it.color;

        // Recommended line
        bars += `<line x1="${labelW + recW}" y1="${y}" x2="${labelW + recW}" y2="${y + barH}" stroke="var(--text-3)" stroke-width="1.5" stroke-dasharray="3,2" opacity="0.6"/>`;
        // Actual bar
        bars += `<rect x="${labelW}" y="${y + 3}" width="${Math.max(actualW, 2)}" height="${barH - 6}" rx="4" fill="${barColor}" opacity="0.75"/>`;
        // Label
        bars += `<text x="${labelW - 6}" y="${y + barH / 2 + 1}" text-anchor="end" fill="var(--text-2)" font-size="11" font-weight="500" dominant-baseline="middle">${it.label}</text>`;
        // Percentage
        bars += `<text x="${labelW + Math.max(actualW, recW) + 6}" y="${y + barH / 2 + 1}" fill="${overLimit ? 'var(--sol-red)' : 'var(--text-2)'}" font-size="10" font-weight="${overLimit ? '700' : '500'}" dominant-baseline="middle">${pct}%</text>`;
    });

    return `<svg viewBox="0 0 ${totalW} ${totalH}" width="100%" height="${totalH}" class="nc-bar-svg" preserveAspectRatio="xMinYMin meet">${bars}</svg>`;
}

/** Parse numeric value from a string like "345.2 kcal" or "12g" */
function parseNutritionVal(str) {
    const m = str.match(/([\d.]+)/);
    return m ? parseFloat(m[1]) : 0;
}

/**
 * Generate a concise, actionable summary based on the meal's nutritional data
 * compared to the recommended values.
 */
// 新增 isZh 参数
function generateNutritionSummary(total, recommended, isZh = true) {
    const THRESHOLDS = { HIGH: 1.35, SLIGHTLY_HIGH: 1.15, LOW: 0.65, SLIGHTLY_LOW: 0.85 };
    const pcts = {
        cal: recommended.calories > 0 ? total.cal / recommended.calories : 0,
        fat: recommended.fat > 0 ? total.fat / recommended.fat : 0,
        carbs: recommended.carbs > 0 ? total.carbs / recommended.carbs : 0,
        protein: recommended.protein > 0 ? total.protein / recommended.protein : 0,
    };

    let summaryParts = [];
    let cal_eval;
    
    // 动态判断能量
    if (pcts.cal > THRESHOLDS.HIGH) cal_eval = isZh ? '能量摄入偏高' : 'Calorie intake is high';
    else if (pcts.cal > THRESHOLDS.SLIGHTLY_HIGH) cal_eval = isZh ? '能量摄入略高' : 'Calorie intake is slightly high';
    else if (pcts.cal < THRESHOLDS.LOW) cal_eval = isZh ? '能量摄入不足' : 'Calorie intake is too low';
    else if (pcts.cal < THRESHOLDS.SLIGHTLY_LOW) cal_eval = isZh ? '能量摄入略低' : 'Calorie intake is slightly low';
    else cal_eval = isZh ? '能量摄入均衡' : 'Calorie intake is balanced';
    summaryParts.push(cal_eval);

    // 动态判断三大营养素
    const macros = [
        { name: isZh ? '脂肪' : 'Fat', pct: pcts.fat },
        { name: isZh ? '碳水' : 'Carbs', pct: pcts.carbs },
        { name: isZh ? '蛋白' : 'Protein', pct: pcts.protein }
    ];
    macros.sort((a, b) => Math.abs(a.pct - 1) - Math.abs(b.pct - 1)).reverse();

    for (let i = 0; i < 2; i++) {
        const macro = macros[i];
        let macro_eval = '';
        if (Math.abs(macro.pct - 1) > (1 - THRESHOLDS.SLIGHTLY_LOW)) {
            if (macro.pct > THRESHOLDS.HIGH) macro_eval = isZh ? `${macro.name}摄入偏高` : `${macro.name} is high`;
            else if (macro.pct > THRESHOLDS.SLIGHTLY_HIGH) macro_eval = isZh ? `${macro.name}摄入略高` : `${macro.name} is slightly high`;
            else if (macro.pct < THRESHOLDS.LOW) macro_eval = isZh ? `${macro.name}摄入不足` : `${macro.name} is too low`;
            else if (macro.pct < THRESHOLDS.SLIGHTLY_LOW) macro_eval = isZh ? `${macro.name}摄入略低` : `${macro.name} is slightly low`;
            
            if (macro_eval) summaryParts.push(macro_eval);
        }
    }
    
    // 动态拼接结论
    const balancedStr = isZh ? '能量摄入均衡' : 'Calorie intake is balanced';
    if (summaryParts.length === 1 && summaryParts[0] === balancedStr) {
        return isZh ? '营养均衡，请继续保持！' : 'Nutrition is balanced, keep it up!';
    } else if (summaryParts.length > 3) {
        summaryParts = summaryParts.slice(0, 3);
    }
    
    if (summaryParts.length > 1 && summaryParts[0] === balancedStr) {
        summaryParts.shift();
    }
    
    return isZh ? `本次用餐建议：${summaryParts.join('，')}。` : `Meal suggestion: ${summaryParts.join(', ')}.`;
}

/**
 * Build the full nutrition visualization (cards + pie + bar).
 * Called from the ```nutrition fenced-block handler in _getMd().
 * Input: array of {name, weight_g, calories, fat_g, carbs_g, protein_g}.
 */
function buildNutritionViz(items) {
    if (!Array.isArray(items) || items.length === 0) return '';

    const chartId = `nchart-${_chartIdCounter++}`;
    const toNum = v => {
        const n = typeof v === 'number' ? v : parseFloat(v);
        return Number.isFinite(n) ? n : 0;
    };
    const fmtG = v => `${toNum(v).toFixed(1)} g`;
    const fmtCal = v => `${Math.round(toNum(v))} kcal`;
    const fmtMass = v => `${Math.round(toNum(v))} g`;

    const namesConcat = items.map(it => String((it && it.name) || '')).join('');
    const isZh = /[\u4e00-\u9fa5]/.test(namesConcat);

    const foods = items
        .filter(it => it && typeof it === 'object')
        .map(it => {
            const mass = toNum(it.weight_g);
            const cal = toNum(it.calories);
            const fat = toNum(it.fat_g);
            const carbs = toNum(it.carbs_g);
            const protein = toNum(it.protein_g);
            return {
                name: String(it.name || '').trim() || (isZh ? '未命名' : 'Item'),
                mass, cal, fat, carbs, protein,
                massStr: fmtMass(mass),
                calStr: fmtCal(cal),
                fatStr: fmtG(fat),
                carbsStr: fmtG(carbs),
                proteinStr: fmtG(protein),
            };
        });

    if (foods.length === 0) return '';

    const total = {
        name: isZh ? '总计' : 'Total',
        cal: 0, fat: 0, carbs: 0, protein: 0, mass: 0,
    };
    foods.forEach(f => {
        total.cal += f.cal;
        total.fat += f.fat;
        total.carbs += f.carbs;
        total.protein += f.protein;
        total.mass += f.mass;
    });
    total.calStr = fmtCal(total.cal);
    total.massStr = fmtMass(total.mass);
    total.fatStr = fmtG(total.fat);
    total.carbsStr = fmtG(total.carbs);
    total.proteinStr = fmtG(total.protein);

    // --- Pie chart: calorie contribution by food item ---
    const calSlices = foods.map((f, i) => ({
        value: f.cal,
        color: CHART_COLORS[i % CHART_COLORS.length],
        label: f.name,
    }));

    const pieSVG = buildPieChartSVG(calSlices, 130);

    // --- Food buttons (pill toggles) ---
    let foodBtnsHtml = foods.map((f, i) => {
        const icon = getFoodIcon(f.name);
        const color = CHART_COLORS[i % CHART_COLORS.length];
        return `<button class="nc-food-btn" data-chart-target="${chartId}" data-food-idx="${i}" style="--food-color:${color}">
            <span class="nc-food-dot" style="background:${color}"></span>
            <span class="nc-food-icon">${icon}</span>
            <span class="nc-food-name">${f.name}</span>
            <span class="nc-food-cal">${Math.round(f.cal)} kcal</span>
        </button>`;
    }).join('');

    // --- Bar chart: actual vs recommended ---
    const rec = getRecommendedMeal();
    const barItems = [
        { label: 'Cal',     actual: total.cal,     recommended: rec.calories, color: 'var(--sol-cyan)' },
        { label: 'Fat',     actual: total.fat,     recommended: rec.fat,      color: MACRO_COLORS.fat },
        { label: 'Carbs',   actual: total.carbs,   recommended: rec.carbs,    color: MACRO_COLORS.carbs },
        { label: 'Protein', actual: total.protein,  recommended: rec.protein,  color: MACRO_COLORS.protein },
    ];

    const barSVG = buildBarChartSVG(barItems, rec);

    // --- Per-food detail cards (compact) ---
    let cardsHtml = '';
    foods.forEach((f, i) => {
        const color = CHART_COLORS[i % CHART_COLORS.length];
        const icon = getFoodIcon(f.name);
        // Mini macro bar
        const macroTotal = f.fat + f.carbs + f.protein;
        const fatPct = macroTotal > 0 ? (f.fat / macroTotal * 100) : 0;
        const carbsPct = macroTotal > 0 ? (f.carbs / macroTotal * 100) : 0;
        const proteinPct = macroTotal > 0 ? (f.protein / macroTotal * 100) : 0;

        cardsHtml += `
        <div class="nutrition-card" data-chart-container="${chartId}" data-food-idx="${i}" style="--card-accent:${color}">
            <div class="nc-header"><span class="nc-icon" style="background:${color}22">${icon}</span>${f.name}</div>
            <div class="nc-macro-bar">
                <div class="nc-macro-seg" style="width:${fatPct}%;background:${MACRO_COLORS.fat}" title="Fat ${f.fatStr}"></div>
                <div class="nc-macro-seg" style="width:${carbsPct}%;background:${MACRO_COLORS.carbs}" title="Carbs ${f.carbsStr}"></div>
                <div class="nc-macro-seg" style="width:${proteinPct}%;background:${MACRO_COLORS.protein}" title="Protein ${f.proteinStr}"></div>
            </div>
            <div class="nc-stats">
                <span class="nc-btn nc-cal"><span class="nc-label">Cal</span> ${f.calStr}</span>
                <span class="nc-btn nc-mass"><span class="nc-label">Wt</span> ${f.massStr}</span>
                <span class="nc-btn nc-macros"><span class="nc-label">F</span> ${f.fatStr} <span class="nc-dot"></span> <span class="nc-label">C</span> ${f.carbsStr} <span class="nc-dot"></span> <span class="nc-label">P</span> ${f.proteinStr}</span>
            </div>
        </div>`;
    });

    // Total card
    cardsHtml += `
    <div class="nutrition-card total-card">
        <div class="nc-header"><span class="nc-icon nc-icon-total">&Sigma;</span>${total.name}</div>
        <div class="nc-stats">
            <span class="nc-btn nc-cal"><span class="nc-label">Cal</span> ${total.calStr || Math.round(total.cal) + ' kcal'}</span>
            <span class="nc-btn nc-mass"><span class="nc-label">Wt</span> ${total.massStr || Math.round(total.mass) + ' g'}</span>
            <span class="nc-btn nc-macros"><span class="nc-label">F</span> ${total.fatStr || total.fat.toFixed(1) + ' g'} <span class="nc-dot"></span> <span class="nc-label">C</span> ${total.carbsStr || total.carbs.toFixed(1) + ' g'} <span class="nc-dot"></span> <span class="nc-label">P</span> ${total.proteinStr || total.protein.toFixed(1) + ' g'}</span>
        </div>
    </div>`;

    // --- Assemble everything ---
    const mealType = getCurrentMealType(isZh);
    const hasProfile = getCurrentProfileValues().weight_kg > 0;
    const guidelineNote = hasProfile
        ? '<span class="nc-guide-note">Based on your profile</span>'
        : '<span class="nc-guide-note">Based on average adult</span>';

    const summaryHtml = generateNutritionSummary(total, rec, isZh); // Generate the summary with the appropriate language setting

    return `
    <div class="nutrition-viz" id="${chartId}" data-total="${JSON.stringify(total).replace(/"/g, '&quot;')}" data-is-zh="${isZh}">
        <div class="nc-charts-row">
            <div class="nc-pie-section">
                <div class="nc-section-title">Calorie Breakdown</div>
                ${pieSVG}
                <div class="nc-food-btns">${foodBtnsHtml}</div>
            </div>
            <div class="nc-bar-section">
                <div class="nc-section-title"><span style="white-space: nowrap;">vs Recommended ${mealType}</span> ${guidelineNote}</div>
                ${barSVG}
                <div class="nc-bar-legend">
                    <span class="nc-legend-item"><span class="nc-legend-bar"></span>Actual</span>
                    <span class="nc-legend-item"><span class="nc-legend-dash"></span>Recommended</span>
                </div>
            </div>
        </div>
        <div class="nutrition-cards-container">${cardsHtml}</div>
        <div class="nc-summary">${summaryHtml}</div>
    </div>`;
}

/** Re-render bar chart + summary on all existing nutrition cards after profile changes. */
function refreshAllNutritionViz() {
    document.querySelectorAll('.nutrition-viz[data-total]').forEach(el => {
        let total;
        try { total = JSON.parse(el.dataset.total); } catch { return; }
        const isZh = el.dataset.isZh === 'true';
        const rec = getRecommendedMeal();

        const barItems = [
            { label: 'Cal',     actual: total.cal,     recommended: rec.calories, color: 'var(--sol-cyan)' },
            { label: 'Fat',     actual: total.fat,     recommended: rec.fat,      color: MACRO_COLORS.fat },
            { label: 'Carbs',   actual: total.carbs,   recommended: rec.carbs,    color: MACRO_COLORS.carbs },
            { label: 'Protein', actual: total.protein,  recommended: rec.protein,  color: MACRO_COLORS.protein },
        ];
        const newBarSVG = buildBarChartSVG(barItems, rec);
        const newSummary = generateNutritionSummary(total, rec, isZh);
        const hasProfile = getCurrentProfileValues().weight_kg > 0;
        const guidelineNote = hasProfile
            ? '<span class="nc-guide-note">Based on your profile</span>'
            : '<span class="nc-guide-note">Based on average adult</span>';
        const mealType = getCurrentMealType(isZh);

        const barSection = el.querySelector('.nc-bar-section');
        if (barSection) {
            barSection.querySelector('.nc-section-title').innerHTML =
                `<span style="white-space: nowrap;">vs Recommended ${mealType}</span> ${guidelineNote}`;
            // Replace bar SVG (first svg inside nc-bar-section)
            const oldSvg = barSection.querySelector('svg');
            if (oldSvg) oldSvg.outerHTML = newBarSVG;
        }
        const summaryEl = el.querySelector('.nc-summary');
        if (summaryEl) summaryEl.innerHTML = newSummary;
    });
}

function buildRestaurantCards(items) {
    if (!Array.isArray(items) || items.length === 0) return '';

    const healthMeta = {
        healthy:   { zh: '健康', en: 'Healthy',   cls: 'rc-health-good',  icon: '🥗' },
        balanced:  { zh: '均衡', en: 'Balanced',  cls: 'rc-health-mid',   icon: '🍽️' },
        indulgent: { zh: '放纵', en: 'Indulgent', cls: 'rc-health-high',  icon: '🍔' },
    };

    const cards = items.map((r) => {
        const name = escapeHtml(String(r.name || '—'));
        const address = escapeHtml(String(r.address || ''));
        const cuisine = escapeHtml(String(r.cuisine || ''));
        const advice = escapeHtml(String(r.advice || ''));
        const ratingNum = typeof r.rating === 'number' ? r.rating : parseFloat(r.rating);
        const rating = Number.isFinite(ratingNum) && ratingNum > 0
            ? `<span class="rc-rating">⭐ ${ratingNum.toFixed(1)}</span>`
            : '';
        const key = String(r.health || 'balanced').toLowerCase();
        const h = healthMeta[key] || healthMeta.balanced;
        const isZh = /[\u4e00-\u9fa5]/.test([r.cuisine, r.advice].join(''));
        const healthLabel = isZh ? h.zh : h.en;
        const cuisinePill = cuisine ? `<span class="rc-pill">${cuisine}</span>` : '';

        return `
            <article class="rc-card">
                <header class="rc-head">
                    <h4 class="rc-name">${name}</h4>
                    ${rating}
                </header>
                ${address ? `<div class="rc-addr">📍 ${address}</div>` : ''}
                <div class="rc-tags">
                    ${cuisinePill}
                    <span class="rc-health ${h.cls}">${h.icon} ${healthLabel}</span>
                </div>
                ${advice ? `<div class="rc-advice">💡 ${advice}</div>` : ''}
            </article>
        `;
    }).join('');

    return `<div class="restaurant-cards">${cards}</div>`;
}

// Singleton markdown-it instance. Must be created lazily so we don't crash
// if the vendor script failed to load; callers fall back to plain escaped text.
let _md = null;
function _getMd() {
    if (_md) return _md;
    if (typeof window.markdownit !== 'function') return null;
    _md = window.markdownit({
        html: false,        // do not parse raw HTML from model/user — XSS-safe default
        linkify: true,      // auto-link bare URLs
        breaks: true,       // treat single newline as <br>, matches old renderer
        typographer: false,
    });

    // Custom fence handlers: map ```restaurants (and, later, ```nutrition)
    // to first-class widgets. Unknown fences fall through to the default.
    const _defaultFence = _md.renderer.rules.fence || ((tokens, idx, opts, env, self) =>
        self.renderToken(tokens, idx, opts));
    _md.renderer.rules.fence = function (tokens, idx, opts, env, self) {
        const token = tokens[idx];
        const info = (token.info || '').trim();
        if (info === 'restaurants') {
            try {
                return buildRestaurantCards(JSON.parse(token.content.trim()));
            } catch (_) { /* fall through to default rendering */ }
        }
        if (info === 'nutrition') {
            try {
                return buildNutritionViz(JSON.parse(token.content.trim()));
            } catch (_) { /* fall through to default rendering */ }
        }
        return _defaultFence(tokens, idx, opts, env, self);
    };

    // Wrap every rendered table with our scroll container and classes so
    // existing .md-table styling carries over.
    _md.renderer.rules.table_open = () => '<div class="table-wrapper"><table class="md-table">';
    _md.renderer.rules.table_close = () => '</table></div>';

    return _md;
}

function renderMarkdown(text) {
    if (!text) return '';
    const md = _getMd();
    if (!md) return escapeHtml(text).replace(/\n/g, '<br/>');
    return md.render(text);
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function formatTime(isoString) {
    if (!isoString) return '';
    try {
        const d = new Date(isoString);
        const now = new Date();
        const isToday = d.toDateString() === now.toDateString();
        if (isToday) {
            return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        }
        return d.toLocaleDateString([], { month: 'short', day: 'numeric' });
    } catch {
        return '';
    }
}

// ---------------------------------------------------------------------------
// Auto-resize textarea
// ---------------------------------------------------------------------------

function autoResize(textarea) {
    textarea.style.height = 'auto';
    textarea.style.height = Math.min(textarea.scrollHeight, 120) + 'px';
}

// ---------------------------------------------------------------------------
// Event Listeners
// ---------------------------------------------------------------------------

// New chat buttons
dom.btnNewChat.addEventListener('click', showNewChatModal);
dom.btnWelcomeNewChat.addEventListener('click', showNewChatModal);

// Modal
dom.modalCancel.addEventListener('click', hideModal);
dom.modalConfirm.addEventListener('click', createUser);
dom.modalNameInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') createUser();
    if (e.key === 'Escape') hideModal();
});
dom.modalOverlay.addEventListener('click', (e) => {
    if (e.target === dom.modalOverlay) hideModal();
});

// Send
dom.btnSend.addEventListener('click', sendMessage);
dom.messageInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});
dom.messageInput.addEventListener('input', () => autoResize(dom.messageInput));

// Image
dom.imageInput.addEventListener('change', handleImageSelect);
dom.btnRemoveImg.addEventListener('click', clearImagePreview);

// Back button (mobile)
dom.btnBack.addEventListener('click', () => {
    dom.sidebar.classList.remove('hidden');
});

// Delete
dom.btnDeleteChat.addEventListener('click', deleteCurrentUser);

// Profile
dom.btnProfile.addEventListener('click', toggleProfilePanel);
dom.btnCloseProfile.addEventListener('click', closeProfilePanel);
dom.drawerBackdrop.addEventListener('click', closeProfilePanel);
dom.btnSaveProfile.addEventListener('click', saveProfile);

// Search
dom.searchInput.addEventListener('input', (e) => {
    renderUserList(e.target.value);
});

// Food button interaction (event delegation on message area)
// 食物按钮交互 + 饼图切片反向交互（双向联动）
document.addEventListener('click', (e) => {
    const btn = e.target.closest('.nc-food-btn');
    if (!btn) return;

    const chartId = btn.dataset.chartTarget;
    const idx = btn.dataset.foodIdx;
    const container = document.getElementById(chartId);
    if (!container) return;

    const isActive = btn.classList.contains('active');

    // 重置所有状态
    container.querySelectorAll('.nc-food-btn').forEach(b => b.classList.remove('active'));
    container.querySelectorAll('.nc-pie-svg path, .nc-pie-svg circle').forEach(p => {
        p.style.opacity = '0.85';
        p.style.transform = '';
    });
    container.querySelectorAll('.nc-pie-svg .nc-pie-label').forEach(l => {
        l.style.opacity = '0';
    });
    container.querySelectorAll('.nutrition-card[data-food-idx]').forEach(c => {
        c.classList.remove('highlighted');
    });

    if (!isActive) {
        // 高亮当前按钮
        btn.classList.add('active');

        // 高亮对应切片（支持 path + circle 单切片）
        container.querySelectorAll('.nc-pie-svg path, .nc-pie-svg circle').forEach(p => {
            if (p.dataset.chartSlice === idx) {
                p.style.opacity = '1';
                p.style.transform = 'scale(1.06)'; // 放大动画
            } else {
                p.style.opacity = '0.3';
            }
        });

        // 显示百分比标签
        const label = container.querySelector(`.nc-pie-label[data-chart-label="${idx}"]`);
        if (label) label.style.opacity = '0.9';

        // 高亮营养卡片
        const card = container.querySelector(`.nutrition-card[data-food-idx="${idx}"]`);
        if (card) card.classList.add('highlighted');
    }
});

// 🔥 新增：饼图切片点击 → 自动选中对应按钮（反向联动）
document.addEventListener('click', (e) => {
    const slice = e.target.closest('.nc-pie-svg path, .nc-pie-svg circle');
    if (!slice) return;
    const chart = slice.closest('.nutrition-viz');
    const index = slice.dataset.chartSlice;
    const targetBtn = chart?.querySelector(`.nc-food-btn[data-food-idx="${index}"]`);
    targetBtn?.click();
});

// ---------------------------------------------------------------------------
// Geolocation Initialization
// ---------------------------------------------------------------------------

// A promise that resolves when geolocation succeeds, fails, or is denied.
// We give it a generous timeout (e.g. 15s) so it doesn't block forever if the user ignores the prompt.
let locationPromise = null;

function requestLocation() {
    if (!('geolocation' in navigator)) {
        console.warn('Geolocation is not supported by this browser.');
        return Promise.resolve();
    }

    return new Promise((resolve) => {
        // Fallback timeout in case the user just leaves the permission prompt open indefinitely
        const timeoutId = setTimeout(() => {
            console.warn('Geolocation prompt timed out (user ignored).');
            resolve();
        }, 15000);

        navigator.geolocation.getCurrentPosition(
            (position) => {
                clearTimeout(timeoutId);
                state.userLocation.lat = position.coords.latitude;
                state.userLocation.lng = position.coords.longitude;
                console.log('User location acquired:', state.userLocation);
                resolve();
            },
            (error) => {
                clearTimeout(timeoutId);
                console.warn('Geolocation failed or denied:', error.message);
                resolve();
            },
            {
                enableHighAccuracy: true,
                timeout: 10000,
                maximumAge: 0 // Force fetching a new location
            }
        );
    });
}

function initGeolocation() {
    // Initial fetch on load
    locationPromise = requestLocation();
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

loadUsers();
initGeolocation();
