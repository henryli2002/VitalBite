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
    history.forEach((msg) => appendMessage(msg.role, msg.content, msg.timestamp));
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
                content: state.pendingImage.base64,
                text: text,
                mime_type: state.pendingImage.mimeType,
                lat: state.userLocation.lat,
                lng: state.userLocation.lng,
            })
        );
        // Show in UI
        const dataUrl = `data:${state.pendingImage.mimeType};base64,${state.pendingImage.base64}`;
        const displayText = text ? `${text}\n\n![image](${dataUrl})` : `![image](${dataUrl})`;
        appendMessage('user', displayText, now);
        clearImagePreview();
    } else {
        // Send text only
        state.ws.send(
            JSON.stringify({
                type: 'message',
                content: text,
                lat: state.userLocation.lat,
                lng: state.userLocation.lng,
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

function appendMessage(role, content, timestamp) {
    const el = document.createElement('div');
    el.className = `message ${role}`;

    const rendered = renderMarkdown(content);
    const timeStr = formatTime(timestamp);

    el.innerHTML = `
        <div class="message-content">${rendered}</div>
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

    const pendingMessage = ensurePendingAssistantMessage();
    let container = pendingMessage.querySelector('.thinking-container');
    const thinkingText = extractThinkingText(data, nodeName);

    if (!container) {
        container = document.createElement('div');
        container.className = 'thinking-container';
        container.innerHTML = `
            <button type="button" class="thinking-chip" aria-expanded="false">
                <span class="thinking-icon">✨</span>
                <span class="thinking-title">Thinking</span>
                <span class="thinking-summary"></span>
                <span class="thinking-toggle">⌄</span>
            </button>
            <div class="thinking-content">
                <ul class="thinking-logs"></ul>
            </div>
        `;
        pendingMessage.querySelector('.message-content').prepend(container);

        const chip = container.querySelector('.thinking-chip');
        const content = container.querySelector('.thinking-content');
        chip.addEventListener('click', () => {
            const expanded = container.classList.toggle('expanded');
            chip.setAttribute('aria-expanded', String(expanded));
            content.style.maxHeight = expanded ? `${content.scrollHeight}px` : '0px';
        });
    }

    const logs = container.querySelector('.thinking-logs');
    let li = logs.querySelector(`li[data-stream-item="${nodeName || 'thinking'}"]`);
    if (!li) {
        li = document.createElement('li');
        li.dataset.streamItem = nodeName || 'thinking';
        logs.appendChild(li);
    }
    if (li.innerHTML !== thinkingText) {
        li.innerHTML = thinkingText;
    }

    const summary = container.querySelector('.thinking-summary');
    summary.textContent = buildThinkingSummary(thinkingText);

    const content = container.querySelector('.thinking-content');
    if (container.classList.contains('expanded')) {
        content.style.maxHeight = `${content.scrollHeight}px`;
    }
}

function showThinkingPlaceholder() {
    const pendingMessage = ensurePendingAssistantMessage();
    let container = pendingMessage.querySelector('.thinking-container');
    if (container) return;

    container = document.createElement('div');
    container.className = 'thinking-container';
    container.innerHTML = `
        <button type="button" class="thinking-chip" aria-expanded="false">
            <span class="thinking-icon">✨</span>
            <span class="thinking-title">Thinking</span>
            <span class="thinking-summary">...</span>
            <span class="thinking-toggle">⌄</span>
        </button>
        <div class="thinking-content">
            <ul class="thinking-logs"></ul>
        </div>
    `;
    pendingMessage.querySelector('.message-content').prepend(container);

    const chip = container.querySelector('.thinking-chip');
    const content = container.querySelector('.thinking-content');
    chip.addEventListener('click', () => {
        const expanded = container.classList.toggle('expanded');
        chip.setAttribute('aria-expanded', String(expanded));
        content.style.maxHeight = expanded ? `${content.scrollHeight}px` : '0px';
    });
}

function buildThinkingSummary(text) {
    const plain = stripHtml(text).replace(/\s+/g, ' ').trim();
    const combined = plain || '...';
    const maxChars = window.innerWidth <= 768 ? 18 : 32;
    if (combined.length <= maxChars) return combined;
    return `${combined.slice(0, maxChars - 1)}…`;
}

function extractThinkingText(data, nodeName = '') {
    const analysis = data.analysis || {};
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
        return formatThinkingStep('Recognizing user intent', body);
    }
    if (analysis.reasoning && typeof analysis.reasoning === 'string') {
        const raw = analysis.reasoning.trim();
        const match = raw.match(/^([^:]+):\s*([\s\S]*)$/);
        if (match) {
            const title = match[1].trim();
            const body = match[2].trim();
            return formatThinkingStep(title, body);
        }
        return formatThinkingStep('Thinking', raw);
    }
    if (typeof analysis === 'string' && analysis.trim()) return escapeHtml(analysis.trim());
    const kv = Object.entries(analysis).filter(([, value]) => (
        typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean'
    ));
    if (kv.length) {
        return escapeHtml(kv.map(([key, value]) => `${key}: ${String(value)}`).join(' | '));
    }
    return '...';
}

function formatThinkingStep(title, body) {
    const safeTitle = escapeHtml(title);
    const safeBody = escapeHtml(body || '');
    return `<span class="thinking-step-title">${safeTitle}:</span><br/><span class="thinking-step-body">${safeBody}</span>`;
}

function finalizeThinkingContainer(messageEl) {
    const container = messageEl.querySelector('.thinking-container');
    if (!container) return;
    const chip = container.querySelector('.thinking-chip');
    const title = container.querySelector('.thinking-title');
    const summary = container.querySelector('.thinking-summary');
    const content = container.querySelector('.thinking-content');
    container.classList.remove('expanded');
    chip.setAttribute('aria-expanded', 'false');
    title.textContent = 'Show thinking';
    summary.textContent = '';
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

function renderMarkdown(text) {
    if (!text) return '';

    let html = escapeHtml(text);

    // Nutrition Table -> Cards (Must do this before line break replacements)
    html = html.replace(/((?:\|.*\|\n?)+)/g, (match) => {
        const lines = match.trim().split('\n');
        if (lines.length < 3) return match;

        const header = lines[0];
        const isNutrition = header.includes('热量') || header.includes('Calories') || header.includes('重量');

        if (isNutrition) {
            let cardsHtml = '<div class="nutrition-cards-container">';
            for (let i = 2; i < lines.length; i++) {
                const cols = lines[i].split('|').slice(1, -1).map(c => c.trim());
                if (cols.length >= 6) {
                    const isTotal = cols[0].includes('总计') || cols[0].includes('Total') || cols[0].includes('**');
                    const cardClass = isTotal ? 'nutrition-card total-card' : 'nutrition-card';
                    const clean = str => str.replace(/\*\*/g, '').trim();
                    
                    cardsHtml += `
                    <div class="${cardClass}">
                        <div class="nc-header">${clean(cols[0])}</div>
                        <div class="nc-stats">
                            <span class="nc-btn nc-cal">🔥 ${clean(cols[2])}</span>
                            <span class="nc-btn nc-mass">⚖️ ${clean(cols[1])}</span>
                            <span class="nc-btn nc-macros">🥑 脂 ${clean(cols[3])} • 🍞 碳 ${clean(cols[4])} • 🥩 蛋 ${clean(cols[5])}</span>
                        </div>
                    </div>`;
                }
            }
            cardsHtml += '</div>';
            return cardsHtml;
        } else {
            // Generic table
            let tableHtml = '<div class="table-wrapper"><table class="md-table">';
            lines.forEach((line, index) => {
                if (index === 1) return; // skip separator
                const cols = line.split('|').slice(1, -1).map(c => c.trim());
                tableHtml += '<tr>';
                cols.forEach(col => {
                    const clean = col.replace(/\*\*/g, '');
                    if (index === 0) tableHtml += `<th>${clean}</th>`;
                    else tableHtml += `<td>${clean}</td>`;
                });
                tableHtml += '</tr>';
            });
            tableHtml += '</table></div>';
            return tableHtml;
        }
    });

    // Images
    html = html.replace(/!\[([^\]]*)\]\(([^)]+)\)/g, '<img src="$2" alt="$1" style="max-width: 300px; max-height: 300px; width: auto; height: auto; object-fit: contain; border-radius: 8px; margin-top: 8px;"/>');

    // Headers
    html = html.replace(/^#### (.+)$/gm, '<h4>$1</h4>');
    html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');

    // Bold
    html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');

    // Italic
    html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');

    // Code blocks
    html = html.replace(/```(\w*)\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>');

    // Inline code
    html = html.replace(/`(.+?)`/g, '<code>$1</code>');

    // Unordered lists
    html = html.replace(/^- (.+)$/gm, '<li>$1</li>');
    html = html.replace(/(<li>.*<\/li>)/gs, '<ul>$1</ul>');
    // Clean up nested <ul> tags
    html = html.replace(/<\/ul>\s*<ul>/g, '');

    // Line breaks (ignore line breaks inside our newly generated div blocks)
    // Actually, simple replace will convert \n to <br/>. We need to be careful with the generated HTML.
    // So let's temporarily swap out our divs.
    return html.replace(/\n\n/g, '<br/><br/>').replace(/\n/g, '<br/>').replace(/<br\/>\s*(<div class="nutrition|<div class="table-wrapper|<\/div>|<table|<\/table>|<tr|<\/tr>|<td|<\/td>|<th|<\/th>)/g, '$1');
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
