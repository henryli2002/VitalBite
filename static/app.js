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
            showTypingIndicator();
        } else if (data.type === 'message') {
            removeTypingIndicator();
            appendMessage(data.role, data.content, data.timestamp);
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

    // Disable input while waiting
    dom.messageInput.disabled = true;
    dom.btnSend.disabled = true;
    const oldPlaceholder = dom.messageInput.placeholder;
    dom.messageInput.placeholder = "Waiting for location permission...";

    // Wait for the initial location prompt to resolve (allow, deny, or timeout)
    if (locationPromise) {
        await locationPromise;
    }

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

    // Line breaks
    html = html.replace(/\n\n/g, '<br/><br/>');
    html = html.replace(/\n/g, '<br/>');

    return html;
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

function initGeolocation() {
    if (!('geolocation' in navigator)) {
        console.warn('Geolocation is not supported by this browser.');
        locationPromise = Promise.resolve();
        return;
    }

    locationPromise = new Promise((resolve) => {
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
                maximumAge: 0
            }
        );
    });
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

loadUsers();
initGeolocation();
