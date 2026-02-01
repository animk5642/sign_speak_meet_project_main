// Utility Functions
class SignSpeakUtils {
    static generateRoomId() {
        return Math.random().toString(36).substring(2, 10).toUpperCase();
    }

    static formatTime(seconds) {
        const hrs = Math.floor(seconds / 3600);
        const mins = Math.floor((seconds % 3600) / 60);
        const secs = seconds % 60;

        if (hrs > 0) {
            return `${hrs}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
        }
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    }

    static getCookie(name) {
        let cookieValue = null;
        if (document.cookie && document.cookie !== '') {
            const cookies = document.cookie.split(';');
            for (let i = 0; i < cookies.length; i++) {
                const cookie = cookies[i].trim();
                if (cookie.substring(0, name.length + 1) === (name + '=')) {
                    cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                    break;
                }
            }
        }
        return cookieValue;
    }
}

// Agora-based Manager (Replaces WebRTCManager)
class AgoraRTCManager {
    constructor(appId, roomId, localVideoElement, onRemoteStream, onRemoteLeave) {
        this.appId = appId;
        this.roomId = roomId;
        this.localVideoElement = localVideoElement;
        this.onRemoteStream = onRemoteStream;
        this.onRemoteLeave = onRemoteLeave;

        this.client = AgoraRTC.createClient({
            mode: 'rtc',
            codec: 'vp8',
            logUpload: false  // Disable stats collection to prevent ad blocker errors
        });
        this.localTracks = [];
        this.remoteUsers = {};
        this.UID = sessionStorage.getItem('UID');
        this.TOKEN = null;
    }

    async initialize() {
        try {
            // Fetch token
            const response = await fetch(`/video/get_token/?channel=${this.roomId}&uid=${this.UID}`);
            const data = await response.json();
            this.TOKEN = data.token;

            // Join channel
            this.UID = await this.client.join(this.appId, this.roomId, this.TOKEN, this.UID);

            // Set up event listeners
            this.client.on('user-published', (user, mediaType) => this.handleUserPublished(user, mediaType));
            this.client.on('user-left', (user) => this.handleUserLeft(user));

            // Create local tracks with echo cancellation and noise suppression
            this.localTracks = await AgoraRTC.createMicrophoneAndCameraTracks(
                {
                    // Audio track configuration
                    AEC: true,  // Acoustic Echo Cancellation
                    ANS: true,  // Automatic Noise Suppression
                    AGC: true   // Auto Gain Control
                },
                {
                    // Video track configuration
                    encoderConfig: "480p_1"
                }
            );

            // Play local video ONLY (NOT audio - to prevent hearing yourself)
            this.localTracks[1].play(this.localVideoElement.id);

            // IMPORTANT: Do NOT play local audio track
            // this.localTracks[0].play() would cause you to hear yourself
            // Local audio is only sent to remote users via publish()

            // Publish tracks to remote users
            await this.client.publish([this.localTracks[0], this.localTracks[1]]);

            // START MUTED BY DEFAULT - To match the initial Gray/Unmarked appearance
            this.localTracks[0].setMuted(true);
            this.localTracks[1].setMuted(true);

            return true;
        } catch (error) {
            console.error('Error initializing Agora:', error);
            return false;
        }
    }

    async handleUserPublished(user, mediaType) {
        await this.client.subscribe(user, mediaType);
        this.remoteUsers[user.uid] = user;

        if (mediaType === 'video' || mediaType === 'audio') {
            this.onRemoteStream(user.uid, user);
        }
    }

    handleUserLeft(user) {
        delete this.remoteUsers[user.uid];
        this.onRemoteLeave(user.uid);
    }

    toggleVideo() {
        if (this.localTracks[1]) {
            const muted = this.localTracks[1].muted;
            this.localTracks[1].setMuted(!muted);
            return !this.localTracks[1].muted;
        }
        return false;
    }

    toggleAudio() {
        if (this.localTracks[0]) {
            const wasMuted = this.localTracks[0].muted;
            this.localTracks[0].setMuted(!wasMuted);
            const nowMuted = this.localTracks[0].muted;

            console.log(`🎤 Audio: was ${wasMuted ? 'muted' : 'unmuted'} → now ${nowMuted ? 'muted' : 'unmuted'}`);

            // Return true if enabled (not muted), false if disabled (muted)
            return !nowMuted;
        }
        return false;
    }

    async stop() {
        if (this.localTracks) {
            this.localTracks.forEach(track => {
                track.stop();
                track.close();
            });
        }
        await this.client.leave();
    }
}

// Chat Manager
class ChatManager {
    constructor(roomId, chatContainer, userEmail) {
        this.roomId = roomId;
        this.chatContainer = chatContainer;
        this.userEmail = userEmail;
        this.socket = null;
    }

    initialize() {
        const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
        this.socket = new WebSocket(`${protocol}://${window.location.host}/ws/chat/${this.roomId}/`);

        this.socket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.displayMessage(data);
        };
    }

    sendMessage(message) {
        if (this.socket && this.socket.readyState === WebSocket.OPEN && message.trim()) {
            this.socket.send(JSON.stringify({
                message: message,
                message_type: 'text'
            }));
        }
    }

    displayMessage(data) {
        const isOwn = data.username === this.userEmail;
        const messageElement = document.createElement('div');
        messageElement.className = `chat-message ${isOwn ? 'own' : 'other'} fade-in`;

        const timestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

        // Extract name from email (part before @)
        const displayName = isOwn ? 'You' : (data.username ? data.username.split('@')[0] : 'Participant');

        if (data.message_type === 'system') {
            messageElement.className = 'chat-message system text-center w-100 my-2';
            messageElement.innerHTML = `<small class="text-muted italic">${data.message}</small>`;
        } else {
            messageElement.innerHTML = `
                <div class="fw-bold small" style="opacity: 0.8; margin-bottom: 4px;">${displayName}</div>
                <div>${data.message}</div>
                <small class="opacity-50 d-block text-end" style="font-size: 0.7rem; margin-top: 4px;">${timestamp}</small>
            `;
        }

        this.chatContainer.appendChild(messageElement);
        this.chatContainer.scrollTop = this.chatContainer.scrollHeight;
    }
}


// Join Request Manager
class JoinRequestManager {
    constructor(roomId, onNewRequest) {
        this.roomId = roomId;
        this.onNewRequest = onNewRequest;
        this.pollInterval = null;
    }

    startPolling() {
        this.checkRequests();
        this.pollInterval = setInterval(() => this.checkRequests(), 10000); // Every 10 seconds
    }

    stopPolling() {
        if (this.pollInterval) clearInterval(this.pollInterval);
    }

    async checkRequests() {
        try {
            const response = await fetch(`/video/get-pending-requests/${this.roomId}/`);
            const data = await response.json();
            if (data.success) {
                this.updateUI(data.requests);
                if (data.new_requests_count > 0 && this.onNewRequest) {
                    this.onNewRequest(data.requests[0]);
                }
            }
        } catch (error) {
            console.error('Error checking join requests:', error);
        }
    }

    updateUI(requests) {
        const countEl = document.getElementById('pendingCount');
        const listEl = document.getElementById('pendingRequestsList');
        if (!countEl || !listEl) return;

        countEl.textContent = requests.length;

        if (requests.length === 0) {
            listEl.innerHTML = '<div class="text-muted small italic px-3">No pending requests</div>';
            return;
        }

        listEl.innerHTML = requests.map(req => `
            <div class="list-group-item bg-transparent text-white border-secondary px-0 mb-2">
                <div class="d-flex justify-content-between align-items-center">
                    <div class="small">
                        <div class="fw-bold text-truncate" style="max-width: 150px;">${req.user_email}</div>
                        <div class="text-muted" style="font-size: 0.7rem;">requested to join</div>
                    </div>
                    <div class="d-flex gap-1">
                        <button onclick="app.requests.handle('${req.id}', 'approve')" class="btn btn-success btn-sm p-1" title="Approve"><i class="fas fa-check"></i></button>
                        <button onclick="app.requests.handle('${req.id}', 'reject')" class="btn btn-outline-danger btn-sm p-1" title="Reject"><i class="fas fa-times"></i></button>
                    </div>
                </div>
            </div>
        `).join('');
    }

    async handle(requestId, action) {
        const formData = new FormData();
        formData.append('action', action);
        formData.append('csrfmiddlewaretoken', SignSpeakUtils.getCookie('csrftoken'));

        try {
            await fetch(`/video/handle-request/${requestId}/`, {
                method: 'POST',
                body: formData,
                headers: {
                    'X-CSRFToken': SignSpeakUtils.getCookie('csrftoken'),
                }
            });
            this.checkRequests();
        } catch (error) {
            console.error(`Error ${action}ing request:`, error);
        }
    }
}

// Main Application
class SignSpeakMeetApp {
    constructor(appId, roomId, userEmail) {
        this.appId = appId;
        this.roomId = roomId;
        this.userEmail = userEmail;
        this.agora = null;
        this.chat = null;
        this.requests = null;
        this.timerInterval = null;
        this.seconds = 0;
        this.participantNames = {};  // Store participant names by UID
    }

    async init() {
        // 1. Init Agora
        const localVideo = document.getElementById('localVideo');
        this.agora = new AgoraRTCManager(
            this.appId,
            this.roomId,
            localVideo,
            (uid, user) => this.addRemoteStream(uid, user),
            (uid) => this.removeRemoteStream(uid)
        );

        const success = await this.agora.initialize();
        if (!success) {
            this.showNotification('Could not initialize Agora', 'danger');
        }

        // Initialize button states to match actual track states
        this.syncButtonStates();

        // 2. Init Chat
        const chatContainer = document.getElementById('chatMessages');
        this.chat = new ChatManager(this.roomId, chatContainer, this.userEmail);
        this.chat.initialize();

        // 3. Init Request Manager if host
        if (document.getElementById('pendingRequestsSection')) {
            this.requests = new JoinRequestManager(this.roomId, (req) => {
                this.showNotification(`New join request: ${req.user_email}`, 'warning');
            });
            this.requests.startPolling();
        }

        this.startTimer();

        // Notify backend about member creation
        await this.createMember();
    }

    async addRemoteStream(userId, user) {
        // Fetch participant name from backend
        let participantName = 'Participant';
        try {
            const memberData = await this.getMember(userId);
            if (memberData && memberData.name) {
                participantName = memberData.name;
                this.participantNames[userId] = participantName;
            }
        } catch (error) {
            console.log('Could not fetch participant name:', error);
        }

        let videoItem = document.getElementById(`video-${userId}`);
        if (!videoItem) {
            videoItem = document.createElement('div');
            videoItem.id = `video-${userId}`;
            videoItem.className = 'video-item';
            videoItem.innerHTML = `
                <div id="player-${userId}" class="video-element"></div>
                <div class="video-overlay"><i class="fas fa-user me-1"></i>${participantName}</div>
            `;
            document.getElementById('videoGrid').appendChild(videoItem);

            // Add to sidebar list (Privacy mode)
            this.updateParticipantList(userId, 'add', participantName);
        }

        // Play video track
        if (user.videoTrack) {
            user.videoTrack.play(`player-${userId}`);
        }

        // Play audio track with volume control
        if (user.audioTrack) {
            try {
                // Set volume to 100%
                user.audioTrack.setVolume(100);

                // Play audio - browser will handle autoplay restrictions
                user.audioTrack.play();

                console.log(`✅ Playing audio from user ${userId}`);
            } catch (error) {
                console.error('Error playing audio track:', error);
            }
        }
    }

    removeRemoteStream(userId) {
        const videoItem = document.getElementById(`video-${userId}`);
        if (videoItem) videoItem.remove();
        this.updateParticipantList(userId, 'remove');
    }

    updateParticipantList(userId, action, participantName = null) {
        const listEl = document.getElementById('participantsList');
        if (!listEl) return;

        if (action === 'add') {
            if (document.getElementById(`participant-${userId}`)) return;

            const item = document.createElement('div');
            item.id = `participant-${userId}`;
            item.className = 'list-group-item bg-transparent text-white border-secondary px-0 fade-in';
            item.innerHTML = `
                <div class="d-flex align-items-center py-2">
                    <div class="avatar me-2 bg-secondary rounded-circle d-flex align-items-center justify-content-center" 
                         style="width: 32px; height: 32px; font-weight: bold;">P</div>
                    <div>
                        <div class="fw-bold small">${participantName || this.participantNames[userId] || 'Participant'}</div>
                        <div class="text-muted" style="font-size: 0.7rem;">Member</div>
                    </div>
                </div>
            `;
            listEl.appendChild(item);
        } else {
            const item = document.getElementById(`participant-${userId}`);
            if (item) item.remove();
        }
    }

    startTimer() {
        this.timerInterval = setInterval(() => {
            this.seconds++;
            const timerEl = document.getElementById('meetingTimer');
            if (timerEl) timerEl.textContent = SignSpeakUtils.formatTime(this.seconds);
        }, 1000);
    }

    showNotification(msg, type) {
        const toastContainer = document.getElementById('toastContainer');
        if (toastContainer) {
            const toast = document.createElement('div');
            toast.className = `toast align-items-center text-white bg-${type} border-0 show mb-2`;
            toast.innerHTML = `<div class="d-flex"><div class="toast-body">${msg}</div><button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button></div>`;
            toastContainer.appendChild(toast);
            setTimeout(() => toast.remove(), 4000);
        }
    }


    toggleVideo() {
        const enabled = this.agora.toggleVideo();
        // Unmuted (enabled=true) -> fa-video, Muted (enabled=false) -> fa-video-slash
        this.updateBtn('videoBtn', enabled, 'fa-video', 'fa-video-slash');
    }

    toggleAudio() {
        const enabled = this.agora.toggleAudio();
        // Unmuted (enabled=true) -> fa-microphone, Muted (enabled=false) -> fa-microphone-slash
        this.updateBtn('audioBtn', enabled, 'fa-microphone', 'fa-microphone-slash');
    }

    syncButtonStates() {
        // Sync audio button - Muted = Red/Slash, Unmuted = Gray/Normal
        if (this.agora.localTracks[0]) {
            const audioEnabled = !this.agora.localTracks[0].muted;
            this.updateBtn('audioBtn', audioEnabled, 'fa-microphone', 'fa-microphone-slash');
        }

        // Sync video button - Muted = Red/Slash, Unmuted = Gray/Normal
        if (this.agora.localTracks[1]) {
            const videoEnabled = !this.agora.localTracks[1].muted;
            this.updateBtn('videoBtn', videoEnabled, 'fa-video', 'fa-video-slash');
        }
    }

    updateBtn(id, enabled, iconOn, iconOff) {
        const btn = document.getElementById(id);
        const icon = btn.querySelector('i');

        // STANDARD LOGIC (User Request):
        // enabled=true (Unmuted/Live) -> Gray (secondary) + Normal Icon
        // enabled=false (Muted/Standby) -> Red (danger) + Slash Icon
        btn.className = `control-btn ${enabled ? 'secondary' : 'danger'}`;
        icon.className = `fas ${enabled ? iconOn : iconOff}`;

        // Force repaint (mobile fix)
        btn.style.display = 'none';
        btn.offsetHeight;
        btn.style.display = '';
    }

    async leave() {
        await this.agora.stop();
        await this.deleteMember();
        window.location.href = '/';
    }

    async createMember() {
        const UID = sessionStorage.getItem('UID');
        const NAME = sessionStorage.getItem('name');
        await fetch('/video/create_member/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': SignSpeakUtils.getCookie('csrftoken')
            },
            body: JSON.stringify({ 'name': NAME, 'room_name': this.roomId, 'UID': UID })
        });
    }

    async getMember(uid) {
        let response = await fetch(`/video/get_member/?UID=${uid}&room_name=${this.roomId}`)
        let member = await response.json()
        return member
    }

    async deleteMember() {
        const UID = sessionStorage.getItem('UID');
        const NAME = sessionStorage.getItem('name');
        await fetch('/video/delete_member/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': SignSpeakUtils.getCookie('csrftoken')
            },
            body: JSON.stringify({ 'name': NAME, 'room_name': this.roomId, 'UID': UID })
        });
    }
}

// Initialize on page load if needed
window.SignSpeakApp = SignSpeakMeetApp;
window.addEventListener("beforeunload", () => {
    if (window.app) window.app.deleteMember();
});