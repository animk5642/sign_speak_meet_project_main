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
            const wasMuted = this.localTracks[1].muted;
            this.localTracks[1].setMuted(!wasMuted);
            // wasMuted=true means we just unmuted → enabled=true
            return wasMuted;
        }
        return false;
    }

    toggleAudio() {
        if (this.localTracks[0]) {
            const wasMuted = this.localTracks[0].muted;
            this.localTracks[0].setMuted(!wasMuted);

            // Compute from old state — setMuted is async so .muted is stale
            const nowEnabled = wasMuted; // was muted → now enabled
            console.log(`🎤 Audio: was ${wasMuted ? 'muted' : 'unmuted'} → now ${nowEnabled ? 'unmuted' : 'muted'}`);

            return nowEnabled;
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
        this.signLanguage = null;
        this.word3SignLanguage = null;
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

        // 3. Init Sign Language Manager - Pass the Agora manager to access video track
        this.signLanguage = new SignLanguageManager(this.roomId, this.agora);
        this.signLanguage.initialize();

        // 3b. Init Word3 Sign Language Manager (letter-based A-Z)
        this.word3SignLanguage = new Word3SignLanguageManager(this.roomId, this.agora);
        this.word3SignLanguage.initialize();

        // 4. Init Request Manager if host
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

        // INVERTED UI (User Request):
        // enabled=true (Unmuted/Live) -> Green (active) + Normal Icon
        // enabled=false (Muted/Off) -> Gray (secondary) + Slash Icon
        btn.className = `control-btn ${enabled ? 'active' : 'secondary'}`;
        icon.className = `fas ${enabled ? iconOn : iconOff}`;

        // Force repaint (mobile fix)
        btn.style.display = 'none';
        btn.offsetHeight;
        btn.style.display = '';
    }

    toggleSignLanguage() {
        if (!this.signLanguage) return;

        const btn = document.getElementById('signLanguageBtn');

        if (this.signLanguage.isActive) {
            this.signLanguage.stop();
            btn.className = 'control-btn primary';
            btn.querySelector('i').className = 'fas fa-hands';
            this.showNotification('Sign language detection stopped', 'info');
        } else {
            this.signLanguage.start();
            btn.className = 'control-btn active';
            btn.querySelector('i').className = 'fas fa-hands';
            this.showNotification('Sign language detection started', 'success');
        }
    }

    toggleWord3SignLanguage() {
        if (!this.word3SignLanguage) return;

        const btn = document.getElementById('word3SignBtn');

        if (this.word3SignLanguage.isActive) {
            this.word3SignLanguage.stop();
            btn.className = 'control-btn primary';
            btn.querySelector('i').className = 'fas fa-font';
            this.showNotification('Letter sign language stopped', 'info');
        } else {
            this.word3SignLanguage.start();
            btn.className = 'control-btn active';
            btn.querySelector('i').className = 'fas fa-font';
            this.showNotification('Letter sign language started', 'success');
        }
    }

    async leave() {
        // Cleanup sign language
        if (this.signLanguage) {
            this.signLanguage.disconnect();
        }
        // Cleanup Word3 sign language
        if (this.word3SignLanguage) {
            this.word3SignLanguage.disconnect();
        }

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


// Sign Language Manager
class SignLanguageManager {
    constructor(roomId, agoraManager) {
        this.roomId = roomId;
        this.agoraManager = agoraManager;  // Agora RTC manager to access video track
        this.websocket = null;
        this.isActive = false;
        this.frameInterval = null;
        this.predictions = new Map(); // Store predictions by userId
        this.canvas = document.createElement('canvas');
        this.ctx = this.canvas.getContext('2d');
        this.videoElement = null;  // Will create from Agora track
    }

    initialize() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/sign-language/${this.roomId}/`;

        this.websocket = new WebSocket(wsUrl);

        this.websocket.onopen = () => {
            console.log('Sign language WebSocket connected');
            this.sendStatus();
        };

        this.websocket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleMessage(data);
        };

        this.websocket.onerror = (error) => {
            console.error('Sign language WebSocket error:', error);
        };

        this.websocket.onclose = () => {
            console.log('Sign language WebSocket closed');
        };
    }

    handleMessage(data) {
        if (data.type === 'sign_prediction') {
            // Show overlay for ALL users when ANY user has a prediction
            const overlay = document.getElementById('signLanguageOverlay');
            if (overlay && !overlay.classList.contains('active')) {
                overlay.classList.add('active');
            }

            // Update prediction for this user
            this.predictions.set(data.user_id, {
                username: data.username,
                sign: data.sign,
                confidence: data.confidence,
                timestamp: Date.now()
            });

            this.updateUI();

            // Clear old predictions after 3 seconds (for real-time updates)
            setTimeout(() => {
                this.predictions.delete(data.user_id);
                this.updateUI();

                // Hide overlay if no more predictions
                if (this.predictions.size === 0) {
                    overlay.classList.remove('active');
                }
            }, 3000);
        } else if (data.type === 'error') {
            console.error('Sign language error:', data.message);
        }
    }

    updateUI() {
        const overlay = document.getElementById('signLanguageOverlay');
        const predictionsDiv = document.getElementById('signPredictions');

        if (this.predictions.size === 0) {
            predictionsDiv.innerHTML = '<div class="text-muted text-center py-3"><small>No active signs...</small></div>';
            // Auto-hide overlay when no predictions
            if (overlay && !this.isActive) {
                overlay.classList.remove('active');
            }
        } else {
            const html = Array.from(this.predictions.entries()).map(([userId, pred]) => {
                const confidencePercent = Math.round(pred.confidence * 100);
                const confidenceColor = confidencePercent >= 80 ? '#4caf50' : confidencePercent >= 70 ? '#ff9800' : '#f44336';

                return `
                <div class="sign-prediction">
                    <div class="username">${pred.username}</div>
                    <div class="sign-text">${pred.sign}</div>
                    <div class="confidence" style="color: ${confidenceColor}; font-weight: bold;">
                        ${confidencePercent}% confident
                    </div>
                </div>
            `}).join('');

            predictionsDiv.innerHTML = html;

            // Always show overlay when there are predictions
            if (overlay) {
                overlay.classList.add('active');
            }
        }
    }

    start() {
        if (this.isActive || !this.websocket) return;

        // Check if video track is available and not muted
        const videoTrack = this.agoraManager?.localTracks?.[1];
        if (!videoTrack) {
            console.error('Cannot start sign language detection: No video track');
            return;
        }

        if (videoTrack.muted) {
            console.warn('Video is muted - enabling for sign language detection');
            // Note: We can still try to capture, but frames might be black
        }

        this.isActive = true;

        // Show overlay
        const overlay = document.getElementById('signLanguageOverlay');
        if (overlay) {
            overlay.classList.add('active');
        }

        // Reset video element to get fresh track
        if (this.videoElement) {
            this.videoElement.srcObject = null;
            this.videoElement = null;
        }

        // Start capturing and sending frames - REAL-TIME: 15 FPS for instant detection
        this.frameInterval = setInterval(() => {
            this.captureAndSendFrame();
        }, 66);  // 66ms = ~15 FPS for real-time detection

        console.log('Sign language detection started (15 FPS real-time mode)');
    }

    stop() {
        if (!this.isActive) return;

        this.isActive = false;

        // Stop frame capture
        if (this.frameInterval) {
            clearInterval(this.frameInterval);
            this.frameInterval = null;
        }

        // Clean up video element
        if (this.videoElement) {
            this.videoElement.srcObject = null;
            this.videoElement = null;
        }

        // Don't hide overlay - other users might still be signing
        // Overlay will auto-hide when all predictions clear

        // Send reset message
        if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
            this.websocket.send(JSON.stringify({ type: 'reset' }));
        }

        console.log('Sign language detection stopped');
    }

    captureAndSendFrame() {
        if (!this.websocket || this.websocket.readyState !== WebSocket.OPEN) {
            return;
        }

        try {
            // Get the video track from Agora
            const videoTrack = this.agoraManager?.localTracks?.[1];
            if (!videoTrack) {
                console.warn('No video track available for sign language detection');
                return;
            }

            // Check if video track is muted - still capture but may be black frames
            if (videoTrack.muted) {
                console.log('Video track is muted - sign language detection may not work optimally');
            }

            // Create video element from Agora track if not exists
            if (!this.videoElement) {
                const mediaStreamTrack = videoTrack.getMediaStreamTrack();
                if (!mediaStreamTrack) {
                    console.warn('Could not get MediaStreamTrack from Agora');
                    return;
                }

                // Create a video element to capture frames
                this.videoElement = document.createElement('video');
                this.videoElement.srcObject = new MediaStream([mediaStreamTrack]);
                this.videoElement.autoplay = true;
                this.videoElement.playsInline = true;
                this.videoElement.muted = true;
                this.videoElement.play().catch(err => console.error('Video play error:', err));

                console.log('Created video element for sign language capture');
            }

            // Check if video is ready
            if (this.videoElement.videoWidth === 0 || this.videoElement.videoHeight === 0) {
                console.log('Video element not ready yet, dimensions:', this.videoElement.videoWidth, 'x', this.videoElement.videoHeight);
                return; // Video not ready yet
            }

            // Set canvas size to video size (optimized: use lower resolution)
            const targetWidth = 640;  // Reduced from full video width
            const targetHeight = 480; // Reduced from full video height

            this.canvas.width = targetWidth;
            this.canvas.height = targetHeight;

            // Draw current video frame to canvas (scaled down for faster processing)
            this.ctx.drawImage(this.videoElement, 0, 0, targetWidth, targetHeight);

            // Convert to base64 JPEG with optimized quality (0.6 = 60% quality for speed)
            const frameData = this.canvas.toDataURL('image/jpeg', 0.6);

            // Send to WebSocket
            this.websocket.send(JSON.stringify({
                type: 'video_frame',
                frame: frameData
            }));

        } catch (error) {
            console.error('Error capturing frame:', error);
        }
    }

    sendStatus() {
        if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
            this.websocket.send(JSON.stringify({ type: 'status' }));
        }
    }

    disconnect() {
        this.stop();
        if (this.websocket) {
            this.websocket.close();
            this.websocket = null;
        }
    }
}


// Word3 Sign Language Manager (Letter-based A-Z)
class Word3SignLanguageManager {
    constructor(roomId, agoraManager) {
        this.roomId = roomId;
        this.agoraManager = agoraManager;
        this.websocket = null;
        this.isActive = false;
        this.frameInterval = null;
        this.predictions = new Map(); // userId -> prediction data
        this.canvas = document.createElement('canvas');
        this.ctx = this.canvas.getContext('2d');
        this.videoElement = null;
    }

    initialize() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/word3-sign/${this.roomId}/`;

        this.websocket = new WebSocket(wsUrl);

        this.websocket.onopen = () => {
            console.log('Word3 WebSocket connected');
            this.sendStatus();
        };

        this.websocket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleMessage(data);
        };

        this.websocket.onerror = (error) => {
            console.error('Word3 WebSocket error:', error);
        };

        this.websocket.onclose = () => {
            console.log('Word3 WebSocket closed');
        };
    }

    handleMessage(data) {
        if (data.type === 'word3_prediction') {
            // Update prediction for this user
            this.predictions.set(data.user_id, {
                username: data.username,
                letter: data.letter,
                holdProgress: data.hold_progress,
                currentWord: data.current_word,
                sentence: data.sentence,
                handLandmarks: data.hand_landmarks,
                didBackspace: data.did_backspace || false,
                isSwiping: data.is_swiping || false,
                timestamp: Date.now()
            });

            this.updateVideoOverlays();

            // Clear stale predictions after 1.5s of no updates
            setTimeout(() => {
                const pred = this.predictions.get(data.user_id);
                if (pred && (Date.now() - pred.timestamp) > 1400) {
                    this.predictions.delete(data.user_id);
                    this.updateVideoOverlays();
                }
            }, 1500);

        } else if (data.type === 'error') {
            console.error('Word3 error:', data.message);
        }
    }

    updateVideoOverlays() {
        // Update overlays on each user's video element
        for (const [userId, pred] of this.predictions.entries()) {
            this._updateSingleOverlay(userId, pred);
        }
    }

    _updateSingleOverlay(userId, pred) {
        // Determine the video container — local or remote
        let containerEl = null;
        const localUID = sessionStorage.getItem('UID');

        if (userId === localUID) {
            // Local user — find the local video container
            containerEl = document.getElementById('localVideo')?.closest('.video-item');
        } else {
            containerEl = document.getElementById(`video-${userId}`);
        }

        if (!containerEl) return;

        // ── Letter badge overlay ──
        let badge = containerEl.querySelector('.word3-letter-badge');
        if (!badge) {
            badge = document.createElement('div');
            badge.className = 'word3-letter-badge';
            containerEl.style.position = 'relative';
            containerEl.appendChild(badge);
        }

        if (pred.didBackspace) {
            // Flash backspace indicator
            badge.textContent = '⌫';
            badge.style.display = 'block';
            badge.style.background = '#f44336';
        } else if (pred.isSwiping) {
            badge.textContent = '←';
            badge.style.display = 'block';
            badge.style.background = '#ff5722';
        } else if (pred.letter) {
            badge.textContent = pred.letter;
            badge.style.display = 'block';
            // Colour based on hold progress
            const pct = Math.round(pred.holdProgress * 100);
            if (pct >= 100) {
                badge.style.background = '#34a853';
            } else if (pct >= 50) {
                badge.style.background = '#ff9800';
            } else {
                badge.style.background = 'rgba(102,126,234,0.85)';
            }
        } else {
            badge.style.display = 'none';
        }

        // ── Word + Sentence overlay (bottom of video) ──
        let wordOverlay = containerEl.querySelector('.word3-word-overlay');
        if (!wordOverlay) {
            wordOverlay = document.createElement('div');
            wordOverlay.className = 'word3-word-overlay';
            containerEl.appendChild(wordOverlay);
        }

        const wordText = pred.currentWord || '';
        const sentText = pred.sentence || '';
        const display = sentText ? `${sentText} ${wordText}` : wordText;
        if (display) {
            wordOverlay.textContent = display;
            wordOverlay.style.display = 'block';
        } else {
            wordOverlay.style.display = 'none';
        }

        // ── Hand keypoint canvas overlay ──
        if (pred.handLandmarks && pred.handLandmarks.length > 0) {
            let kpCanvas = containerEl.querySelector('.word3-keypoint-canvas');
            if (!kpCanvas) {
                kpCanvas = document.createElement('canvas');
                kpCanvas.className = 'word3-keypoint-canvas';
                containerEl.appendChild(kpCanvas);
            }

            // Size canvas to match container
            const rect = containerEl.getBoundingClientRect();
            kpCanvas.width = rect.width;
            kpCanvas.height = rect.height;
            kpCanvas.style.display = 'block';

            const kpCtx = kpCanvas.getContext('2d');
            kpCtx.clearRect(0, 0, kpCanvas.width, kpCanvas.height);

            // Draw each hand's landmarks
            for (const hand of pred.handLandmarks) {
                const lm = hand.landmarks;  // Array of [x, y] in original frame coords
                if (!lm || lm.length < 21) continue;

                // Scale landmarks from original frame (640x480) to canvas
                // The backend flips the frame, coordinates are already mirrored
                const scaleX = kpCanvas.width / 640;
                const scaleY = kpCanvas.height / 480;

                const scaled = lm.map(p => [p[0] * scaleX, p[1] * scaleY]);

                // Hand connections from word3.py
                const connections = [
                    [2, 3], [3, 4], [5, 6], [6, 7], [7, 8], [9, 10], [10, 11], [11, 12],
                    [13, 14], [14, 15], [15, 16], [17, 18], [18, 19], [19, 20],
                    [0, 1], [1, 2], [2, 5], [5, 9], [9, 13], [13, 17], [17, 0]
                ];

                const color = hand.label === 'Right' ? '#00BFFF' : '#FF6347';

                // Draw connections
                kpCtx.strokeStyle = color;
                kpCtx.lineWidth = 2;
                for (const [a, b] of connections) {
                    kpCtx.beginPath();
                    kpCtx.moveTo(scaled[a][0], scaled[a][1]);
                    kpCtx.lineTo(scaled[b][0], scaled[b][1]);
                    kpCtx.stroke();
                }

                // Draw landmark points
                for (let i = 0; i < scaled.length; i++) {
                    const r = [4, 8, 12, 16, 20].includes(i) ? 5 : 3;
                    kpCtx.beginPath();
                    kpCtx.arc(scaled[i][0], scaled[i][1], r, 0, 2 * Math.PI);
                    kpCtx.fillStyle = 'white';
                    kpCtx.fill();
                    kpCtx.strokeStyle = 'black';
                    kpCtx.lineWidth = 1;
                    kpCtx.stroke();
                }
            }
        } else {
            // No hands — hide canvas
            const kpCanvas = containerEl.querySelector('.word3-keypoint-canvas');
            if (kpCanvas) kpCanvas.style.display = 'none';
        }
    }

    start() {
        if (this.isActive || !this.websocket) return;

        const videoTrack = this.agoraManager?.localTracks?.[1];
        if (!videoTrack) {
            console.error('Cannot start Word3: No video track');
            return;
        }

        this.isActive = true;

        if (this.videoElement) {
            this.videoElement.srcObject = null;
            this.videoElement = null;
        }

        // Capture frames at ~10 FPS (word3 is single-frame, doesn't need 15)
        this.frameInterval = setInterval(() => {
            this.captureAndSendFrame();
        }, 100);

        console.log('Word3 sign language started (10 FPS)');
    }

    stop() {
        if (!this.isActive) return;
        this.isActive = false;

        if (this.frameInterval) {
            clearInterval(this.frameInterval);
            this.frameInterval = null;
        }

        if (this.videoElement) {
            this.videoElement.srcObject = null;
            this.videoElement = null;
        }

        // Clean up overlays
        document.querySelectorAll('.word3-letter-badge, .word3-word-overlay, .word3-keypoint-canvas').forEach(el => {
            el.style.display = 'none';
        });

        this.predictions.clear();

        if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
            this.websocket.send(JSON.stringify({ type: 'reset' }));
        }

        console.log('Word3 sign language stopped');
    }

    captureAndSendFrame() {
        if (!this.websocket || this.websocket.readyState !== WebSocket.OPEN) return;

        try {
            const videoTrack = this.agoraManager?.localTracks?.[1];
            if (!videoTrack) return;

            if (!this.videoElement) {
                const mediaStreamTrack = videoTrack.getMediaStreamTrack();
                if (!mediaStreamTrack) return;

                this.videoElement = document.createElement('video');
                this.videoElement.srcObject = new MediaStream([mediaStreamTrack]);
                this.videoElement.autoplay = true;
                this.videoElement.playsInline = true;
                this.videoElement.muted = true;
                this.videoElement.play().catch(err => console.error('Word3 video play error:', err));
            }

            if (this.videoElement.videoWidth === 0 || this.videoElement.videoHeight === 0) return;

            this.canvas.width = 640;
            this.canvas.height = 480;
            this.ctx.drawImage(this.videoElement, 0, 0, 640, 480);

            const frameData = this.canvas.toDataURL('image/jpeg', 0.6);
            this.websocket.send(JSON.stringify({
                type: 'video_frame',
                frame: frameData
            }));

        } catch (error) {
            console.error('Word3 capture error:', error);
        }
    }

    sendStatus() {
        if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
            this.websocket.send(JSON.stringify({ type: 'status' }));
        }
    }

    disconnect() {
        this.stop();
        if (this.websocket) {
            this.websocket.close();
            this.websocket = null;
        }
    }
}