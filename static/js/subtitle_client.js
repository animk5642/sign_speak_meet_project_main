/**
 * subtitle_client.js - Real-time multilingual subtitle client
 *
 * Captures microphone audio in the browser, sends WAV chunks to HPC server
 * for Whisper STT + NLLB translation, and broadcasts results via WebSocket.
 *
 * Based on the tested Python GUI (transcribe_gui_nllb.py) logic.
 */

// ─── Configuration ──────────────────────────────────────────────

const SUBTITLE_CONFIG = {
  SAMPLE_RATE: 16000,
  CHUNK_DURATION_INDIC: 8,   // seconds for Indic languages
  CHUNK_DURATION_EN: 5,      // seconds for English

  // Auto-silence detection
  SILENCE_THRESHOLD_RMS: 0.03,
  SILENCE_GAP_MS: 400,
  AUTO_MIN_CHUNK_S: 1.5,
  AUTO_MAX_CHUNK_S: 15.0,

  // Language codes that need longer chunks
  INDIC_CODES: new Set(['ml', 'ml-en', 'ml-roman', 'ml-via-en', 'hi', 'hi-en', 'ta', 'te', 'kn']),

  // STT language options: display name → whisper mode
  STT_LANGUAGES: {
    'Off (No Subtitles)': '',
    'English': 'en',
    'Hindi (हिन्दी)': 'hi',
    'Hindi + English Mix': 'hi-en',
    'Malayalam (മലയാളം)': 'ml',
    'Malayalam → Manglish': 'ml-roman',
    'Malayalam + English Mix': 'ml-en',
    'Tamil (தமிழ்)': 'ta',
    'Telugu (తెలుగు)': 'te',
    'Kannada (ಕನ್ನಡ)': 'kn',
    'Auto-detect': 'auto',
  },

  // NLLB target languages: display name → NLLB code
  TARGET_LANGUAGES: {
    'None': '',
    'English': 'eng_Latn',
    'Hindi (हिन्दी)': 'hin_Deva',
    'Malayalam (മലയാളം)': 'mal_Mlym',
    'Tamil (தமிழ்)': 'tam_Taml',
    'Telugu (తెలుగు)': 'tel_Telu',
    'Kannada (ಕನ್ನಡ)': 'kan_Knda',
    'Bengali (বাংলা)': 'ben_Beng',
    'Marathi (मराठी)': 'mar_Deva',
    'Gujarati (ગુજરાતી)': 'guj_Gujr',
    'Punjabi (ਪੰਜਾਬੀ)': 'pan_Guru',
    'Urdu (اردو)': 'urd_Arab',
    'Arabic (عربي)': 'arb_Arab',
    'French': 'fra_Latn',
    'German': 'deu_Latn',
    'Spanish': 'spa_Latn',
    'Portuguese': 'por_Latn',
    'Russian': 'rus_Cyrl',
    'Chinese (简体)': 'zho_Hans',
    'Japanese (日本語)': 'jpn_Jpan',
    'Korean (한국어)': 'kor_Hang',
  },

  // Whisper language code → NLLB code mapping
  // Used to determine source_nllb for translation routing
  WHISPER_TO_NLLB: {
    'en': 'eng_Latn',
    'hi': 'hin_Deva',
    'hi-en': 'hin_Deva',
    'ml': 'mal_Mlym',
    'ml-en': 'mal_Mlym',
    'ml-roman': 'mal_Mlym',
    'ml-via-en': 'eng_Latn',   // via-en mode outputs English
    'ta': 'tam_Taml',
    'te': 'tel_Telu',
    'kn': 'kan_Knda',
    // Auto-detect uses HPC detected_language to resolve
  },

  // HPC detected_language → NLLB code mapping
  // When mode is 'auto', HPC returns detected_language like 'en', 'hi', etc.
  DETECTED_TO_NLLB: {
    'en': 'eng_Latn',
    'hi': 'hin_Deva',
    'ml': 'mal_Mlym',
    'ta': 'tam_Taml',
    'te': 'tel_Telu',
    'kn': 'kan_Knda',
    'bn': 'ben_Beng',
    'mr': 'mar_Deva',
    'gu': 'guj_Gujr',
    'pa': 'pan_Guru',
    'ur': 'urd_Arab',
    'ar': 'arb_Arab',
    'fr': 'fra_Latn',
    'de': 'deu_Latn',
    'es': 'spa_Latn',
    'pt': 'por_Latn',
    'ru': 'rus_Cyrl',
    'zh': 'zho_Hans',
    'ja': 'jpn_Jpan',
    'ko': 'kor_Hang',
  },
};

// ─── SubtitleClient Class ────────────────────────────────────────

class SubtitleClient {
  constructor(roomId, hpcServerUrl, userId, username) {
    this.roomId = roomId;
    this.hpcUrl = hpcServerUrl;              // kept for reference
    this.proxyBase = '/video/proxy';          // Django proxy base URL
    this.userId = String(userId);            // ensure string for comparison
    this.username = username;

    // Language prefs
    this.speakLang = '';        // whisper mode (empty = off)
    this.subtitleLang = '';     // NLLB target code

    // Chunk mode: 'default' | 'custom' | 'auto'
    this.chunkMode = 'default';
    this.customChunkDuration = 8;

    // Audio state
    this.isRecording = false;
    this.audioChunks = [];
    this.audioContext = null;
    this.mediaStream = null;
    this.processorNode = null;
    this.sourceNode = null;
    this.isSending = false;
    this.actualSampleRate = SUBTITLE_CONFIG.SAMPLE_RATE;

    // Auto-silence state
    this.silenceStart = null;
    this.hasSpeech = false;
    this.chunkStartTime = null;

    // WebSocket
    this.ws = null;
    this.wsReconnectTimer = null;

    // HPC status
    this.hpcAvailable = false;

    // UI callbacks
    this.onCaption = null;        // (speakerName, text, type) => {}
    this.onStatus = null;         // (message) => {}
    this.onAudioLevel = null;     // (rms) => {}
    this.onProgress = null;       // (percent) => {}
  }

  // ─── Initialization ─────────────────────────────────────────

  /**
   * Get CSRF token from cookies (required for Django POST requests).
   */
  getCsrfToken() {
    const match = document.cookie.match(/csrftoken=([^;]+)/);
    return match ? match[1] : '';
  }

  /**
   * Check if HPC server is reachable. Called once on init.
   */
  async checkHpcHealth() {
    try {
      const resp = await fetch(`${this.proxyBase}/health/`, {
        method: 'GET',
        credentials: 'same-origin',
      });
      if (resp.ok) {
        const data = await resp.json();
        this.hpcAvailable = data.status === 'healthy';
        console.log('[Subtitle] HPC health:', data);
        if (this.onStatus) {
          this.onStatus(this.hpcAvailable ? 'HPC connected' : 'HPC unhealthy');
        }
      }
    } catch (err) {
      this.hpcAvailable = false;
      console.warn('[Subtitle] HPC unreachable:', err.message);
      if (this.onStatus) this.onStatus('HPC server unreachable');
    }
  }

  // ─── WebSocket ──────────────────────────────────────────────

  connectWebSocket() {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) return;

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/subtitle/${this.roomId}/`;

    this.ws = new WebSocket(wsUrl);

    this.ws.onopen = () => {
      console.log('[Subtitle] WebSocket connected');
      this.sendLanguagePrefs();
      // Check HPC health on WS connect
      this.checkHpcHealth();
    };

    this.ws.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data);
        this.handleWsMessage(data);
      } catch (err) {
        console.error('[Subtitle] WS message parse error:', err);
      }
    };

    this.ws.onclose = () => {
      console.log('[Subtitle] WebSocket closed, reconnecting in 3s...');
      this.wsReconnectTimer = setTimeout(() => this.connectWebSocket(), 3000);
    };

    this.ws.onerror = (err) => {
      console.error('[Subtitle] WebSocket error:', err);
    };
  }

  disconnectWebSocket() {
    clearTimeout(this.wsReconnectTimer);
    if (this.ws) {
      this.ws.onclose = null;  // prevent reconnect on intentional close
      this.ws.close();
      this.ws = null;
    }
  }

  sendLanguagePrefs() {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({
        type: 'set_language',
        speak_lang: this.speakLang,
        subtitle_lang: this.subtitleLang,
      }));
    }
  }

  handleWsMessage(data) {
    if (data.type === 'subtitle') {
      const isSelf = String(data.speaker_id) === this.userId;

      // Skip self — we already show own caption locally
      if (isSelf) return;

      if (!this.onCaption) return;

      const sourceNllb = data.source_nllb || '';
      const myLang = this.subtitleLang;

      if (!myLang || !sourceNllb || sourceNllb === myLang) {
        // Same language or no translation configured — show original
        this.onCaption(data.speaker_name, data.original_text, 'original');
      } else {
        // Need translation to my language → call HPC /translate
        this.translateAndShow(
          data.original_text,
          sourceNllb,
          myLang,
          data.speaker_name
        );
      }
    } else if (data.type === 'language_set') {
      console.log('[Subtitle] Language prefs confirmed:', data);
    } else if (data.type === 'error') {
      console.error('[Subtitle] Server error:', data.message);
    }
  }

  /**
   * Call HPC /translate endpoint to translate text for the receiving user.
   * Each user independently translates to their own subtitleLang.
   */
  async translateAndShow(text, sourceNllb, targetNllb, speakerName) {
    if (!text || !sourceNllb || !targetNllb || sourceNllb === targetNllb) {
      if (this.onCaption) this.onCaption(speakerName, text, 'original');
      return;
    }

    try {
      const formData = new FormData();
      formData.append('text', text);
      formData.append('source_language', sourceNllb);
      formData.append('target_language', targetNllb);

      const resp = await fetch(`${this.proxyBase}/translate/`, {
        method: 'POST',
        body: formData,
        credentials: 'same-origin',
        headers: { 'X-CSRFToken': this.getCsrfToken() },
      });

      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const result = await resp.json();

      if (result.translated_text && result.success) {
        if (this.onCaption) {
          this.onCaption(speakerName, result.translated_text, 'translated');
        }
      } else {
        // Fallback to original text
        if (this.onCaption) this.onCaption(speakerName, text, 'original');
      }
    } catch (err) {
      console.error('[Subtitle] Translation error:', err);
      // Fallback to original text on error
      if (this.onCaption) this.onCaption(speakerName, text, 'original');
    }
  }

  // ─── Audio Recording ─────────────────────────────────────────

  async startRecording() {
    if (this.isRecording || !this.speakLang) return;

    try {
      // Request microphone — use a SEPARATE stream from Agora
      this.mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: SUBTITLE_CONFIG.SAMPLE_RATE,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        }
      });

      // Create AudioContext — try requested sample rate, fallback to browser default
      this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: SUBTITLE_CONFIG.SAMPLE_RATE,
      });

      // Check actual sample rate (browser may ignore our request)
      this.actualSampleRate = this.audioContext.sampleRate;
      if (this.actualSampleRate !== SUBTITLE_CONFIG.SAMPLE_RATE) {
        console.warn(
          `[Subtitle] AudioContext sampleRate: ${this.actualSampleRate} `
          + `(requested ${SUBTITLE_CONFIG.SAMPLE_RATE}). Will resample.`
        );
      }

      this.sourceNode = this.audioContext.createMediaStreamSource(this.mediaStream);

      // ScriptProcessor for raw PCM access
      this.processorNode = this.audioContext.createScriptProcessor(4096, 1, 1);
      this.sourceNode.connect(this.processorNode);
      this.processorNode.connect(this.audioContext.destination);

      this.isRecording = true;
      this.audioChunks = [];
      this.chunkStartTime = Date.now();
      this.silenceStart = null;
      this.hasSpeech = false;

      this.processorNode.onaudioprocess = (e) => {
        if (!this.isRecording) return;

        const pcm = new Float32Array(e.inputBuffer.getChannelData(0));
        this.audioChunks.push(pcm);

        // Calculate RMS for audio level
        const rms = Math.sqrt(
          pcm.reduce((sum, v) => sum + v * v, 0) / pcm.length
        );
        if (this.onAudioLevel) this.onAudioLevel(rms);

        this.handleChunkLogic(rms);
      };

      if (this.onStatus) this.onStatus('Recording...');
      console.log('[Subtitle] Recording started');

    } catch (err) {
      console.error('[Subtitle] Mic access error:', err);
      if (this.onStatus) this.onStatus('Mic error: ' + err.message);
    }
  }

  stopRecording() {
    if (!this.isRecording) return;
    this.isRecording = false;

    // Send remaining audio (if >= 1s)
    const totalSamples = this.audioChunks.reduce((s, c) => s + c.length, 0);
    if (totalSamples > this.actualSampleRate) {
      this.sendCurrentChunk();
    }

    // Clean up audio resources
    if (this.processorNode) {
      this.processorNode.disconnect();
      this.processorNode = null;
    }
    if (this.sourceNode) {
      this.sourceNode.disconnect();
      this.sourceNode = null;
    }
    if (this.audioContext) {
      this.audioContext.close().catch(() => { });
      this.audioContext = null;
    }
    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach(t => t.stop());
      this.mediaStream = null;
    }

    this.audioChunks = [];
    if (this.onStatus) this.onStatus('Stopped');
    if (this.onProgress) this.onProgress(0);
    console.log('[Subtitle] Recording stopped');
  }

  // ─── Chunk Logic ─────────────────────────────────────────────

  getChunkDuration() {
    if (this.chunkMode === 'custom') return this.customChunkDuration;
    if (this.chunkMode === 'auto') return SUBTITLE_CONFIG.AUTO_MAX_CHUNK_S;
    // Default: language-based
    return SUBTITLE_CONFIG.INDIC_CODES.has(this.speakLang)
      ? SUBTITLE_CONFIG.CHUNK_DURATION_INDIC
      : SUBTITLE_CONFIG.CHUNK_DURATION_EN;
  }

  handleChunkLogic(rms) {
    const totalSamples = this.audioChunks.reduce((s, c) => s + c.length, 0);
    const totalSeconds = totalSamples / this.actualSampleRate;
    const chunkDuration = this.getChunkDuration();

    if (this.chunkMode === 'auto') {
      // Auto-silence detection
      if (rms >= SUBTITLE_CONFIG.SILENCE_THRESHOLD_RMS) {
        this.hasSpeech = true;
        this.silenceStart = null;
      } else if (!this.silenceStart) {
        this.silenceStart = Date.now();
      }

      const silenceDuration = this.silenceStart ? (Date.now() - this.silenceStart) : 0;
      const shouldSend = (
        (this.hasSpeech
          && totalSeconds >= SUBTITLE_CONFIG.AUTO_MIN_CHUNK_S
          && silenceDuration >= SUBTITLE_CONFIG.SILENCE_GAP_MS)
        || totalSeconds >= SUBTITLE_CONFIG.AUTO_MAX_CHUNK_S
      );

      // Progress
      const progress = Math.min(totalSeconds / SUBTITLE_CONFIG.AUTO_MAX_CHUNK_S * 100, 100);
      if (this.onProgress) this.onProgress(progress);

      if (shouldSend && totalSeconds >= 1.0) {
        this.sendCurrentChunk();
      }
    } else {
      // Default / Custom: fixed duration
      const progress = Math.min(totalSeconds / chunkDuration * 100, 100);
      if (this.onProgress) this.onProgress(progress);

      if (totalSeconds >= chunkDuration) {
        this.sendCurrentChunk();
      }
    }
  }

  // ─── Resampling ─────────────────────────────────────────────

  /**
   * Downsample PCM data from actualSampleRate to targetRate.
   * Uses simple linear interpolation.
   */
  resample(samples, fromRate, toRate) {
    if (fromRate === toRate) return samples;

    const ratio = fromRate / toRate;
    const newLength = Math.round(samples.length / ratio);
    const result = new Float32Array(newLength);

    for (let i = 0; i < newLength; i++) {
      const srcIndex = i * ratio;
      const floor = Math.floor(srcIndex);
      const ceil = Math.min(floor + 1, samples.length - 1);
      const frac = srcIndex - floor;
      result[i] = samples[floor] * (1 - frac) + samples[ceil] * frac;
    }

    return result;
  }

  // ─── Send to HPC ─────────────────────────────────────────────

  async sendCurrentChunk() {
    if (this.audioChunks.length === 0 || this.isSending) return;

    // Grab chunks and reset
    const chunks = this.audioChunks;
    this.audioChunks = [];
    this.silenceStart = null;
    this.hasSpeech = false;
    this.chunkStartTime = Date.now();

    // Merge all chunks
    let allSamples = WavEncoder.mergeChunks(chunks);

    // Resample if AudioContext used a different sample rate
    if (this.actualSampleRate !== SUBTITLE_CONFIG.SAMPLE_RATE) {
      allSamples = this.resample(allSamples, this.actualSampleRate, SUBTITLE_CONFIG.SAMPLE_RATE);
    }

    // Encode to WAV
    const wavBlob = WavEncoder.encode(allSamples, SUBTITLE_CONFIG.SAMPLE_RATE);

    // Build form data — do NOT request translation here.
    // Each receiver translates independently via /translate.
    const formData = new FormData();
    formData.append('file', wavBlob, 'audio.wav');
    formData.append('language_mode', this.speakLang);
    formData.append('target_language', '');  // no sender-side translation

    this.isSending = true;
    if (this.onStatus) this.onStatus('Processing...');

    try {
      const resp = await fetch(`${this.proxyBase}/transcribe/`, {
        method: 'POST',
        body: formData,
        credentials: 'same-origin',
        headers: { 'X-CSRFToken': this.getCsrfToken() },
      });

      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const result = await resp.json();

      if (result.text && result.success) {
        // Determine source NLLB code for this transcription
        const sourceNllb = this.resolveSourceNllb(result);

        // Show own caption locally
        if (this.onCaption) {
          this.onCaption('You', result.text, 'self');
        }

        // Broadcast to room via WebSocket
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
          this.ws.send(JSON.stringify({
            type: 'transcription',
            original_text: result.text,
            source_nllb: sourceNllb,
            detected_language: result.detected_language || '',
          }));
        }
      } else if (result.note) {
        // "No speech detected" etc.
        if (this.onStatus) this.onStatus(result.note);
      }

    } catch (err) {
      console.error('[Subtitle] HPC error:', err);
      if (this.onStatus) this.onStatus('Server error: ' + err.message);
    } finally {
      this.isSending = false;
      if (this.isRecording && this.onStatus) this.onStatus('Recording...');
    }
  }

  /**
   * Determine the NLLB source language code for a transcription result.
   * For 'auto' mode, uses the HPC's detected_language field.
   * For explicit modes, uses the WHISPER_TO_NLLB mapping.
   */
  resolveSourceNllb(result) {
    // If we know the speak language and it's not 'auto', use the mapping
    if (this.speakLang && this.speakLang !== 'auto') {
      return SUBTITLE_CONFIG.WHISPER_TO_NLLB[this.speakLang] || 'eng_Latn';
    }

    // For 'auto' mode, use detected_language from HPC response
    const detected = result.detected_language || result.language_mode || '';
    if (detected) {
      return SUBTITLE_CONFIG.DETECTED_TO_NLLB[detected] || 'eng_Latn';
    }

    // Fallback
    return 'eng_Latn';
  }

  // ─── Language Selection ──────────────────────────────────────

  setSpeakLanguage(langCode) {
    const wasRecording = this.isRecording;
    if (wasRecording) this.stopRecording();

    this.speakLang = langCode;
    this.sendLanguagePrefs();

    // Start recording if a valid language is selected
    if (langCode) {
      this.startRecording();
    }
  }

  setSubtitleLanguage(nllbCode) {
    this.subtitleLang = nllbCode;
    this.sendLanguagePrefs();
  }

  setChunkMode(mode, customDuration) {
    this.chunkMode = mode;
    if (customDuration) this.customChunkDuration = customDuration;
  }
}


// ─── Subtitle UI Manager ─────────────────────────────────────────

class SubtitleUI {
  constructor(subtitleClient) {
    this.client = subtitleClient;
    this.captionContainer = null;
    this.maxCaptions = 5;         // keep last 5 captions visible
    this.captionTimeout = 12000;   // fade after 12s
  }

  /**
   * Initialize UI: create caption overlay + controls.
   * Call this after DOM is ready.
   */
  init() {
    this.createCaptionOverlay();
    this.createControlPanel();
    this.bindCallbacks();
  }

  createCaptionOverlay() {
    // Caption bar at bottom of video area
    const overlay = document.createElement('div');
    overlay.id = 'subtitleCaptionOverlay';
    overlay.className = 'subtitle-caption-overlay';
    overlay.innerHTML = `
            <div id="subtitleCaptions" class="subtitle-captions-list"></div>
        `;

    // Insert before the meeting controls bar
    const videoArea = document.querySelector('.col-12.col-lg-9');
    if (videoArea) {
      videoArea.appendChild(overlay);
    }

    this.captionContainer = document.getElementById('subtitleCaptions');
  }

  createControlPanel() {
    // Find the meeting controls bar and add subtitle button
    const controlsBar = document.querySelector('.meeting-controls .d-flex');
    if (!controlsBar) return;

    // Add subtitle toggle button (before the leave button)
    const leaveBtn = controlsBar.querySelector('.danger');
    const subtitleBtn = document.createElement('button');
    subtitleBtn.id = 'subtitleBtn';
    subtitleBtn.className = 'control-btn secondary';
    subtitleBtn.onclick = () => this.togglePanel();
    subtitleBtn.title = 'Subtitle Settings';
    subtitleBtn.innerHTML = '<i class="fas fa-closed-captioning"></i>';
    controlsBar.insertBefore(subtitleBtn, leaveBtn);

    // Create floating mini status bar (audio level + progress + status)
    const miniBar = document.createElement('div');
    miniBar.id = 'subtitleMiniBar';
    miniBar.className = 'subtitle-mini-bar';
    miniBar.style.display = 'none';
    miniBar.innerHTML = `
      <div class="subtitle-mini-inner">
        <span id="subtitleMiniStatus" class="subtitle-mini-status">Ready</span>
        <div class="subtitle-mini-level-container">
          <div id="subtitleMiniLevel" class="subtitle-mini-level"></div>
        </div>
        <div class="subtitle-mini-progress-container">
          <div id="subtitleMiniProgress" class="subtitle-mini-progress"></div>
        </div>
      </div>
    `;
    const meetingControls = document.querySelector('.meeting-controls');
    meetingControls.parentElement.insertBefore(miniBar, meetingControls);

    // Create the settings panel (hidden by default)
    const panel = document.createElement('div');
    panel.id = 'subtitleSettingsPanel';
    panel.className = 'subtitle-settings-panel';
    panel.style.display = 'none';
    panel.innerHTML = this.buildSettingsHTML();

    // Insert above meeting controls
    meetingControls.parentElement.insertBefore(panel, meetingControls);

    // Bind control events
    this.bindControlEvents();
  }

  buildSettingsHTML() {
    // Build target language options
    let targetOptions = '';
    for (const [name, code] of Object.entries(SUBTITLE_CONFIG.TARGET_LANGUAGES)) {
      targetOptions += `<option value="${code}">${name}</option>`;
    }

    return `
            <div class="subtitle-settings-header">
                <span><i class="fas fa-closed-captioning me-2"></i>Subtitle Settings</span>
                <button class="subtitle-close-btn" onclick="document.getElementById('subtitleSettingsPanel').style.display='none'">
                    <i class="fas fa-times"></i>
                </button>
            </div>
            <div class="subtitle-settings-body">
                <div class="subtitle-setting-row">
                    <label>Show subtitles in</label>
                    <select id="subtitleTargetLang">${targetOptions}</select>
                </div>
                <div class="subtitle-setting-row">
                    <label>Chunk mode</label>
                    <select id="subtitleChunkMode">
                        <option value="default">Default</option>
                        <option value="custom">Custom</option>
                        <option value="auto">Auto (Silence)</option>
                    </select>
                </div>
                <div class="subtitle-setting-row" id="subtitleCustomRow" style="display:none;">
                    <label>Duration (s)</label>
                    <input type="number" id="subtitleCustomDuration" min="3" max="30" value="8">
                </div>
                <div class="subtitle-setting-row">
                    <div class="subtitle-level-bar-container">
                        <div id="subtitleLevelBar" class="subtitle-level-bar"></div>
                    </div>
                    <div id="subtitleProgress" class="subtitle-progress-bar-container">
                        <div id="subtitleProgressBar" class="subtitle-progress-bar"></div>
                    </div>
                </div>
                <div id="subtitleStatus" class="subtitle-status">Ready</div>
            </div>
        `;
  }

  bindControlEvents() {
    const targetSel = document.getElementById('subtitleTargetLang');
    const modeSel = document.getElementById('subtitleChunkMode');
    const customRow = document.getElementById('subtitleCustomRow');
    const customDur = document.getElementById('subtitleCustomDuration');

    if (targetSel) {
      targetSel.addEventListener('change', () => {
        this.client.setSubtitleLanguage(targetSel.value);
      });
    }

    if (modeSel) {
      modeSel.addEventListener('change', () => {
        const mode = modeSel.value;
        if (customRow) customRow.style.display = mode === 'custom' ? 'flex' : 'none';
        this.client.setChunkMode(mode, parseInt(customDur.value) || 8);
      });
    }

    if (customDur) {
      customDur.addEventListener('change', () => {
        this.client.setChunkMode('custom', parseInt(customDur.value) || 8);
      });
    }
  }

  bindCallbacks() {
    this.client.onCaption = (speaker, text, type) => {
      this.addCaption(speaker, text, type);
    };

    this.client.onStatus = (msg) => {
      const el = document.getElementById('subtitleStatus');
      if (el) el.textContent = msg;
      const miniEl = document.getElementById('subtitleMiniStatus');
      if (miniEl) miniEl.textContent = msg;

      // Show/hide mini bar based on recording state
      const miniBar = document.getElementById('subtitleMiniBar');
      if (miniBar) {
        const isActive = msg === 'Recording...' || msg === 'Processing...';
        miniBar.style.display = isActive ? 'block' : 'none';
      }
    };

    this.client.onAudioLevel = (rms) => {
      const pct = Math.min(rms * 300, 100);
      const color = pct > 60 ? '#f80' : '#0f0';

      const bar = document.getElementById('subtitleLevelBar');
      if (bar) { bar.style.width = pct + '%'; bar.style.background = color; }
      const miniBar = document.getElementById('subtitleMiniLevel');
      if (miniBar) { miniBar.style.width = pct + '%'; miniBar.style.background = color; }
    };

    this.client.onProgress = (pct) => {
      const bar = document.getElementById('subtitleProgressBar');
      if (bar) bar.style.width = pct + '%';
      const miniBar = document.getElementById('subtitleMiniProgress');
      if (miniBar) miniBar.style.width = pct + '%';
    };
  }

  togglePanel() {
    const panel = document.getElementById('subtitleSettingsPanel');
    if (panel) {
      panel.style.display = panel.style.display === 'none' ? 'block' : 'none';
    }
  }

  updateBtnState(active) {
    const btn = document.getElementById('subtitleBtn');
    if (btn) {
      btn.className = active ? 'control-btn active' : 'control-btn secondary';
    }
  }

  addCaption(speaker, text, type) {
    if (!this.captionContainer || !text) return;

    const overlay = document.getElementById('subtitleCaptionOverlay');
    if (overlay) overlay.style.display = 'block';

    const caption = document.createElement('div');
    caption.className = `subtitle-caption ${type}`;
    caption.innerHTML = `
            <span class="subtitle-speaker">${this.escapeHtml(speaker)}</span>
            <span class="subtitle-text">${this.escapeHtml(text)}</span>
        `;
    this.captionContainer.appendChild(caption);

    // Animate in
    requestAnimationFrame(() => caption.classList.add('visible'));

    // Remove old captions
    while (this.captionContainer.children.length > this.maxCaptions) {
      this.captionContainer.removeChild(this.captionContainer.firstChild);
    }

    // Auto-fade
    setTimeout(() => {
      caption.classList.add('fading');
      setTimeout(() => {
        if (caption.parentNode) caption.parentNode.removeChild(caption);
      }, 500);
    }, this.captionTimeout);
  }

  escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  }
}
