# BridgeMeet: Real-Time Multilingual Subtitle Integration Plan

## System Overview

Integrate the **Whisper STT + NLLB Translation** pipeline (currently working as a standalone Python GUI) into the existing **Django + Agora WebRTC** meeting application, so each user sees real-time subtitles in their chosen language.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    USER A's BROWSER                             │
│                                                                 │
│  ┌──────────┐     ┌────────────────────────┐                    │
│  │ Microphone│────►│ Audio Buffer (JS)       │                   │
│  └──────────┘     │ MediaRecorder / PCM     │                   │
│       │           │ Chunk: 5-8s or silence  │                   │
│       │           └───────────┬─────────────┘                   │
│       ▼                       │                                 │
│  ┌──────────┐                 │ HTTP POST (WAV blob)            │
│  │ Agora SDK│                 ▼                                 │
│  │ WebRTC   │        ┌─────────────────┐                        │
│  │ ~50-100ms│        │ HPC SERVER (GPU)│                        │
│  └──────────┘        │ • Whisper STT   │                        │
│       │              │ • NLLB Translate │                        │
│       │              └────────┬────────┘                        │
│       │                       │ JSON { text, translations }     │
│       │                       ▼                                 │
│       │              ┌─────────────────┐                        │
│       │              │ Django Server   │                        │
│       │              │ WebSocket       │                        │
│       │              │ (SubtitleConsumer│                        │
│       │              └────────┬────────┘                        │
│       │                       │ broadcast to all in room        │
│       ▼                       ▼                                 │
│  Users HEAR audio     Users SEE subtitles                       │
│  in 100ms             in their language (4-5s)                  │
└─────────────────────────────────────────────────────────────────┘
```

### Key Insight: Per-User Language Selection

Each user in the meeting independently chooses:
1. **"I speak"** — their input language (Malayalam, Hindi, English, etc.)
2. **"Show subtitles in"** — their display language (what OTHER people's speech appears as)

When User A speaks Malayalam, the server transcribes it and translates to ALL needed target languages. User B (who wants Hindi subtitles) sees Hindi. User C (who wants English) sees English. All from the same audio.

---

## Existing Codebase Summary

### HPC Server (`BridgeMeet/server/server.py`) — **NO CHANGES NEEDED**
| Component | Details |
|-----------|---------|
| Framework | FastAPI, uvicorn |
| STT | Whisper `large-v3` (GPU) |
| VAD | Silero VAD (CPU) |
| Translation | NLLB-200 distilled 600M |
| Endpoint | `POST /transcribe` — accepts WAV file + `language_mode` + `target_language` |
| Features | Temperature fallback, hallucination filter, script validation, retry mechanism, via-English pipeline for low-resource languages |
| Docker | `pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime` |

### Django Meeting App (`videoconf/`)
| Component | Details |
|-----------|---------|
| Framework | Django 4.2.7 + Channels 4.0 + Daphne |
| WebRTC | Agora SDK (token-based) |
| WebSockets | MeetingConsumer, ChatConsumer, SignLanguageConsumer |
| Templates | `meeting_room.html`, `dashboard.html` |
| Auth | django-allauth |
| DB | SQLite (dev) / PostgreSQL ready |
| Existing Features | Video calls, chat, sign language detection |

### Test GUI (`transcribe_gui_nllb.py`) — Reference Only
PyQt6 GUI that proves the pipeline works. Features to port to browser:
- Audio recording with level meter
- 3 chunk modes (Default, Custom, Auto-silence)
- Language selection dropdowns
- Original speech + translation display

---

## Files to Create/Modify

### Django Backend

#### [NEW] `video_app/subtitle_consumer.py`
WebSocket consumer for subtitle broadcasting.

```python
class SubtitleConsumer(AsyncWebsocketConsumer):
    """
    Each user connects with their preferred subtitle language.
    When any user's transcription arrives, broadcast to all
    with per-user translation.
    """
    async def connect(self):
        self.room_id = self.scope['url_route']['kwargs']['room_id']
        self.room_group_name = f'subtitle_{self.room_id}'
        self.user_id = str(self.scope['user'].id)
        self.subtitle_language = 'eng_Latn'  # default
        self.speak_language = 'en'            # default
        await self.channel_layer.group_add(self.room_group_name, self.channel_name)
        await self.accept()

    async def receive(self, text_data):
        data = json.loads(text_data)
        msg_type = data['type']

        if msg_type == 'set_language':
            # User changed their subtitle/speak preference
            self.subtitle_language = data.get('subtitle_lang', 'eng_Latn')
            self.speak_language = data.get('speak_lang', 'en')

        elif msg_type == 'audio_transcription':
            # Browser sent audio to HPC, got transcription back,
            # now forward to all users in the room
            await self.channel_layer.group_send(
                self.room_group_name,
                {
                    'type': 'subtitle_broadcast',
                    'speaker_id': self.user_id,
                    'speaker_name': self.scope['user'].email,
                    'text': data['text'],                 # original speech
                    'language_mode': data['language_mode'],
                    'source_nllb': data.get('source_nllb', ''),
                    'timestamp': data.get('timestamp', ''),
                }
            )

    async def subtitle_broadcast(self, event):
        """
        Each user receives the broadcast and requests translation
        in their preferred language if needed.
        """
        await self.send(text_data=json.dumps({
            'type': 'subtitle',
            'speaker_id': event['speaker_id'],
            'speaker_name': event['speaker_name'],
            'text': event['text'],
            'source_nllb': event['source_nllb'],
            'my_subtitle_lang': self.subtitle_language,
        }))
```

#### [MODIFY] `video_app/routing.py`
Add subtitle WebSocket route:
```python
re_path(r'ws/subtitle/(?P<room_id>\w+)/$', consumers.SubtitleConsumer.as_asgi()),
```

#### [NEW] `video_app/views_api.py`
(Optional) Proxy endpoint if browser can't directly reach HPC server:
```python
@csrf_exempt
async def proxy_transcribe(request):
    """Forward audio to HPC server, return transcription."""
    # Forward to HPC_SERVER_URL/transcribe
    # Return JSON response
```

### Frontend (JavaScript)

#### [NEW] `static/js/subtitle_client.js`
Core subtitle client — port of `transcribe_gui_nllb.py` logic to browser JavaScript:

```javascript
class SubtitleClient {
    constructor(roomId, hpcServerUrl) {
        this.roomId = roomId;
        this.hpcUrl = hpcServerUrl;
        this.speakLanguage = 'en';
        this.subtitleLanguage = 'eng_Latn';  // NLLB code
        this.chunkMode = 'default';  // default | custom | auto
        this.chunkDuration = 8;       // seconds
        this.isRecording = false;

        // Audio recording
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.audioContext = null;
        this.analyser = null;

        // WebSocket for subtitle broadcast
        this.ws = null;

        // Auto-silence detection
        this.silenceThreshold = 0.03;
        this.silenceGapMs = 400;
        this.silenceStart = null;
        this.hasSpeech = false;
        this.recordingStart = null;
    }

    // Connect subtitle WebSocket
    connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        this.ws = new WebSocket(
            `${protocol}//${window.location.host}/ws/subtitle/${this.roomId}/`
        );
        this.ws.onmessage = (e) => this.onSubtitleReceived(JSON.parse(e.data));
    }

    // Start recording from microphone
    async startRecording() {
        const stream = await navigator.mediaDevices.getUserMedia({
            audio: { sampleRate: 16000, channelCount: 1 }
        });

        this.audioContext = new AudioContext({ sampleRate: 16000 });
        const source = this.audioContext.createMediaStreamSource(stream);

        // Audio level meter
        this.analyser = this.audioContext.createAnalyser();
        source.connect(this.analyser);

        // ScriptProcessor for raw PCM access (for silence detection)
        const processor = this.audioContext.createScriptProcessor(4096, 1, 1);
        source.connect(processor);
        processor.connect(this.audioContext.destination);

        processor.onaudioprocess = (e) => {
            if (!this.isRecording) return;
            const pcm = e.inputBuffer.getChannelData(0);
            this.audioChunks.push(new Float32Array(pcm));
            this.handleChunkLogic(pcm);
        };

        this.isRecording = true;
    }

    // Handle chunk timing based on mode
    handleChunkLogic(pcmFrame) {
        const totalSamples = this.audioChunks.reduce((s, c) => s + c.length, 0);
        const totalSeconds = totalSamples / 16000;

        if (this.chunkMode === 'auto') {
            // Silence detection
            const rms = Math.sqrt(
                pcmFrame.reduce((sum, v) => sum + v * v, 0) / pcmFrame.length
            );

            if (rms >= this.silenceThreshold) {
                this.hasSpeech = true;
                this.silenceStart = null;
            } else if (!this.silenceStart) {
                this.silenceStart = Date.now();
            }

            const silenceDuration = this.silenceStart
                ? (Date.now() - this.silenceStart) : 0;

            const shouldSend = (
                (this.hasSpeech && totalSeconds >= 1.5
                    && silenceDuration >= this.silenceGapMs)
                || totalSeconds >= 15.0
            );

            if (shouldSend && totalSeconds >= 1.0) {
                this.sendChunk();
            }
        } else {
            // Default / Custom: fixed duration
            if (totalSeconds >= this.chunkDuration) {
                this.sendChunk();
            }
        }
    }

    // Encode PCM to WAV and send to HPC
    async sendChunk() {
        const allSamples = this.mergeChunks(this.audioChunks);
        this.audioChunks = [];
        this.silenceStart = null;
        this.hasSpeech = false;

        const wavBlob = this.encodeWAV(allSamples, 16000);
        const formData = new FormData();
        formData.append('file', wavBlob, 'audio.wav');
        formData.append('language_mode', this.speakLanguage);
        formData.append('target_language', this.subtitleLanguage);

        try {
            const resp = await fetch(`${this.hpcUrl}/transcribe`, {
                method: 'POST',
                body: formData
            });
            const result = await resp.json();

            if (result.text) {
                // Show own caption locally
                this.showCaption(result.text, 'self');

                // Broadcast to room via WebSocket
                this.ws.send(JSON.stringify({
                    type: 'audio_transcription',
                    text: result.text,
                    translated_text: result.translated_text,
                    language_mode: result.language_mode,
                    source_nllb: result.source_nllb || '',
                    timestamp: new Date().toISOString()
                }));
            }
        } catch (err) {
            console.error('Transcription error:', err);
        }
    }

    // When another user's subtitle arrives via WebSocket
    onSubtitleReceived(data) {
        if (data.type === 'subtitle' && data.speaker_id !== currentUserId) {
            // Request translation to my language if needed
            this.translateAndShow(data);
        }
    }
}
```

#### [MODIFY] `templates/video_app/meeting_room.html`
Add subtitle UI controls to the existing meeting room:

```html
<!-- Subtitle Controls Dropdown -->
<div id="subtitle-controls" class="subtitle-panel">
    <select id="speak-language">
        <option value="en">English</option>
        <option value="ml">Malayalam (മലയാളം)</option>
        <option value="ml-roman">Malayalam → Manglish</option>
        <option value="ml-via-en">Malayalam (via English)</option>
        <option value="hi">Hindi (हिन्दी)</option>
        <option value="hi-en">Hindi + English Mix</option>
        <option value="ta">Tamil (தமிழ்)</option>
        <option value="te">Telugu (తెలుగు)</option>
        <option value="kn">Kannada (ಕನ್ನಡ)</option>
    </select>

    <select id="subtitle-language">
        <option value="">Off</option>
        <option value="eng_Latn">English</option>
        <option value="hin_Deva">Hindi</option>
        <option value="mal_Mlym">Malayalam</option>
        <option value="tam_Taml">Tamil</option>
        <!-- ... more NLLB codes -->
    </select>

    <select id="chunk-mode">
        <option value="default">Default</option>
        <option value="custom">Custom</option>
        <option value="auto">Auto (Silence)</option>
    </select>
</div>

<!-- Caption Overlay -->
<div id="caption-overlay" class="caption-bar">
    <div id="caption-text"></div>
</div>
```

#### [NEW] `static/css/subtitles.css`
Styling for caption overlay, controls panel, audio level meter.

---

## Data Flow (Step by Step)

### When User A speaks Malayalam and User B wants Hindi subtitles:

```
1. User A's browser captures mic audio (JavaScript AudioContext)
2. Audio chunk (8s or silence-triggered) → WAV blob
3. HTTP POST to HPC: /transcribe
   - file: audio.wav
   - language_mode: "ml" (Malayalam)
   - target_language: "" (no translation at HPC - all users need different targets)
4. HPC returns: { text: "എനിക്ക് നിങ്ങളെ കേൾക്കാം", detected_language: "ml" }
5. User A's browser sends via WebSocket to Django SubtitleConsumer:
   { type: "audio_transcription", text: "...", source_nllb: "mal_Mlym" }
6. Django broadcasts to ALL users in the room
7. Each user's browser receives the broadcast
8. User B's browser sees source_nllb="mal_Mlym", needs "hin_Deva"
   → HTTP POST to HPC: /translate
   - text: "എനിക്ക് നിങ്ങളെ കേൾക്കാം"
   - source_language: "mal_Mlym"
   - target_language: "hin_Deva"
9. HPC returns Hindi translation → User B sees Hindi caption
10. User C (wants English) → same /translate call → sees English caption
```

### Alternative: Server-Side Multi-Translation
Instead of each client calling `/translate`, the Django server can batch-translate for all target languages in the room and push pre-translated text to each user. This reduces client complexity but increases HPC load.

---

## File Structure (New/Modified)

```
videoconf/
├── video_app/
│   ├── consumers.py          # [MODIFY] import SubtitleConsumer
│   ├── subtitle_consumer.py  # [NEW] WebSocket subtitle handler
│   ├── routing.py            # [MODIFY] add ws/subtitle/ route
│   ├── views.py              # [MODIFY] add HPC proxy view (optional)
│   └── views_api.py          # [NEW] REST API for language config (optional)
├── templates/video_app/
│   └── meeting_room.html     # [MODIFY] add subtitle UI controls + caption overlay
├── static/
│   ├── js/
│   │   ├── subtitle_client.js # [NEW] core audio capture + subtitle logic
│   │   └── wav_encoder.js     # [NEW] PCM → WAV encoding utility
│   └── css/
│       └── subtitles.css      # [NEW] caption overlay styling
├── requirements.txt           # [MODIFY] add new dependencies
└── .env                       # [MODIFY] add HPC_SERVER_URL
```

---

## Environment Variables

Add to `.env`:
```
HPC_SERVER_URL=http://192.168.200.75:8200
```

---

## Implementation Order

1. **Static files first** — `subtitle_client.js`, `wav_encoder.js`, `subtitles.css`
2. **WebSocket consumer** — `subtitle_consumer.py` + routing
3. **Meeting room UI** — add controls to `meeting_room.html`
4. **Test 2 users** — open 2 browser tabs, verify subtitles appear cross-user
5. **Polish** — auto-silence mode, audio level meter, error handling
6. **CORS/proxy** — if browser can't reach HPC directly, add Django proxy

---

## Key Design Decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| Audio capture | Web Audio API (ScriptProcessor → AudioWorklet) | Native browser, no plugins |
| Audio format | 16kHz mono WAV | Whisper requires 16kHz |
| Chunk mode | Same 3 modes as GUI | Already tested and working |
| Translation routing | Client-side `/translate` calls | Simpler, no Django changes needed |
| Caption display | CSS overlay on video area | Non-intrusive, like YouTube captions |
| Subtitle WebSocket | Separate from MeetingConsumer | Clean separation of concerns |

---

## Prompt for Implementation

> **Implement real-time multilingual subtitles in my Django meeting app.**
>
> **Context**: I have a working HPC server at `http://192.168.200.75:8200` running Whisper large-v3 + NLLB-200 translation (see `server.py`). It accepts `POST /transcribe` (WAV file + language_mode + target_language) and `POST /translate` (text + source_language + target_language).
>
> **My Django app** (`videoconf/`) uses Django 4.2.7 + Channels 4.0 + Agora SDK for WebRTC video calls. It already has WebSocket consumers for meetings (`MeetingConsumer`), chat (`ChatConsumer`), and sign language detection (`SignLanguageConsumer`).
>
> **What I need**:
> 1. A **subtitle control panel** in `meeting_room.html` with: "I speak" dropdown (en, ml, hi, ta, etc.), "Show subtitles in" dropdown (NLLB language codes), chunk mode selector (Default/Custom/Auto-silence).
> 2. **JavaScript audio capture** (`subtitle_client.js`) that records mic audio in chunks, sends WAV to HPC `/transcribe` endpoint, and receives transcription.
> 3. A **SubtitleConsumer** WebSocket that broadcasts transcriptions to all users in the room.
> 4. Each user's browser translates received text to their preferred language by calling HPC `/translate`.
> 5. **Caption overlay** at the bottom of the video area showing subtitles with speaker name.
> 6. **Auto-silence detection** (400ms gap, RMS threshold 0.03) as an alternative to fixed chunks.
>
> **Important constraints**:
> - Do NOT modify `server.py` — the HPC server is deployed separately
> - Audio must be 16kHz mono WAV (Whisper requirement)
> - Use the existing Django Channels infrastructure (Redis channel layer)
> - Follow the same pattern as `SignLanguageConsumer` for WebSocket design
> - The subtitle WebSocket is SEPARATE from the meeting WebSocket (clean separation)
