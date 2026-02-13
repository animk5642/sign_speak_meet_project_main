#!/usr/bin/env python3
"""
simple_transcribe_gui.py - Speech-to-Text + NLLB Translation GUI

Uses HTTP POST for reliable transcription + neural translation.
Speech and translation shown in separate panels.

Features:
- Record 4-second chunks -> Whisper STT on HPC GPU
- Optional NLLB translation to 200+ languages (great for low-resource)
- Original speech text in one box, translation in another
- Language mode selector (Malayalam, Hindi, English, etc.)
- Target translation language selector

Usage:
  python simple_transcribe_gui.py
"""

import sys
import numpy as np
import threading
import time
import io
import requests

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QTextEdit, QProgressBar, QGroupBox,
    QSplitter, QFrame, QSpinBox
)
from PyQt6.QtCore import Qt, pyqtSignal, QObject
from PyQt6.QtGui import QFont, QColor, QPalette

import sounddevice as sd
import soundfile as sf

# =============================================================================
# CONFIGURATION
# =============================================================================

SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_DURATION = 8       # default for Indic languages (need more context)
CHUNK_DURATION_EN = 5    # shorter for English

# Languages that need longer chunks for reliable transcription
INDIC_LANGUAGE_CODES = {"ml", "ml-en", "ml-roman", "ml-via-en", "hi", "hi-en", "ta", "te", "kn"}

# Chunk mode options
CHUNK_MODE_DEFAULT = "Default"
CHUNK_MODE_CUSTOM  = "Custom"
CHUNK_MODE_AUTO    = "Auto (Silence)"

# Auto-silence detection parameters
# Average inter-sentence pause in human speech is 300-600ms.
# Ambient mic noise is typically RMS 0.01-0.03, speech is > 0.05.
SILENCE_THRESHOLD_RMS = 0.03    # RMS below this = silence
SILENCE_GAP_SECONDS = 0.4       # 400ms gap triggers send
AUTO_MIN_CHUNK = 1.5            # minimum 1.5s before sending
AUTO_MAX_CHUNK = 15.0           # force send after 15s regardless

DEFAULT_SERVER = "http://192.168.200.75:8200"

# Speech recognition language modes
STT_LANGUAGES = {
    "English": "en",
    "Hindi (हिन्दी)": "hi",
    "Hindi + English Mix": "hi-en",
    "Malayalam (മലയാളം)": "ml",
    "Malayalam → Manglish": "ml-roman",
    "Malayalam (via English)": "ml-via-en",
    "Malayalam + English Mix": "ml-en",
    "Tamil (தமிழ்)": "ta",
    "Telugu (తెలుగు)": "te",
    "Kannada (ಕನ್ನಡ)": "kn",
    "Auto-detect": "auto",
}

# NLLB translation target languages (200+ supported)
# Maps display name -> NLLB language code
NLLB_TARGET_LANGUAGES = {
    "None (No Translation)": "",
    # Indian languages
    "English": "eng_Latn",
    "Hindi (हिन्दी)": "hin_Deva",
    "Malayalam (മലയാളം)": "mal_Mlym",
    "Tamil (தமிழ்)": "tam_Taml",
    "Telugu (తెలుగు)": "tel_Telu",
    "Kannada (ಕನ್ನಡ)": "kan_Knda",
    "Bengali (বাংলা)": "ben_Beng",
    "Gujarati (ગુજરાતી)": "guj_Gujr",
    "Marathi (मराठी)": "mar_Deva",
    "Punjabi (ਪੰਜਾਬੀ)": "pan_Guru",
    "Urdu (اردو)": "urd_Arab",
    "Nepali (नेपाली)": "npi_Deva",
    "Sinhala (සිංහල)": "sin_Sinh",
    # Major world languages
    "Arabic (العربية)": "arb_Arab",
    "French (Français)": "fra_Latn",
    "Spanish (Español)": "spa_Latn",
    "German (Deutsch)": "deu_Latn",
    "Portuguese (Português)": "por_Latn",
    "Italian (Italiano)": "ita_Latn",
    "Russian (Русский)": "rus_Cyrl",
    "Chinese Simplified (中文)": "zho_Hans",
    "Japanese (日本語)": "jpn_Jpan",
    "Korean (한국어)": "kor_Hang",
    "Turkish (Türkçe)": "tur_Latn",
    "Vietnamese (Tiếng Việt)": "vie_Latn",
    "Thai (ภาษาไทย)": "tha_Thai",
    "Indonesian (Bahasa Indonesia)": "ind_Latn",
    "Swahili (Kiswahili)": "swh_Latn",
    "Burmese (မြန်မာ)": "mya_Mymr",
    "Khmer (ខ្មែរ)": "khm_Khmr",
}

# =============================================================================
# SIGNALS
# =============================================================================

class Signals(QObject):
    transcription = pyqtSignal(str, str, str, float, float)  # text, lang, translation, proc_t, transl_t
    status = pyqtSignal(str)
    level = pyqtSignal(float)
    error = pyqtSignal(str)
    recording_progress = pyqtSignal(float)

# =============================================================================
# SEPARATOR HELPER
# =============================================================================

def make_separator():
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setStyleSheet("background: #333; max-height: 1px; margin: 2px 0;")
    return line

# =============================================================================
# MAIN GUI
# =============================================================================

class SimpleTranscribeGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Whisper STT + NLLB Translation")
        self.setMinimumSize(680, 720)

        self.is_recording = False
        self.audio_data = []
        self.record_start_time = 0

        self.signals = Signals()
        self.signals.transcription.connect(self.on_transcription)
        self.signals.status.connect(self.on_status)
        self.signals.level.connect(self.on_level)
        self.signals.error.connect(self.on_error)
        self.signals.recording_progress.connect(self.on_progress)

        self.setup_ui()

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(10)
        layout.setContentsMargins(14, 14, 14, 14)

        # ── Title ─────────────────────────────────────────────────────
        title = QLabel("Whisper STT  +  NLLB Translation")
        title.setFont(QFont("Arial", 15, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("color: #7df; padding: 4px;")
        layout.addWidget(title)

        layout.addWidget(make_separator())

        # ── Server ────────────────────────────────────────────────────
        server_box = QGroupBox("Server")
        server_layout = QHBoxLayout(server_box)
        self.server_label = QLabel(f"HPC: {DEFAULT_SERVER}")
        self.server_label.setStyleSheet("color: #aaa; font-size: 11px;")
        server_layout.addWidget(self.server_label)
        layout.addWidget(server_box)

        # ── Language controls row ─────────────────────────────────────
        lang_row = QHBoxLayout()

        # Speech input language
        stt_box = QGroupBox("Speech Input Language")
        stt_layout = QHBoxLayout(stt_box)
        self.stt_combo = QComboBox()
        for name in STT_LANGUAGES.keys():
            self.stt_combo.addItem(name)
        self.stt_combo.setCurrentText("Malayalam (മലయാളം)")
        self.stt_combo.setMinimumWidth(190)
        self.stt_combo.currentTextChanged.connect(self.on_language_changed)
        stt_layout.addWidget(self.stt_combo)
        lang_row.addWidget(stt_box)

        # Translation target language
        trans_box = QGroupBox("Translate To (NLLB)")
        trans_layout = QHBoxLayout(trans_box)
        self.trans_combo = QComboBox()
        for name in NLLB_TARGET_LANGUAGES.keys():
            self.trans_combo.addItem(name)
        self.trans_combo.setCurrentText("English")
        self.trans_combo.setMinimumWidth(190)
        trans_layout.addWidget(self.trans_combo)
        lang_row.addWidget(trans_box)

        layout.addLayout(lang_row)

        # ── Audio Level ───────────────────────────────────────────────
        level_box = QGroupBox("Audio Level")
        level_layout = QVBoxLayout(level_box)
        level_layout.setContentsMargins(8, 4, 8, 4)
        self.level_bar = QProgressBar()
        self.level_bar.setRange(0, 100)
        self.level_bar.setValue(0)
        self.level_bar.setFixedHeight(14)
        self.level_bar.setStyleSheet("""
            QProgressBar { border: 1px solid #444; border-radius: 4px;
                           background: #1a1a1a; text-align: center; }
            QProgressBar::chunk { background: #0f0; border-radius: 3px; }
        """)
        level_layout.addWidget(self.level_bar)
        layout.addWidget(level_box)

        # ── Recording Progress ────────────────────────────────────────
        progress_box = QGroupBox(f"Recording Progress ({CHUNK_DURATION}s chunks)")
        progress_layout = QVBoxLayout(progress_box)
        progress_layout.setContentsMargins(8, 4, 8, 4)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(14)
        self.progress_bar.setStyleSheet("""
            QProgressBar { border: 1px solid #444; border-radius: 4px;
                           background: #1a1a1a; text-align: center; }
            QProgressBar::chunk { background: #f80; border-radius: 3px; }
        """)
        progress_layout.addWidget(self.progress_bar)
        self.progress_label = QLabel("Ready to record")
        self.progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.progress_label.setStyleSheet("color: #888; font-size: 11px;")
        progress_layout.addWidget(self.progress_label)
        layout.addWidget(progress_box)

        # ── Chunk Mode ─────────────────────────────────────────────────
        chunk_row = QHBoxLayout()

        mode_box = QGroupBox("Chunk Mode")
        mode_layout = QHBoxLayout(mode_box)
        self.chunk_mode_combo = QComboBox()
        self.chunk_mode_combo.addItems([CHUNK_MODE_DEFAULT, CHUNK_MODE_CUSTOM, CHUNK_MODE_AUTO])
        self.chunk_mode_combo.setCurrentText(CHUNK_MODE_DEFAULT)
        self.chunk_mode_combo.currentTextChanged.connect(self.on_chunk_mode_changed)
        mode_layout.addWidget(self.chunk_mode_combo)
        chunk_row.addWidget(mode_box)

        custom_box = QGroupBox("Chunk Duration (s)")
        custom_layout = QHBoxLayout(custom_box)
        self.chunk_spin = QSpinBox()
        self.chunk_spin.setRange(3, 30)
        self.chunk_spin.setValue(8)
        self.chunk_spin.setSuffix("s")
        self.chunk_spin.setEnabled(False)  # only enabled in Custom mode
        custom_layout.addWidget(self.chunk_spin)
        chunk_row.addWidget(custom_box)

        layout.addLayout(chunk_row)

        # ── Record Button ─────────────────────────────────────────────
        self.record_btn = QPushButton("  Start Recording")
        self.record_btn.setMinimumHeight(50)
        self.record_btn.setFont(QFont("Arial", 13, QFont.Weight.Bold))
        self.record_btn.clicked.connect(self.toggle_record)
        self._set_btn_idle()
        layout.addWidget(self.record_btn)

        # ── Status ────────────────────────────────────────────────────
        self.status_label = QLabel("Click to start recording")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(self.status_label)

        layout.addWidget(make_separator())

        # ── Results: two panels side by side ──────────────────────────
        results_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Panel A: Speech transcription
        speech_box = QGroupBox("Original Speech")
        speech_box.setStyleSheet("QGroupBox { font-weight: bold; color: #7df; }")
        speech_layout = QVBoxLayout(speech_box)

        self.speech_text = QTextEdit()
        self.speech_text.setReadOnly(True)
        self.speech_text.setMinimumHeight(150)
        self.speech_text.setStyleSheet("""
            QTextEdit {
                background: #0d1b2a;
                color: #e0f0ff;
                border: 1px solid #2a4a6a;
                padding: 8px;
                font-size: 13px;
            }
        """)
        speech_layout.addWidget(self.speech_text)

        speech_clear = QPushButton("Clear")
        speech_clear.setFixedHeight(24)
        speech_clear.setStyleSheet("background: #2a3a4a; color: #aaa; border-radius: 4px;")
        speech_clear.clicked.connect(lambda: self.speech_text.clear())
        speech_layout.addWidget(speech_clear)

        # Panel B: Translation
        trans_result_box = QGroupBox("Translation")
        trans_result_box.setStyleSheet("QGroupBox { font-weight: bold; color: #fd7; }")
        trans_result_layout = QVBoxLayout(trans_result_box)

        self.trans_text = QTextEdit()
        self.trans_text.setReadOnly(True)
        self.trans_text.setMinimumHeight(150)
        self.trans_text.setStyleSheet("""
            QTextEdit {
                background: #1a1a0d;
                color: #fff0d0;
                border: 1px solid #6a4a1a;
                padding: 8px;
                font-size: 13px;
            }
        """)
        trans_result_layout.addWidget(self.trans_text)

        trans_clear = QPushButton("Clear")
        trans_clear.setFixedHeight(24)
        trans_clear.setStyleSheet("background: #3a2a1a; color: #aaa; border-radius: 4px;")
        trans_clear.clicked.connect(lambda: self.trans_text.clear())
        trans_result_layout.addWidget(trans_clear)

        results_splitter.addWidget(speech_box)
        results_splitter.addWidget(trans_result_box)
        results_splitter.setSizes([340, 340])
        layout.addWidget(results_splitter)

    # ── Button style helpers ───────────────────────────────────────────

    def _set_btn_idle(self):
        self.record_btn.setText("  Start Recording")
        self.record_btn.setStyleSheet("""
            QPushButton {
                background: #2a7a2a; color: white;
                border: none; border-radius: 8px;
            }
            QPushButton:hover { background: #3a9a3a; }
        """)

    def _set_btn_recording(self):
        self.record_btn.setText("  Stop Recording")
        self.record_btn.setStyleSheet("""
            QPushButton {
                background: #8a1a1a; color: white;
                border: none; border-radius: 8px;
            }
            QPushButton:hover { background: #aa2a2a; }
        """)

    # ── Recording control ──────────────────────────────────────────────

    def toggle_record(self):
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self):
        self.is_recording = True
        self.audio_data = []
        self.record_start_time = time.time()
        self._set_btn_recording()
        lang_name = self.stt_combo.currentText()
        self.signals.status.emit(f"Recording... ({lang_name})")
        threading.Thread(target=self.record_thread, daemon=True).start()

    def get_chunk_duration(self):
        """Get appropriate chunk duration based on mode and language."""
        mode = self.chunk_mode_combo.currentText()
        if mode == CHUNK_MODE_CUSTOM:
            return self.chunk_spin.value()
        elif mode == CHUNK_MODE_AUTO:
            return AUTO_MAX_CHUNK  # max limit for auto
        # Default mode: language-based
        stt_code = STT_LANGUAGES.get(self.stt_combo.currentText(), "en")
        if stt_code in INDIC_LANGUAGE_CODES:
            return CHUNK_DURATION  # 8s for Indic
        return CHUNK_DURATION_EN   # 5s for English

    def on_chunk_mode_changed(self, mode):
        """Enable/disable chunk duration spinbox based on mode."""
        self.chunk_spin.setEnabled(mode == CHUNK_MODE_CUSTOM)
        if mode == CHUNK_MODE_AUTO:
            self.progress_label.setText("Ready — auto-detect silence")
        elif mode == CHUNK_MODE_CUSTOM:
            self.progress_label.setText(f"Ready — {self.chunk_spin.value()}s chunks")
        else:
            dur = self.get_chunk_duration()
            self.progress_label.setText(f"Ready — {dur}s chunks")

    def on_language_changed(self, lang_name):
        """Update UI when language changes."""
        mode = self.chunk_mode_combo.currentText()
        if mode == CHUNK_MODE_DEFAULT:
            duration = self.get_chunk_duration()
            self.progress_label.setText(f"Ready — {duration}s chunks")

    def stop_recording(self):
        self.is_recording = False
        self.progress_bar.setValue(0)
        self.progress_label.setText("Processing...")
        self._set_btn_idle()

    # ── Audio recording thread ─────────────────────────────────────────

    def record_thread(self):
        """Record audio in chunks and ship each to the server."""
        try:
            chunk_mode = self.chunk_mode_combo.currentText()
            duration = self.get_chunk_duration()
            chunk_samples = int(duration * SAMPLE_RATE)

            # For auto-silence mode: track silence duration
            silence_start = [None]  # mutable ref for closure
            has_speech = [False]

            def callback(indata, frames, time_info, status):
                if not self.is_recording:
                    return

                self.audio_data.extend(indata.copy().flatten())

                rms = np.sqrt(np.mean(indata ** 2))
                self.signals.level.emit(rms)

                if chunk_mode == CHUNK_MODE_AUTO:
                    # Auto-silence detection mode
                    audio_len_s = len(self.audio_data) / SAMPLE_RATE
                    progress = min(audio_len_s / AUTO_MAX_CHUNK * 100, 100)
                    self.signals.recording_progress.emit(progress)

                    if rms >= SILENCE_THRESHOLD_RMS:
                        # Speech detected
                        has_speech[0] = True
                        silence_start[0] = None
                    else:
                        # Silence detected
                        if silence_start[0] is None:
                            silence_start[0] = time.time()

                    # Send conditions for auto mode:
                    # 1. Had speech + silence gap >= 500ms + min chunk reached
                    # 2. OR max chunk limit reached
                    silence_duration = (
                        (time.time() - silence_start[0])
                        if silence_start[0] else 0
                    )
                    should_send = (
                        (has_speech[0]
                         and audio_len_s >= AUTO_MIN_CHUNK
                         and silence_duration >= SILENCE_GAP_SECONDS)
                        or audio_len_s >= AUTO_MAX_CHUNK
                    )

                    if should_send and len(self.audio_data) > SAMPLE_RATE:
                        audio_chunk = np.array(
                            self.audio_data, dtype=np.float32
                        )
                        self.audio_data = []
                        silence_start[0] = None
                        has_speech[0] = False
                        threading.Thread(
                            target=self.send_to_server,
                            args=(audio_chunk,),
                            daemon=True
                        ).start()
                else:
                    # Default / Custom mode: fixed chunk duration
                    progress = min(len(self.audio_data) / chunk_samples * 100, 100)
                    self.signals.recording_progress.emit(progress)

                    if len(self.audio_data) >= chunk_samples:
                        audio_chunk = np.array(
                            self.audio_data[:chunk_samples], dtype=np.float32
                        )
                        self.audio_data = self.audio_data[chunk_samples:]
                        threading.Thread(
                            target=self.send_to_server,
                            args=(audio_chunk,),
                            daemon=True
                        ).start()

            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype=np.float32,
                callback=callback
            ):
                while self.is_recording:
                    time.sleep(0.1)

            # Send remaining audio if >= 1 second
            if len(self.audio_data) > SAMPLE_RATE:
                audio_chunk = np.array(self.audio_data, dtype=np.float32)
                self.send_to_server(audio_chunk)

            self.signals.status.emit("Recording stopped")
            self.signals.recording_progress.emit(0)

        except Exception as e:
            self.signals.error.emit(f"Audio error: {e}")

    # ── Send audio to server ───────────────────────────────────────────

    def send_to_server(self, audio: np.ndarray):
        """Send audio chunk to HPC server; receive STT + optional translation."""
        try:
            self.signals.status.emit("Sending to server...")

            wav_buffer = io.BytesIO()
            sf.write(wav_buffer, audio, SAMPLE_RATE, format='WAV')
            wav_buffer.seek(0)

            # STT language mode
            stt_name = self.stt_combo.currentText()
            stt_code = STT_LANGUAGES.get(stt_name, "en")

            # Translation target (empty string = no translation)
            trans_name = self.trans_combo.currentText()
            target_nllb = NLLB_TARGET_LANGUAGES.get(trans_name, "")

            start_time = time.time()
            response = requests.post(
                f"{DEFAULT_SERVER}/transcribe",
                files={"file": ("audio.wav", wav_buffer, "audio/wav")},
                data={
                    "language_mode": stt_code,
                    "target_language": target_nllb,
                },
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                text = result.get("text", "").strip()
                translated = result.get("translated_text", "").strip()
                detected = result.get("detected_language", stt_code)
                proc_time = result.get("processing_time", time.time() - start_time)
                transl_time = result.get("translation_time", 0.0)

                if text:
                    self.signals.transcription.emit(
                        text, detected, translated, proc_time, transl_time
                    )
                    status_msg = f"STT: {proc_time:.2f}s"
                    if translated:
                        status_msg += f"  |  Translation: {transl_time:.2f}s"
                    self.signals.status.emit(f"Done — {status_msg}")
                else:
                    note = result.get("note", "No speech detected")
                    self.signals.status.emit(note)
            else:
                self.signals.error.emit(f"Server error: {response.status_code}")

        except requests.exceptions.ConnectionError:
            self.signals.error.emit(f"Cannot connect to {DEFAULT_SERVER}")
        except Exception as e:
            self.signals.error.emit(f"Error: {e}")

    # ── Qt slot handlers ───────────────────────────────────────────────

    def on_level(self, rms):
        self.level_bar.setValue(int(min(rms * 200, 100)))

    def on_progress(self, progress):
        self.progress_bar.setValue(int(progress))
        remaining = CHUNK_DURATION - (progress / 100 * CHUNK_DURATION)
        self.progress_label.setText(f"Recording: {remaining:.1f}s remaining")

    def on_transcription(self, text, lang, translation, proc_time, transl_time):
        timestamp = time.strftime("%H:%M:%S")

        # ── Speech panel ──────────────────────────────────────────────
        speech_html = (
            f"<span style='color:#555;font-size:10px;'>[{timestamp}]</span> "
            f"<span style='color:#7df;font-size:10px;'>[{lang.upper()}]</span> "
            f"<span style='font-size:13px;'>{text}</span> "
            f"<span style='color:#555;font-size:10px;'>({proc_time:.2f}s)</span><br>"
        )
        self.speech_text.insertHtml(speech_html)
        self.speech_text.verticalScrollBar().setValue(
            self.speech_text.verticalScrollBar().maximum()
        )

        # ── Translation panel ─────────────────────────────────────────
        trans_name = self.trans_combo.currentText()
        target_code = NLLB_TARGET_LANGUAGES.get(trans_name, "")

        if target_code and translation:
            trans_html = (
                f"<span style='color:#555;font-size:10px;'>[{timestamp}]</span> "
                f"<span style='color:#fd7;font-size:10px;'>[{trans_name[:12]}]</span> "
                f"<span style='font-size:13px;'>{translation}</span> "
                f"<span style='color:#555;font-size:10px;'>({transl_time:.2f}s)</span><br>"
            )
            self.trans_text.insertHtml(trans_html)
            self.trans_text.verticalScrollBar().setValue(
                self.trans_text.verticalScrollBar().maximum()
            )
        elif target_code and not translation:
            # Translation was requested but returned empty (NLLB unavailable, etc.)
            self.trans_text.insertHtml(
                f"<span style='color:#866;font-size:11px;'>[{timestamp}] Translation unavailable</span><br>"
            )
            self.trans_text.verticalScrollBar().setValue(
                self.trans_text.verticalScrollBar().maximum()
            )

    def on_status(self, msg):
        self.status_label.setText(msg)

    def on_error(self, msg):
        self.status_label.setText(f"Error: {msg}")
        print(f"[ERROR] {msg}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Dark palette
    p = QPalette()
    p.setColor(QPalette.ColorRole.Window,        QColor(28, 28, 38))
    p.setColor(QPalette.ColorRole.WindowText,    QColor(220, 220, 235))
    p.setColor(QPalette.ColorRole.Base,          QColor(20, 20, 30))
    p.setColor(QPalette.ColorRole.AlternateBase, QColor(35, 35, 50))
    p.setColor(QPalette.ColorRole.Text,          QColor(220, 220, 235))
    p.setColor(QPalette.ColorRole.Button,        QColor(45, 45, 65))
    p.setColor(QPalette.ColorRole.ButtonText,    QColor(220, 220, 235))
    p.setColor(QPalette.ColorRole.Highlight,     QColor(60, 140, 200))
    p.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    app.setPalette(p)

    window = SimpleTranscribeGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()