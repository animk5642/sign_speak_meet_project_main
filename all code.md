
this is the server code for indic tranlator

#!/usr/bin/env python3
"""
this is the ned
IndicConformer STT + IndicTrans2 Translation Server
- STT  : ai4bharat/indic-conformer-600m-multilingual  (22 Indian languages)
- Trans: ai4bharat/indictrans2-*-1B                   (best accuracy for Indian langs)

Translation directions:
  Indian → English   : indictrans2-indic-en-1B
  English → Indian   : indictrans2-en-indic-1B
  Indian → Indian    : indictrans2-indic-indic-1B
"""

import os
import time
import tempfile
import shutil
import threading
import traceback
from typing import Tuple

import torch
import torchaudio
from fastapi import FastAPI, File, UploadFile, Form, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import (
    AutoModel,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    logging as hf_logging,
)

hf_logging.set_verbosity_error()

app = FastAPI(title="IndicConformer STT + IndicTrans2", version="7.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
IC_MODEL_NAME = os.environ.get("IC_MODEL", "ai4bharat/indic-conformer-600m-multilingual")
HF_TOKEN      = os.environ.get("HF_TOKEN", None)
API_KEY       = os.environ.get("API_KEY", "")

IT2_INDIC_EN    = "ai4bharat/indictrans2-indic-en-1B"
IT2_EN_INDIC    = "ai4bharat/indictrans2-en-indic-1B"
IT2_INDIC_INDIC = "ai4bharat/indictrans2-indic-indic-1B"

print("=" * 60)
print("IndicConformer STT + IndicTrans2 Server v7.0")
print("Device    :", DEVICE)
print("HF Token  :", HF_TOKEN is not None)
print("=" * 60)

# -----------------------------------------------------------------------------
# LOAD INDICCONFORMER (STT)
# -----------------------------------------------------------------------------

ic_model = None
ic_lock   = threading.Lock()

try:
    print("[STT] Loading IndicConformer...")
    ic_model = AutoModel.from_pretrained(
        IC_MODEL_NAME, token=HF_TOKEN, trust_remote_code=True,
    ).to(DEVICE)
    ic_model.eval()
    print(f"[STT] IndicConformer loaded on {DEVICE} ✓")
except Exception as e:
    print("[STT] FAILED:", e)
    traceback.print_exc()
    ic_model = None

# -----------------------------------------------------------------------------
# LOAD INDICTRANS2
# -----------------------------------------------------------------------------

try:
    try:
        from IndicTransToolkit import IndicProcessor
    except ImportError:
        from IndicTransToolkit.IndicTransToolkit import IndicProcessor
    INDIC_PROCESSOR_AVAILABLE = True
    print("[TRANS] IndicTransToolkit loaded ✓")
except Exception as e:
    INDIC_PROCESSOR_AVAILABLE = False
    print("[TRANS] IndicTransToolkit FAILED:", e)
    traceback.print_exc()

it2_models     = {}
it2_tokenizers = {}
it2_lock       = threading.Lock()

def _load_it2(key: str, model_name: str):
    try:
        print(f"[TRANS] Loading {key}...")
        tok = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, token=HF_TOKEN
        )
        mdl = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            token=HF_TOKEN,
            torch_dtype=torch.float16,
        ).to(DEVICE)
        mdl.eval()
        it2_tokenizers[key] = tok
        it2_models[key]     = mdl
        print(f"[TRANS] {key} loaded on {DEVICE} ✓")
    except Exception as e:
        print(f"[TRANS] {key} FAILED: {e}")
        traceback.print_exc()

if INDIC_PROCESSOR_AVAILABLE:
    _load_it2("indic-en",    IT2_INDIC_EN)
    _load_it2("en-indic",    IT2_EN_INDIC)
    _load_it2("indic-indic", IT2_INDIC_INDIC)
else:
    print("[TRANS] Skipping IndicTrans2 — IndicTransToolkit not available")

# -----------------------------------------------------------------------------
# LANGUAGE MAPS
# -----------------------------------------------------------------------------

SUPPORTED_LANG_CODES = {
    "as": "as", "bn": "bn", "brx": "brx", "doi": "doi",
    "gu": "gu", "hi": "hi", "kn":  "kn",  "ks":  "ks",
    "kok": "kok", "mai": "mai", "ml": "ml", "mni": "mni",
    "mr": "mr", "ne": "ne", "or": "or", "pa": "pa",
    "sa": "sa", "sat": "sat", "sd": "sd", "ta": "ta",
    "te": "te", "ur": "ur", "en": "en",
}

IT2_LANG_CODES = {
    "as":  "asm_Beng", "bn":  "ben_Beng", "brx": "brx_Deva",
    "doi": "doi_Deva", "gu":  "guj_Gujr", "hi":  "hin_Deva",
    "kn":  "kan_Knda", "ks":  "kas_Arab", "kok": "kok_Deva",
    "mai": "mai_Deva", "ml":  "mal_Mlym", "mni": "mni_Mtei",
    "mr":  "mar_Deva", "ne":  "npi_Deva", "or":  "ory_Orya",
    "pa":  "pan_Guru", "sa":  "san_Deva", "sat": "sat_Olck",
    "sd":  "snd_Arab", "ta":  "tam_Taml", "te":  "tel_Telu",
    "ur":  "urd_Arab", "en":  "eng_Latn",
}

def _it2_direction(src: str, tgt: str) -> str:
    if src == "en":  return "en-indic"
    if tgt == "en":  return "indic-en"
    return "indic-indic"

# -----------------------------------------------------------------------------
# TRANSCRIPTION
# -----------------------------------------------------------------------------

def load_audio_as_tensor(audio_path: str) -> torch.Tensor:
    wav, sr = torchaudio.load(audio_path)
    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(wav)
    return wav


def transcribe_audio_ic(audio_path: str, language_mode: str = "ml") -> Tuple[str, str]:
    if ic_model is None:
        raise RuntimeError("IndicConformer not loaded")
    lang_code = SUPPORTED_LANG_CODES.get(language_mode, "ml")
    wav = load_audio_as_tensor(audio_path)
    with ic_lock:
        with torch.no_grad():
            text = ic_model(wav, lang_code, "ctc")
    if isinstance(text, list):
        text = text[0]
    return str(text).strip(), lang_code

# -----------------------------------------------------------------------------
# TRANSLATION
# -----------------------------------------------------------------------------

def translate_text(text: str, src_lang: str, tgt_lang: str) -> str:
    if not text or not INDIC_PROCESSOR_AVAILABLE:
        return ""

    direction = _it2_direction(src_lang, tgt_lang)

    if direction not in it2_models:
        return f"[Translation model '{direction}' not available]"

    src_it2 = IT2_LANG_CODES.get(src_lang)
    tgt_it2 = IT2_LANG_CODES.get(tgt_lang)

    if not src_it2 or not tgt_it2:
        return f"[Unsupported pair: {src_lang} → {tgt_lang}]"

    tokenizer = it2_tokenizers[direction]
    model     = it2_models[direction]
    ip        = IndicProcessor(inference=True)

    try:
        with it2_lock:
            batch = ip.preprocess_batch([text], src_lang=src_it2, tgt_lang=tgt_it2)
            inputs = tokenizer(
                batch,
                truncation=True,
                padding="longest",
                return_tensors="pt",
                return_attention_mask=True,
            ).to(DEVICE)

            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    use_cache=True,
                    min_length=0,
                    max_length=512,
                    num_beams=4,
                    num_return_sequences=1,
                )

            with tokenizer.as_target_tokenizer():
                decoded = tokenizer.batch_decode(
                    generated.detach().cpu().tolist(),
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )

            translations = ip.postprocess_batch(decoded, lang=tgt_it2)
            return translations[0] if translations else ""

    except Exception as e:
        print(f"[TRANS] Runtime error: {e}")
        traceback.print_exc()
        return f"[Translation error: {e}]"

# -----------------------------------------------------------------------------
# AUTH
# -----------------------------------------------------------------------------

def check_api_key(x_api_key: str):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

# -----------------------------------------------------------------------------
# API
# -----------------------------------------------------------------------------

@app.get("/")
async def root():
    return {
        "status": "running",
        "stt_model": IC_MODEL_NAME,
        "translation_models": {
            "indic-en":    IT2_INDIC_EN,
            "en-indic":    IT2_EN_INDIC,
            "indic-indic": IT2_INDIC_INDIC,
        },
        "device": DEVICE,
        "stt_loaded": ic_model is not None,
        "translation_loaded": list(it2_models.keys()),
        "supported_languages": IT2_LANG_CODES,
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "device": DEVICE,
        "stt_loaded": ic_model is not None,
        "translation_loaded": list(it2_models.keys()),
    }

@app.post("/transcribe")
async def transcribe(
    file:            UploadFile = File(...),
    language_mode:   str        = Form(default="ml"),
    target_language: str        = Form(default=""),
    x_api_key:       str        = Header(default=""),
):
    check_api_key(x_api_key)
    start_time = time.time()

    suffix = os.path.splitext(file.filename or "audio.wav")[-1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        text, detected_lang = transcribe_audio_ic(tmp_path, language_mode)

        translated_text  = ""
        translation_time = 0.0

        if target_language and target_language != language_mode:
            t0 = time.time()
            translated_text  = translate_text(text, detected_lang, target_language)
            translation_time = round(time.time() - t0, 3)

        return {
            "text":              text,
            "translated_text":   translated_text,
            "language_mode":     language_mode,
            "detected_language": detected_lang,
            "target_language":   target_language,
            "processing_time":   round(time.time() - start_time, 3),
            "translation_time":  translation_time,
            "success":           True,
        }

    except Exception as e:
        traceback.print_exc()
        return {
            "text": "", "translated_text": "",
            "language_mode": language_mode,
            "detected_language": "error",
            "target_language": target_language,
            "processing_time": round(time.time() - start_time, 3),
            "translation_time": 0.0,
            "success": False, "error": str(e),
        }

    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


this is the docker file for indic trnslator
# ---------------------------------------------------------
# CUDA base this should be end
# ---------------------------------------------------------
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# ---------------------------------------------------------
# System deps
# ---------------------------------------------------------
RUN apt-get update && apt-get install -y \
  python3.10 \
  python3.10-distutils \
  python3-pip \
  git \
  ffmpeg \
  libsndfile1 \
  curl \
  && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3.10 /usr/bin/python
RUN python -m pip install --upgrade pip setuptools wheel

# ---------------------------------------------------------
# Remove conflicting preinstalls
# ---------------------------------------------------------
RUN pip uninstall -y \
  numpy torch torchvision torchaudio \
  transformers huggingface_hub || true

# ---------------------------------------------------------
# PyTorch CUDA 12.1
# ---------------------------------------------------------
RUN pip install --no-cache-dir \
  torch==2.4.1+cu121 \
  torchvision==0.19.1+cu121 \
  torchaudio==2.4.1+cu121 \
  --index-url https://download.pytorch.org/whl/cu121

# ---------------------------------------------------------
# Core ML stack
# KEY FIX: transformers>=4.40.0 required by IndicTrans2
# Previous version 4.38.2 caused silent model load failures
# ---------------------------------------------------------
RUN pip install --no-cache-dir \
  "numpy<2.0" \
  "transformers>=4.40.0,<4.46.0" \
  "tokenizers>=0.19.0" \
  "huggingface_hub>=0.23.0" \
  "accelerate>=0.28.0" \
  sentencepiece \
  sacremoses \
  librosa \
  soundfile \
  fastapi \
  uvicorn[standard] \
  python-multipart \
  onnxruntime-gpu

# ---------------------------------------------------------
# IndicTransToolkit — PyPI, no git needed
# ---------------------------------------------------------
RUN pip install --no-cache-dir indictranstoolkit

# ---------------------------------------------------------
# App
# ---------------------------------------------------------
WORKDIR /app
COPY server.py /app/server.py

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]



whisper


#!/usr/bin/env python3
"""
server.py - Whisper Speech-to-Text Server with VAD + NLLB Translation

Features:
- Voice Activity Detection (Silero VAD) - ONLY transcribe when speech detected
- Real-time speech recognition
- Output in SAME language as input (Malayalam, Hindi, English, etc.)
- GPU acceleration with large-v3 model
- NLLB-200 neural machine translation (600M distilled)
- Supports 200+ languages including low-resource Indian languages
- Optimized for accuracy

Run with:
  uvicorn server:app --host 0.0.0.0 --port 8000
"""

from typing import Tuple, Optional
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import whisper
import torch
import tempfile
import shutil
import os
import time
import threading
import numpy as np
import soundfile as sf

app = FastAPI(title="Whisper STT + NLLB Translation Server", version="4.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# CONFIGURATION
# =========================

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"{'='*60}")
print(f"Whisper STT + NLLB Translation Server")
print(f"{'='*60}")
print(f"Device: {device}")

if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Load Whisper large-v3 for best accuracy
MODEL_NAME = "large-v3"
print(f"\nLoading Whisper model: {MODEL_NAME}...")
model = whisper.load_model(MODEL_NAME, device=device)
print(f"Whisper model loaded!")

# Load Silero VAD (runs on CPU for compatibility)
print(f"\nLoading Silero VAD...")
vad_model, vad_utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    force_reload=False,
    onnx=False
)
(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = vad_utils
vad_model = vad_model.cpu()
print(f"VAD loaded (on CPU)!")

# Load NLLB Translation Model
print(f"\nLoading NLLB-200 translation model...")
NLLB_AVAILABLE = False
NLLB_MODEL_NAME = "facebook/nllb-200-distilled-600M"
nllb_tokenizer = None
nllb_model_obj = None

try:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    nllb_tokenizer = AutoTokenizer.from_pretrained(NLLB_MODEL_NAME)
    nllb_model_obj = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL_NAME)
    nllb_model_obj = nllb_model_obj.to(device)
    nllb_model_obj.eval()
    NLLB_AVAILABLE = True
    print(f"NLLB translation model loaded ({NLLB_MODEL_NAME})!")
except Exception as e:
    print(f"WARNING: NLLB model failed to load: {e}")
    print(f"  Translation will be unavailable.")

# Thread lock for NLLB to prevent concurrent translation issues
nllb_lock = threading.Lock()

print(f"{'='*60}")

# VAD Configuration
VAD_THRESHOLD = 0.3
VAD_THRESHOLD_INDIC = 0.4     # Stricter for pure Indic — reduces false speech detections
MIN_SPEECH_DURATION = 0.3

# =========================
# LANGUAGE MAPPINGS
# =========================

LANGUAGE_CONFIG = {
    "ml": {
        "whisper_lang": "ml",
        "prompt": "മലയാളത്തിൽ ദൈനംദിന സംഭാഷണം. രണ്ടു പേർ തമ്മിൽ സംസാരിക്കുന്നു. വീഡിയോ കോൺഫറൻസ് മീറ്റിംഗ്.",
        "nllb_code": "mal_Mlym",
        "description": "Malayalam"
    },
    "ml-en": {
        "whisper_lang": "ml",
        "prompt": "This is a mix of Malayalam and English. ഇത് മലയാളവും ഇംഗ്ലീഷും ആണ്.",
        "nllb_code": "mal_Mlym",
        "description": "Malayalam + English mixed"
    },
    "ml-roman": {
        "whisper_lang": "ml",
        "prompt": "",
        "nllb_code": "mal_Mlym",
        "description": "Malayalam → Romanized (Manglish)",
        "use_translate": True
    },
    "ml-via-en": {
        "whisper_lang": "ml",
        "prompt": "",
        "nllb_code": "mal_Mlym",
        "description": "Malayalam (via English bridge)",
        "use_translate": True,
        "via_english_target": "mal_Mlym"
    },
    "hi": {
        "whisper_lang": "hi",
        "prompt": "यह हिंदी भाषा है।",
        "nllb_code": "hin_Deva",
        "description": "Hindi"
    },
    "hi-en": {
        "whisper_lang": "hi",
        "prompt": "This is a mix of Hindi and English. यह हिंदी और अंग्रेजी का मिश्रण है।",
        "nllb_code": "hin_Deva",
        "description": "Hindi + English mixed"
    },
    "en": {
        "whisper_lang": "en",
        "prompt": "",
        "nllb_code": "eng_Latn",
        "description": "English"
    },
    "ta": {
        "whisper_lang": "ta",
        "prompt": "இது தமிழ் மொழி.",
        "nllb_code": "tam_Taml",
        "description": "Tamil"
    },
    "te": {
        "whisper_lang": "te",
        "prompt": "ఇది తెలుగు భాష.",
        "nllb_code": "tel_Telu",
        "description": "Telugu"
    },
    "kn": {
        "whisper_lang": "kn",
        "prompt": "ಇದು ಕನ್ನಡ ಭಾಷೆ.",
        "nllb_code": "kan_Knda",
        "description": "Kannada"
    },
    "auto": {
        "whisper_lang": None,
        "prompt": "",
        "nllb_code": None,
        "description": "Auto-detect language"
    }
}

# Whisper language code -> NLLB language code
WHISPER_TO_NLLB = {
    "ml": "mal_Mlym",
    "hi": "hin_Deva",
    "en": "eng_Latn",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "kn": "kan_Knda",
    "bn": "ben_Beng",
    "gu": "guj_Gujr",
    "mr": "mar_Deva",
    "pa": "pan_Guru",
    "ur": "urd_Arab",
    "ar": "arb_Arab",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "es": "spa_Latn",
    "pt": "por_Latn",
    "it": "ita_Latn",
    "ru": "rus_Cyrl",
    "zh": "zho_Hans",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "tr": "tur_Latn",
    "vi": "vie_Latn",
    "th": "tha_Thai",
    "nl": "nld_Latn",
    "pl": "pol_Latn",
    "uk": "ukr_Cyrl",
    "ne": "npi_Deva",
    "si": "sin_Sinh",
    "my": "mya_Mymr",
    "km": "khm_Khmr",
    "sw": "swh_Latn",
}

# =========================
# VOICE ACTIVITY DETECTION
# =========================

def check_speech_activity(audio_path: str, vad_threshold: float = VAD_THRESHOLD) -> Tuple[bool, float]:
    """Check if audio contains actual speech using Silero VAD."""
    try:
        audio_data, sample_rate = sf.read(audio_path)

        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)

        if sample_rate != 16000:
            import scipy.signal as signal
            audio_data = signal.resample(
                audio_data, int(len(audio_data) * 16000 / sample_rate)
            )
            sample_rate = 16000

        audio_tensor = torch.FloatTensor(audio_data)

        speech_timestamps = get_speech_timestamps(
            audio_tensor,
            vad_model,
            sampling_rate=sample_rate,
            threshold=vad_threshold,
            min_speech_duration_ms=int(MIN_SPEECH_DURATION * 1000),
            return_seconds=True
        )

        total_duration = len(audio_data) / sample_rate
        speech_duration = sum(ts['end'] - ts['start'] for ts in speech_timestamps)
        speech_ratio = speech_duration / total_duration if total_duration > 0 else 0
        has_speech = len(speech_timestamps) > 0 and speech_duration >= MIN_SPEECH_DURATION

        print(f"[VAD] Speech: {has_speech}, {speech_duration:.2f}s / {total_duration:.2f}s ({speech_ratio*100:.1f}%) [thr={vad_threshold}]")
        return has_speech, speech_ratio

    except Exception as e:
        print(f"[VAD] Error: {e} - assuming speech present")
        return True, 1.0


# =========================
# NLLB TRANSLATION
# =========================

def translate_text(text: str, source_nllb: str, target_nllb: str) -> str:
    """
    Translate text using NLLB-200 distilled 600M.
    Returns translated string, or empty string on failure.
    Works well for low-resource languages.
    """
    if not NLLB_AVAILABLE:
        return ""
    if not text or not source_nllb or not target_nllb:
        return ""
    if source_nllb == target_nllb:
        return text

    try:
        with nllb_lock:
            t0 = time.time()

            # Set source language on the tokenizer
            nllb_tokenizer.src_lang = source_nllb

            inputs = nllb_tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(device)

            # Force the model to start output with the target language token
            target_lang_id = nllb_tokenizer.lang_code_to_id[target_nllb]

            with torch.no_grad():
                translated_tokens = nllb_model_obj.generate(
                    **inputs,
                    forced_bos_token_id=target_lang_id,
                    max_length=512,
                    num_beams=4,
                    early_stopping=True,
                    no_repeat_ngram_size=3,
                )

            translated = nllb_tokenizer.batch_decode(
                translated_tokens, skip_special_tokens=True
            )[0]

            elapsed = time.time() - t0
            print(f"[NLLB] {source_nllb} -> {target_nllb} in {elapsed:.2f}s: {translated[:80]}")
            return translated

    except Exception as e:
        print(f"[NLLB] Translation error: {e}")
        return ""


# =========================
# SCRIPT VALIDATION
# =========================

def is_target_script(text: str, language_mode: str) -> bool:
    """
    Check if transcription contains expected script characters.
    Returns False if the user selected a non-Latin language but
    Whisper output is entirely Latin (likely a hallucination).
    """
    if not text.strip():
        return True

    latin_chars = sum(1 for c in text if 'a' <= c.lower() <= 'z')
    total_alpha = sum(1 for c in text if c.isalpha())
    if total_alpha == 0:
        return True

    if language_mode == "ml":
        # Malayalam Unicode range: U+0D00 - U+0D7F
        target_chars = sum(1 for c in text if '\u0D00' <= c <= '\u0D7F')
        if target_chars == 0 and latin_chars > 0:
            print(f"[SCRIPT] Malayalam mode but no Malayalam chars found — suppressing: {text[:60]}")
            return False
    elif language_mode == "hi":
        # Devanagari: U+0900 - U+097F
        target_chars = sum(1 for c in text if '\u0900' <= c <= '\u097F')
        if target_chars == 0 and latin_chars > 0:
            print(f"[SCRIPT] Hindi mode but no Devanagari chars found — suppressing: {text[:60]}")
            return False
    elif language_mode == "ta":
        target_chars = sum(1 for c in text if '\u0B80' <= c <= '\u0BFF')
        if target_chars == 0 and latin_chars > 0:
            print(f"[SCRIPT] Tamil mode but no Tamil chars found — suppressing: {text[:60]}")
            return False
    elif language_mode == "te":
        target_chars = sum(1 for c in text if '\u0C00' <= c <= '\u0C7F')
        if target_chars == 0 and latin_chars > 0:
            print(f"[SCRIPT] Telugu mode but no Telugu chars found — suppressing: {text[:60]}")
            return False
    elif language_mode == "kn":
        target_chars = sum(1 for c in text if '\u0C80' <= c <= '\u0CFF')
        if target_chars == 0 and latin_chars > 0:
            print(f"[SCRIPT] Kannada mode but no Kannada chars found — suppressing: {text[:60]}")
            return False
    # Mixed modes (ml-en, hi-en) allow both scripts
    return True


# =========================
# HALLUCINATION FILTER
# =========================

import re

HALLUCINATION_PATTERNS = [
    r"thank you for watching",
    r"thanks for watching",
    r"subscribe to",
    r"like and subscribe",
    r"see you next time",
    r"see you in the next",
    r"music playing",
    r"\[music\]",
    r"\[applause\]",
    r"\[laughter\]",
    r"please subscribe",
    r"don't forget to",
    r"if you enjoyed",
    r"leave a comment",
    r"share this video",
    r"click the bell",
    r"notification",
    r"my video",
    r"this video",
    r"our channel",
    r"my channel",
    # Common hallucinations when Whisper mishears Indian languages
    r"is anything to say to me",
    r"is there anything",
    r"i'm talking about the dirty",
    r"dirty stuff",
    r"what are you doing",
    r"what do you think",
    r"i don't understand",
    r"can you hear me",
    r"are you listening",
    r"good morning",
    r"good evening",
    r"good night",
    r"ladies and gentlemen",
    r"welcome to",
    r"let me tell you",
    r"as i was saying",
    r"you know what",
    r"i want to tell you",
]

# Malayalam-specific hallucination patterns
# Whisper generates these common phrases when it can't decode Malayalam properly
MALAYALAM_HALLUCINATION_PATTERNS = [
    "ഗാനമാണ്",          # "is a song" — common hallucination
    "മലയാള ഗാനം",       # "Malayalam song"
    "സിനിമ",            # "cinema"
    "സിനിമയിലെ",        # "in cinema"
    "സൂക്ഷിക്കാം",       # "let's keep" — nonsense filler
    "സിപ്പോക്ക്",        # nonsense word
    "പാകമാകന്ന",        # hallucinated word
    "ദൈവമേ",            # filler exclamation
    "മനോഹരമായ ഒരു മലയാള",  # "beautiful Malayalam" — filler
    "ഈ വീഡിയോ",          # "this video"
    "സബ്സ്ക്രൈബ്",       # "subscribe"
    "ലൈക്ക്",            # "like"
    "ചാനൽ",             # "channel"
    "ആദ്യമായി",          # filler
]

STANDALONE_HALLUCINATIONS = [
    "thank you", "thanks", "thank you.", "thanks.",
    "you", "you.", "okay", "okay.", "i don't know",
    "i don't know.", "but i don't know", "i know", "i know.",
    "yes", "yes.", "no", "no.", "hmm", "hmm.", "uh", "um",
    "ah", "oh", "so", "well", "right", "okay then",
    "alright", "sure", "yeah", "yep", "nope", "bye",
    "hi", "hello", "hey", "hee", "hee-",
    # Short filler that Whisper outputs for non-English audio
    "mm", "mhm", "uh huh", "huh", "ha", "eh",
    "la la la", "na na na",
]


def filter_hallucinations(text: str, language_mode: str = "") -> str:
    """Filter common Whisper hallucinations."""
    if not text:
        return ""

    text = text.strip()
    text_lower = text.lower()

    for pattern in HALLUCINATION_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            print(f"[FILTER] Blocked hallucination pattern: '{pattern}'")
            return ""

    # Malayalam-specific hallucination check
    if language_mode in ("ml", "ml-en", "ml-roman", "ml-via-en"):
        for pattern in MALAYALAM_HALLUCINATION_PATTERNS:
            if pattern in text:
                print(f"[FILTER-ML] Blocked Malayalam hallucination: '{pattern}' in '{text[:60]}'")
                return ""

    text_clean = text_lower.strip().rstrip('.').strip()
    for phrase in STANDALONE_HALLUCINATIONS:
        if text_clean == phrase.lower().strip().rstrip('.').strip():
            print(f"[FILTER] Blocked standalone hallucination: '{text}'")
            return ""

    words = text.split()
    if len(words) >= 4:
        for pattern_len in range(1, 5):
            if len(words) >= pattern_len * 3:
                pattern_str = ' '.join(words[:pattern_len])
                if text.count(pattern_str) >= 3:
                    print(f"[FILTER] Blocked repetition loop: '{pattern_str}'")
                    return ""

    if len(words) >= 8:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.3:
            print(f"[FILTER] Blocked low-diversity text ({unique_ratio*100:.0f}% unique)")
            return ""

    if "और" in text and text.count("और") >= 4:
        print(f"[FILTER] Blocked Hindi repetition loop")
        return ""

    # Malayalam repetition: check for repeated Malayalam word-groups
    if language_mode == "ml":
        ml_words = [w for w in text.split() if any('\u0D00' <= c <= '\u0D7F' for c in w)]
        if len(ml_words) >= 6:
            unique_ml = len(set(ml_words))
            if unique_ml / len(ml_words) < 0.4:
                print(f"[FILTER-ML] Blocked low-diversity Malayalam ({unique_ml}/{len(ml_words)} unique)")
                return ""

    # Unicode-aware length check (Malayalam chars are multi-byte)
    if len(text.strip()) < 2:
        return ""

    return text


# =========================
# MIXED-LANGUAGE POST-PROCESSING
# =========================

def post_process_mixed_text(text: str) -> str:
    """
    Clean up mixed Malayalam+English Whisper output.
    Removes common artifacts: repeated phrases, orphaned punctuation,
    Whisper noise words, etc.
    """
    if not text:
        return ""

    text = text.strip()

    # Remove repeated trailing/leading punctuation artifacts
    text = re.sub(r'^[\s.,!?;:]+', '', text)
    text = re.sub(r'[\s.,!?;:]+$', '', text)

    # Remove Whisper noise tokens that slip through in mixed mode
    noise_tokens = [
        '...', '…', '♪', '♫', '🎵', '🎶',
        '( )', '[ ]', '(( ))', '[[ ]]',
    ]
    for token in noise_tokens:
        text = text.replace(token, '')

    # Deduplicate: if the same phrase repeats 2+ times back-to-back, keep once
    # Works for both Malayalam and English words
    words = text.split()
    if len(words) >= 4:
        cleaned = []
        i = 0
        while i < len(words):
            # Check for 1-3 word repeat patterns
            found_repeat = False
            for plen in range(1, min(4, len(words) - i)):
                pattern = words[i:i+plen]
                pat_str = ' '.join(pattern)
                # Count consecutive repeats
                repeats = 1
                j = i + plen
                while j + plen <= len(words) and words[j:j+plen] == pattern:
                    repeats += 1
                    j += plen
                if repeats >= 2:
                    # Keep pattern once, skip duplicates
                    cleaned.extend(pattern)
                    i = j
                    found_repeat = True
                    print(f"[POST-PROC] Deduplicated '{pat_str}' (repeated {repeats}x)")
                    break
            if not found_repeat:
                cleaned.append(words[i])
                i += 1
        text = ' '.join(cleaned)

    # Clean up multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def translate_mixed_text(text: str, target_nllb: str) -> str:
    """
    Smart translation for mixed Malayalam+English text.
    Splits text into Malayalam and English chunks,
    translates only Malayalam portions via NLLB,
    preserves English portions as-is.
    """
    if not text or not target_nllb or not NLLB_AVAILABLE:
        return ""

    # If target is English, translate Malayalam portions to English
    if target_nllb == "eng_Latn":
        chunks = []
        current_chunk = []
        current_is_ml = None

        for word in text.split():
            # Check if word contains Malayalam script
            has_ml = any('\u0D00' <= c <= '\u0D7F' for c in word)
            if current_is_ml is None:
                current_is_ml = has_ml
            if has_ml != current_is_ml:
                chunks.append((' '.join(current_chunk), current_is_ml))
                current_chunk = [word]
                current_is_ml = has_ml
            else:
                current_chunk.append(word)
        if current_chunk:
            chunks.append((' '.join(current_chunk), current_is_ml))

        # Translate only Malayalam chunks
        result_parts = []
        for chunk_text, is_ml in chunks:
            if is_ml and chunk_text.strip():
                translated = translate_text(chunk_text, "mal_Mlym", "eng_Latn")
                result_parts.append(translated if translated else chunk_text)
            else:
                result_parts.append(chunk_text)

        return ' '.join(result_parts).strip()
    else:
        # For non-English targets, translate the whole thing
        # First translate Malayalam parts to English, then English to target
        # (NLLB handles eng -> target best)
        english_version = translate_mixed_text(text, "eng_Latn")
        if english_version and target_nllb != "eng_Latn":
            return translate_text(english_version, "eng_Latn", target_nllb)
        return english_version


# =========================
# ENDPOINTS
# =========================
this is the sever code for whisper

@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language_mode: str = Form(default="ml-en"),
    target_language: str = Form(default="")    # NLLB target code, e.g. "eng_Latn"
):
    """
    Speech-to-Text with optional NLLB translation.

    Steps:
    1. VAD - check for speech
    2. Whisper transcription
    3. Hallucination filter
    4. Optional NLLB translation if target_language is provided

    Returns: { text, translated_text, detected_language, ... }
    """
    start_time = time.time()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        # Step 1: VAD (stricter threshold for pure Indic languages)
        is_pure_indic_mode = language_mode in ("ml", "hi", "ta", "te", "kn")
        vad_thr = VAD_THRESHOLD_INDIC if is_pure_indic_mode else VAD_THRESHOLD
        has_speech, speech_ratio = check_speech_activity(tmp_path, vad_threshold=vad_thr)

        if not has_speech:
            return {
                "text": "",
                "translated_text": "",
                "language_mode": language_mode,
                "detected_language": "none",
                "processing_time": round(time.time() - start_time, 3),
                "translation_time": 0,
                "success": True,
                "vad_speech_ratio": round(speech_ratio, 2),
                "note": "No speech detected"
            }

        # Step 2: Whisper Transcription
        config = LANGUAGE_CONFIG.get(language_mode, LANGUAGE_CONFIG["ml-en"])
        whisper_lang = config["whisper_lang"]
        prompt = config["prompt"]

        print(f"\n{'='*50}")
        print(f"Mode: {language_mode} | Lang: {whisper_lang or 'auto'}")
        if target_language:
            print(f"Translate to NLLB: {target_language}")

        # Use relaxed parameters for low-resource / non-Latin languages
        is_low_resource = language_mode in ("ml", "ml-en", "ml-roman", "hi", "hi-en", "ta", "te", "kn")
        is_pure_indic = language_mode in ("ml", "hi", "ta", "te", "kn")
        use_translate = config.get("use_translate", False)

        # Temperature fallback: Whisper retries with higher temperatures
        # when decoding quality is poor. Critical for low-resource languages
        # where greedy decoding (0.0) often falls back to English.
        if is_low_resource:
            temperature = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
        else:
            temperature = 0.0

        # For Manglish mode: task="translate" tells Whisper to output English
        # text from Malayalam audio — this is far more reliable than transcribe
        whisper_task = "translate" if use_translate else "transcribe"
        print(f"Whisper task: {whisper_task}")

        result = model.transcribe(
            tmp_path,
            language=whisper_lang,
            task=whisper_task,
            temperature=temperature,
            beam_size=5 if is_low_resource else 1,
            best_of=5 if is_low_resource else 1,
            fp16=(device == "cuda"),
            condition_on_previous_text=False,
            initial_prompt=prompt,
            verbose=False,
            without_timestamps=True,
            compression_ratio_threshold=2.4 if is_pure_indic else (2.8 if is_low_resource else 1.8),
            logprob_threshold=-0.7 if is_pure_indic else (-1.0 if is_low_resource else -0.3),
            no_speech_threshold=0.5 if is_pure_indic else (0.6 if is_low_resource else 0.4)
        )

        raw_text = result.get("text", "").strip()
        detected_lang = result.get("language", whisper_lang or "unknown")

        # Step 2b: Segment-level confidence filtering
        # Discard individual segments with very low avg_logprob
        # Pure Indic: strict thresholds | Mixed (ml-en): relaxed thresholds
        segments = result.get("segments", [])
        is_mixed_mode = language_mode in ("ml-en", "hi-en")
        if segments and (is_pure_indic or is_mixed_mode):
            # Relaxed thresholds for mixed mode (ml-en works better, don't over-filter)
            lp_threshold = -0.8 if is_pure_indic else -1.0
            cr_threshold = 2.4 if is_pure_indic else 2.8
            good_segments = []
            for seg in segments:
                avg_lp = seg.get("avg_logprob", 0)
                seg_cr = seg.get("compression_ratio", 1.0)
                seg_text = seg.get("text", "").strip()
                no_speech = seg.get("no_speech_prob", 0)
                if avg_lp < lp_threshold:
                    print(f"[SEG-FILTER] Dropping low-confidence segment (avg_logprob={avg_lp:.2f}): {seg_text[:50]}")
                elif seg_cr > cr_threshold:
                    print(f"[SEG-FILTER] Dropping high-compression segment (cr={seg_cr:.2f}): {seg_text[:50]}")
                elif no_speech > 0.8:
                    print(f"[SEG-FILTER] Dropping high no_speech segment (prob={no_speech:.2f}): {seg_text[:50]}")
                else:
                    good_segments.append(seg_text)
            text = " ".join(good_segments).strip()
            if text != raw_text:
                print(f"[SEG-FILTER] Filtered: '{raw_text[:60]}' → '{text[:60]}'")
        else:
            text = raw_text

        # Step 2c: Post-process mixed language output
        if text and is_mixed_mode:
            text = post_process_mixed_text(text)

        # Step 3: Hallucination filter
        text = filter_hallucinations(text, language_mode=language_mode)

        # Step 3b: Script validation — for pure Indic modes (not mixed),
        # if the output is entirely Latin, it's a Whisper hallucination.
        # Retry once with a warmer temperature before giving up.
        if text and is_pure_indic and not is_target_script(text, language_mode):
            print(f"[RETRY] Got Latin text in {language_mode} mode, retrying with temperature=0.4...")
            retry_result = model.transcribe(
                tmp_path,
                language=whisper_lang,
                task="transcribe",
                temperature=0.4,
                beam_size=5,
                best_of=5,
                fp16=(device == "cuda"),
                condition_on_previous_text=False,
                initial_prompt=prompt,
                verbose=False,
                without_timestamps=True,
                compression_ratio_threshold=3.0,
                logprob_threshold=-1.5,
                no_speech_threshold=0.7
            )
            retry_text = retry_result.get("text", "").strip()
            retry_text = filter_hallucinations(retry_text, language_mode=language_mode)
            if retry_text and is_target_script(retry_text, language_mode):
                text = retry_text
                detected_lang = retry_result.get("language", whisper_lang or "unknown")
                print(f"[RETRY] Success: {text[:60]}")
            else:
                # Both attempts failed — suppress entirely
                print(f"[RETRY] Still no {language_mode} script, suppressing.")
                text = ""
        elif text and language_mode in ("ml-en", "hi-en"):
            # For mixed modes, don't suppress — allow both scripts
            pass

        # Step 4: NLLB Translation (with smart mixed-language handling)
        translated_text = ""
        translation_time = 0.0

        if text and use_translate:
            # Whisper gave English text (task="translate").
            # Convert internally so user sees source language + target language.
            english_text = text
            source_nllb = config.get("nllb_code", "")
            t_start = time.time()

            # Original Speech panel: English → source language (e.g., Malayalam)
            if source_nllb and source_nllb != "eng_Latn":
                source_text = translate_text(english_text, "eng_Latn", source_nllb)
                text = source_text if source_text else english_text
            # else: source is English, keep as-is

            # Translation panel: English → user's target language
            if target_language and target_language != "eng_Latn":
                translated_text = translate_text(english_text, "eng_Latn", target_language)
            elif target_language == "eng_Latn":
                translated_text = english_text

            translation_time = round(time.time() - t_start, 3)
            print(f"[TRANSLATE] EN: {english_text[:50]} → src({source_nllb}): {text[:50]}")
            if translated_text:
                print(f"[TRANSLATE] → tgt({target_language}): {translated_text[:50]}")
        elif text and target_language:
            t_start = time.time()
            if is_mixed_mode:
                # Smart mixed-language translation:
                # Split into ML/EN chunks, translate only the ML parts
                translated_text = translate_mixed_text(text, target_language)
                print(f"[SMART-TRANSLATE] Mixed '{text[:50]}' → '{translated_text[:50]}'")
            else:
                # Standard single-language translation
                source_nllb = config.get("nllb_code") or WHISPER_TO_NLLB.get(detected_lang, "")
                if source_nllb:
                    translated_text = translate_text(text, source_nllb, target_language)
            translation_time = round(time.time() - t_start, 3)

        processing_time = time.time() - start_time

        print(f"Detected: {detected_lang} | Total: {processing_time:.2f}s")
        print(f"Original : {text[:100]}")
        if translated_text:
            print(f"Translated: {translated_text[:100]}")

        return {
            "text": text,
            "translated_text": translated_text,
            "language_mode": language_mode,
            "detected_language": detected_lang,
            "processing_time": round(processing_time, 3),
            "translation_time": translation_time,
            "success": True,
            "vad_speech_ratio": round(speech_ratio, 2),
            "nllb_available": NLLB_AVAILABLE
        }

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {
            "text": "",
            "translated_text": "",
            "language_mode": language_mode,
            "detected_language": "error",
            "processing_time": round(time.time() - start_time, 3),
            "translation_time": 0,
            "success": False,
            "error": str(e)
        }
    finally:
        try:
            os.remove(tmp_path)
        except:
            pass


@app.post("/translate")
async def translate_endpoint(
    text: str = Form(...),
    source_language: str = Form(...),
    target_language: str = Form(...)
):
    """Standalone translation endpoint (text-only, no audio)."""
    if not NLLB_AVAILABLE:
        return {"translated_text": "", "success": False, "error": "NLLB model not loaded"}

    t_start = time.time()
    translated = translate_text(text, source_language, target_language)
    return {
        "translated_text": translated,
        "source_language": source_language,
        "target_language": target_language,
        "translation_time": round(time.time() - t_start, 3),
        "success": True
    }


@app.get("/")
async def root():
    return {
        "status": "running",
        "whisper_model": MODEL_NAME,
        "device": device,
        "nllb_available": NLLB_AVAILABLE,
        "nllb_model": NLLB_MODEL_NAME if NLLB_AVAILABLE else None,
        "supported_modes": list(LANGUAGE_CONFIG.keys()),
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "whisper_loaded": True,
        "nllb_loaded": NLLB_AVAILABLE,
        "device": device
    }

......this is the docker for whisper-----
    # Whisper STT + NLLB Translation Server
# Use PyTorch 2.4 (required by latest Transformers)
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime


ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
  apt-get install -y ffmpeg git && \
  rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

# Fix huggingface compatibility
RUN pip install --no-cache-dir \
  "filelock>=3.13.1" \
  "huggingface_hub>=0.23.0"

# Whisper
RUN pip install --no-cache-dir openai-whisper

# API
RUN pip install --no-cache-dir \
  fastapi \
  "uvicorn[standard]" \
  python-multipart

# Audio
RUN pip install --no-cache-dir \
  soundfile \
  scipy \
  numpy

# Stable transformers version (VERY IMPORTANT)
RUN pip install --no-cache-dir \
  "transformers==4.40.2" \
  sentencepiece \
  sacremoses \
  protobuf

WORKDIR /app

COPY server.py .

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]


--------------------------------------------------------------------


thi was my planed architecture


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


prompt


I am building BridgeMeet — a real-time multilingual meeting transcription and translation system.
Architecture
User speaks
  │
  ├── AGORA WebRTC SDK ──► All users HEAR instantly (~50-100ms)
  │
  └── Audio Buffer (4s chunks)
          │
          └── HTTP POST to HPC GPU Server
                  │
                  ├── Speech-to-Text
                  │     ├── English audio     → Whisper STT
                  │     └── Indian languages  → IndicConformer (ai4bharat/indic-conformer-600m-multilingual)
                  │
                  └── Translation
                        └── IndicTrans2 (ai4bharat/indictrans2-*-1B)
                              ├── Indian → English   : indictrans2-indic-en-1B
                              ├── English → Indian   : indictrans2-en-indic-1B
                              └── Indian → Indian    : indictrans2-indic-indic-1B

Users SEE translated text in their language after ~4-5s
Key Rules (ALWAYS follow these)

Whisper is ONLY used for English speech-to-text. Never suggest Whisper for Indian language STT.
IndicConformer handles all Indian language STT (Malayalam, Hindi, Tamil, Telugu, Kannada, Bengali, Gujarati, Marathi, Punjabi, Urdu, and 12 more).

Loaded with AutoModel.from_pretrained(..., trust_remote_code=True) — NOT AutoProcessor or AutoModelForCTC
Inference: model(wav_tensor, lang_code, "ctc")
Input tensor: [1, num_samples] float32 at 16kHz


IndicTrans2 handles ALL translation — never suggest NLLB for this project.

NLLB was tried and gave poor quality for Indian languages
IndicTrans2 is purpose-built for Indian languages with 230M training pairs
It is made by the same team (AI4Bharat, IIT Madras) as IndicConformer


Language codes are short codes (e.g. ml, hi, ta, en) — NOT NLLB-style codes like eng_Latn.

docker run \
  --gpus '"device=2"' \
  -p 9105:8000 \
  --env-file ~/indic.env \
  -v ~/hf_cache:/root/.cache/huggingface \
  -d --restart=unless-stopped \
  --name indicconformer \
  indicconformer-serve
