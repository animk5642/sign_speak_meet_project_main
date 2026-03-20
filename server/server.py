#!/usr/bin/env python3
"""
Combined Translation Server
- Whisper Large-v3 (for International -> English)
- IndicConformer (for Indian -> Indian text)
- IndicTrans2 (for Translation involving Indian languages)
"""

import os
import time
import tempfile
import shutil
import threading
import traceback
import logging
from typing import Tuple, Optional, Dict, Any

import torch
import torchaudio
import soundfile as sf
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import (
    AutoModel,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    logging as hf_logging,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

hf_logging.set_verbosity_error()

app = FastAPI(title="Universal Translator (Whisper + Indic)", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================================================================
# CONFIGURATION
# ==============================================================================

HF_TOKEN = os.environ.get("HF_TOKEN", None)
API_KEY = os.environ.get("API_KEY", "")

# --- GPU Allocation ---
# If multiple GPUs are available, split the load.
# GPU 0: Whisper (International + English)
# GPU 1: IndicConformer + IndicTrans2 (Indian Languages)
if torch.cuda.device_count() > 1:
    DEVICE_WHISPER = "cuda:0"
    DEVICE_INDIC = "cuda:1"
    DEVICE_NLLB = "cuda:0" # Put NLLB with Whisper or distribute as needed
    logger.info(f"Multi-GPU Mode: Whisper/NLLB on {DEVICE_WHISPER}, Indic models on {DEVICE_INDIC}")
else:
    DEVICE_WHISPER = "cuda" if torch.cuda.is_available() else "cpu"
    DEVICE_INDIC = DEVICE_WHISPER
    DEVICE_NLLB = DEVICE_WHISPER
    logger.info(f"Single Device Mode: All models on {DEVICE_WHISPER}")

# --- Model Names ---
# Indic STT
IC_MODEL_NAME = "ai4bharat/indic-conformer-600m-multilingual"

# Indic Translation
IT2_INDIC_EN    = "ai4bharat/indictrans2-indic-en-1B"
IT2_EN_INDIC    = "ai4bharat/indictrans2-en-indic-1B"
IT2_INDIC_INDIC = "ai4bharat/indictrans2-indic-indic-1B"

# Whisper
WHISPER_MODEL_NAME = "large-v3"

logger.info("=" * 60)
logger.info(f"Universal Translator Server starting")
logger.info(f"HF Token present: {HF_TOKEN is not None}")
logger.info("=" * 60)

# ==============================================================================
# WHISPER SETUP (International Languages)
# ==============================================================================
whisper_model = None
whisper_lock = threading.Lock()

try:
    import whisper
    logger.info(f"[WHISPER] Loading {WHISPER_MODEL_NAME}...")
    # Whisper loads entire model to GPU if device='cuda'
    # For concurrent usage with other large models, might need careful VRAM management
    whisper_model = whisper.load_model(WHISPER_MODEL_NAME, device=DEVICE_WHISPER)
    logger.info(f"[WHISPER] Loaded on {DEVICE_WHISPER} ✓")
except Exception as e:
    logger.error(f"[WHISPER] Failed to load: {e}")
    traceback.print_exc()

# ==============================================================================
# INDIC CONFORMER SETUP (Indian Languages STT)
# ==============================================================================
ic_model = None
ic_lock = threading.Lock()

try:
    logger.info(f"[INDIC-STT] Loading {IC_MODEL_NAME}...")
    ic_model = AutoModel.from_pretrained(
        IC_MODEL_NAME, token=HF_TOKEN, trust_remote_code=True
    ).to(DEVICE_INDIC)
    ic_model.eval()
    logger.info(f"[INDIC-STT] Loaded on {DEVICE_INDIC} ✓")
except Exception as e:
    logger.error(f"[INDIC-STT] Failed to load: {e}")
    traceback.print_exc()

# ==============================================================================
# INDIC TRANS 2 SETUP (Translation)
# ==============================================================================
INDIC_PROCESSOR_AVAILABLE = False
try:
    try:
        from IndicTransToolkit import IndicProcessor
    except ImportError:
        from IndicTransToolkit.IndicTransToolkit import IndicProcessor
    INDIC_PROCESSOR_AVAILABLE = True
    logger.info("[INDIC-TRANS] IndicTransToolkit loaded ✓")
except Exception as e:
    logger.error(f"[INDIC-TRANS] IndicTransToolkit FAILED: {e}")

it2_models = {}
it2_tokenizers = {}
it2_lock = threading.Lock()

def load_it2_model(key: str, model_name: str):
    """Lazy load or pre-load IndicTrans2 models."""
    if key in it2_models:
        return
    try:
        logger.info(f"[INDIC-TRANS] Loading {key} ({model_name})...")
        tok = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, token=HF_TOKEN
        )
        mdl = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            token=HF_TOKEN,
            torch_dtype=torch.float16 if "cuda" in DEVICE_INDIC else torch.float32,
        ).to(DEVICE_INDIC)
        mdl.eval()
        it2_tokenizers[key] = tok
        it2_models[key] = mdl
        logger.info(f"[INDIC-TRANS] {key} Loaded ✓")
    except Exception as e:
        logger.error(f"[INDIC-TRANS] {key} Failed: {e}")

if INDIC_PROCESSOR_AVAILABLE:
    # Pre-load En-Indic and Indic-En as they are most critical
    load_it2_model("indic-en",    IT2_INDIC_EN)
    load_it2_model("en-indic",    IT2_EN_INDIC)
    load_it2_model("indic-indic", IT2_INDIC_INDIC)

# ==============================================================================
# NLLB SETUP (Fallback for International Translation)
# ==============================================================================
NLLB_MODEL_NAME = "facebook/nllb-200-distilled-600M"
nllb_model = None
nllb_tokenizer = None
nllb_lock = threading.Lock()

try:
    logger.info(f"[NLLB] Loading {NLLB_MODEL_NAME}...")
    nllb_tokenizer = AutoTokenizer.from_pretrained(NLLB_MODEL_NAME)
    nllb_model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL_NAME).to(DEVICE_NLLB)
    nllb_model.eval()
    logger.info(f"[NLLB] Loaded on {DEVICE_NLLB} ✓")
except Exception as e:
    logger.error(f"[NLLB] Failed to load: {e}")

# ==============================================================================
# VAD SETUP (Optional but good for suppressing silence)
# ==============================================================================
vad_model = None
get_speech_timestamps = None
try:
    logger.info("[VAD] Loading Silero VAD...")
    vad_model, vad_utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False,
        onnx=False
    )
    (get_speech_timestamps, _, _, _, _) = vad_utils
    vad_model = vad_model.cpu() # Run VAD on CPU to save GPU mem
    logger.info("[VAD] Loaded ✓")
except Exception as e:
    logger.warning(f"[VAD] Failed to load: {e}")

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

# -- Mappings --
# Indian languages supported by IndicConformer/IndicTrans2
INDIC_LANGS = {
    "as", "bn", "brx", "doi", "gu", "hi", "kn", "ks", "kok", "mai",
    "ml", "mni", "mr", "ne", "or", "pa", "sa", "sat", "sd", "ta", "te", "ur"
}

# Mapping ISO codes to IndicTrans2 codes
IT2_CODE_MAP = {
    "as":  "asm_Beng", "bn":  "ben_Beng", "brx": "brx_Deva",
    "doi": "doi_Deva", "gu":  "guj_Gujr", "hi":  "hin_Deva",
    "kn":  "kan_Knda", "ks":  "kas_Arab", "kok": "kok_Deva",
    "mai": "mai_Deva", "ml":  "mal_Mlym", "mni": "mni_Mtei",
    "mr":  "mar_Deva", "ne":  "npi_Deva", "or":  "ory_Orya",
    "pa":  "pan_Guru", "sa":  "san_Deva", "sat": "sat_Olck",
    "sd":  "snd_Arab", "ta":  "tam_Taml", "te":  "tel_Telu",
    "ur":  "urd_Arab", "en":  "eng_Latn",
    # French/German/Japan etc mapping for IT2 input/output not supported directly by IT2 models usually
    # IT2 is specifically for Indic <-> Indic or Indic <-> English.
}

# ==============================================================================
# HALLUCINATION FILTERING
# ==============================================================================

# Phrases that are ALWAYS hallucinations regardless of audio energy.
# These are YouTube/podcast boilerplate that real speech never matches.
_ALWAYS_HALLUCINATION_PHRASES = [
    "please subscribe", "like and subscribe",
    "don't forget to subscribe", "hit the bell",
    "see you in the next video", "see you next time",
    "चैनल को सब्सक्राइब",
    "谢谢观看", "ご視聴ありがとう",
]

# Phrases that are hallucinations ONLY when audio energy is below the threshold.
# If the audio has real signal strength the user actually said these words.
_ENERGY_DEPENDENT_HALLUCINATION_PHRASES = [
    "thanks for watching", "thank you for watching",
    "thank you for listening", "thanks for listening",
    "thank you",  "thanks",
    "देखने के लिए धन्यवाद",   # Hindi "thank you for watching"
    "धन्यवाद",                # Hindi "thank you"
    "شكرا للمشاهدة",          # Arabic "thanks for watching"
    "شكرا",                   # Arabic "thank you"
]

# RMS below this → audio is near-silent → "thank you" is almost certainly a
# Whisper hallucination.  Tune empirically; 0.01 works well for float32 PCM.
_HALLUCINATION_ENERGY_THRESHOLD = 0.01

def get_audio_rms(file_path: str) -> float:
    """
    Return the RMS energy of an audio file as a float32 value.
    Returns 1.0 (non-silent safe default) on any read failure so we
    never accidentally over-filter real audio.
    """
    try:
        data, _ = sf.read(file_path, dtype="float32", always_2d=True)
        mono = data.mean(axis=1)
        rms = float(np.sqrt(np.mean(mono ** 2)))
        return rms
    except Exception as e:
        logger.warning(f"[RMS] Could not compute RMS for {file_path}: {e}")
        return 1.0  # Safe default — don't over-filter

def _is_hallucination(text: str, audio_path: Optional[str] = None) -> bool:
    """
    Detect Whisper hallucinations.

    - Empty / whitespace-only output → always hallucination.
    - Phrases in _ALWAYS_HALLUCINATION_PHRASES → always hallucination.
    - Phrases in _ENERGY_DEPENDENT_HALLUCINATION_PHRASES → hallucination ONLY
      when the audio RMS is below _HALLUCINATION_ENERGY_THRESHOLD (i.e. the
      audio clip is near-silent).  If the user actually said "thank you" at
      normal volume the RMS will be above the threshold and the text passes.
    """
    if not text or not text.strip():
        return True

    low = text.strip().lower()

    # 1. Always-filter list (subscribe spam, etc.)
    if any(p in low for p in _ALWAYS_HALLUCINATION_PHRASES):
        return True

    # 2. Energy-dependent list ("thank you" and similar)
    if any(p in low for p in _ENERGY_DEPENDENT_HALLUCINATION_PHRASES):
        # We need the audio energy to decide
        if audio_path:
            rms = get_audio_rms(audio_path)
            logger.info(f"[HALLUCINATION] Energy-dependent phrase detected. RMS={rms:.5f} threshold={_HALLUCINATION_ENERGY_THRESHOLD}")
            if rms < _HALLUCINATION_ENERGY_THRESHOLD:
                logger.info(f"[HALLUCINATION] Low-energy audio → filtering as hallucination: '{text}'")
                return True
            # High-energy audio → user genuinely said it
            logger.info(f"[HALLUCINATION] High-energy audio → keeping real speech: '{text}'")
            return False
        else:
            # No audio path provided — cannot measure energy, so err on the
            # side of keeping the text (don't over-filter).
            return False

    return False

# ==============================================================================

def get_audio_duration(file_path: str) -> float:
    try:
        import soundfile as sf
        f = sf.SoundFile(file_path)
        return len(f) / f.samplerate
    except:
        return 0.0

def load_audio_tensor(audio_path: str) -> torch.Tensor:
    """For IndicConformer"""
    wav, sr = torchaudio.load(audio_path)
    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(wav)
    return wav

def run_indic_trans(text: str, src_lang: str, tgt_lang: str) -> str:
    """Run IndicTrans2 translation."""
    if not text or not INDIC_PROCESSOR_AVAILABLE:
        return text # Return original if trans not available

    # Determine direction
    if src_lang == "en":
        direction = "en-indic"
    elif tgt_lang == "en":
        direction = "indic-en"
    else:
        direction = "indic-indic"

    if direction not in it2_models:
        return f"[Model {direction} missing]"

    # Get codes
    src_code = IT2_CODE_MAP.get(src_lang)
    tgt_code = IT2_CODE_MAP.get(tgt_lang)

    # If codes are not in IT2 map (e.g. 'fr'), we cannot use IT2.
    if not src_code or not tgt_code:
        return None # Return None to signal fallback to NLLB

    try:
        with it2_lock:
            tokenizer = it2_tokenizers[direction]
            model = it2_models[direction]
            ip = IndicProcessor(inference=True)

            batch = ip.preprocess_batch([text], src_lang=src_code, tgt_lang=tgt_code)
            inputs = tokenizer(
                batch, truncation=True, padding="longest", return_tensors="pt", return_attention_mask=True
            ).to(DEVICE_INDIC)

            with torch.no_grad():
                generated = model.generate(
                    **inputs, use_cache=True, min_length=0, max_length=512, num_beams=5, num_return_sequences=1
                )

            with tokenizer.as_target_tokenizer():
                decoded = tokenizer.batch_decode(
                    generated.detach().cpu().tolist(),
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )

            translations = ip.postprocess_batch(decoded, lang=tgt_code)
            return translations[0]
    except Exception as e:
        logger.error(f"IndicTrans Error: {e}")
        return f"[Translation Error]"

def run_nllb_trans(text: str, src_code: str, tgt_code: str) -> str:
    """Run NLLB translation."""
    if not nllb_model or not text:
        return text
    
    try:
        with nllb_lock:
            nllb_tokenizer.src_lang = src_code
            inputs = nllb_tokenizer(text, return_tensors="pt").to(DEVICE_NLLB)
            
            forced_bos_token_id = nllb_tokenizer.lang_code_to_id[tgt_code]
            
            with torch.no_grad():
                generated_tokens = nllb_model.generate(
                    **inputs, forced_bos_token_id=forced_bos_token_id, max_length=512
                )
                
            result = nllb_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
            return result

    except Exception as e:
        logger.error(f"NLLB Error: {e}")
        return f"[NLLB Error]"

def transcribe_indic_conformer(audio_path: str, lang: str) -> str:
    """Run IndicConformer STT."""
    if ic_model is None:
        raise RuntimeError("IndicConformer is not loaded")
    
    wav = load_audio_tensor(audio_path)
    with ic_lock:
        with torch.no_grad():
            # IndicConformer expects 'hi', 'ml' etc directly
            text = ic_model(wav, lang, "ctc")
    
    if isinstance(text, list):
        text = text[0]
    return str(text).strip()

def transcribe_whisper(audio_path: str, task: str = "transcribe", language: str = None) -> Tuple[str, str]:
    """
    Run Whisper. 
    If task="translate", Whisper translates X -> English.
    Returns (text, detected_language)
    """
    if whisper_model is None:
        raise RuntimeError("Whisper is not loaded")
    
    with whisper_lock:
        # whisper load_audio resamples to 16k automatically
        audio = whisper.load_audio(audio_path)
        
        # trim if too long? Whisper handles 30s chunks.
        
        opts = {
            "task": task,
            "beam_size": 5
        }
        if language:
            opts["language"] = language
            
        result = whisper_model.transcribe(audio, **opts)
        
    return result["text"].strip(), result.get("language", "en")

# ==============================================================================
# API ENDPOINTS
# ==============================================================================

@app.post("/translate")
async def translate_endpoint(
    text: str = Form(...),
    source_language: str = Form(...), # NLLB codes usually: 'eng_Latn', 'hin_Deva' etc OR short codes
    target_language: str = Form(...),
    x_api_key: str = Header(default="")
):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
        
    start_time = time.time()
    
    # Heuristic to map NLLB codes back to short codes for IndicTrans check
    # IndicTrans short codes: 'ml', 'hi', 'en'
    # NLLB codes: 'mal_Mlym', 'hin_Deva', 'eng_Latn'
    
    # Reverse map for checking (simple check)
    def to_short_code(nllb_code):
        for k, v in IT2_CODE_MAP.items():
            if v == nllb_code:
                return k
        return nllb_code # fallback to itself (e.g. 'fr' if sent as short code)

    src_short = to_short_code(source_language)
    tgt_short = to_short_code(target_language)
    
    translated = ""
    used_model = "none"

    # Strategy: Try IndicTrans2 first if supported, then NLLB
    # IT2 supports Indic-Indic, Indic-En, En-Indic
    # Supports codes in INDIC_LANGS + 'en'
    
    if (src_short in INDIC_LANGS or src_short == 'en') and \
       (tgt_short in INDIC_LANGS or tgt_short == 'en'):
        # IndicTrans2 candidate
        res = run_indic_trans(text, src_short, tgt_short)
        if res is not None:
             translated = res
             used_model = "indic_trans2"
    
    if not translated:
        # Fallback to NLLB
        # source_language/target_language must be NLLB codes (e.g. eng_Latn)
        # The client sends NLLB codes (see subtitle_client.js TARGET_LANGUAGES)
        translated = run_nllb_trans(text, source_language, target_language)
        used_model = "nllb"

    return {
        "success": True,
        "translated_text": translated,
        "used_model": used_model,
        "processing_time": round(time.time() - start_time, 3)
    }

@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language_mode: str = Form(default="auto"), # 'ml', 'hi', 'en', 'fr', 'auto'
    target_language: str = Form(default="en"), # 'en', 'ml', 'hi'
    x_api_key: str = Header(default="")
):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    start_time = time.time()
    temp_path = None
    
    try:
        # Save file
        suffix = os.path.splitext(file.filename or "audio.wav")[-1] or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            temp_path = tmp.name

        # --- PIPELINE LOGIC ---

        detected_lang = language_mode
        transcribed_text = ""
        translated_text = ""
        
        # Case 1: Source is Indian Language (explicit)
        if language_mode in INDIC_LANGS:
            # Step 1: STT using IndicConformer (Best for Indian langs)
            transcribed_text = transcribe_indic_conformer(temp_path, language_mode)
            detected_lang = language_mode

            # Pass audio_path so energy check applies for "thank you" variants
            if _is_hallucination(transcribed_text, audio_path=temp_path):
                logger.info(f"[HALLUCINATION] Filtered: '{transcribed_text}'")
                transcribed_text = ""
            
            # Step 2: Translation
            if target_language != language_mode and transcribed_text:
                # Use IndicTrans2
                res = run_indic_trans(transcribed_text, language_mode, target_language)
                if res:
                    translated_text = res
                else:
                    # Fallback NLLB (if valid codes provided, but here logic uses short codes)
                    pass 
            else:
                translated_text = transcribed_text

        # Case 2: Source is English or International or Auto
        else:
            # Use Whisper
            # If target is English, we can just use Whisper "translate" task (X -> En)
            
            task = "transcribe"
            if target_language == "en" and language_mode != "en":
                # Direct X -> En translation by Whisper (Best for Int -> En)
                task = "translate" 
            
            # Run Whisper
            w_lang = None if language_mode == "auto" else language_mode
            text, det_lang = transcribe_whisper(temp_path, task=task, language=w_lang)

            # Pass audio_path so energy check applies for "thank you" variants
            if _is_hallucination(text, audio_path=temp_path):
                logger.info(f"[HALLUCINATION] Filtered: '{text}'")
                text = ""

            transcribed_text = text
            if language_mode == "auto":
                detected_lang = det_lang

            # If task was "translate" (X->En), transcribed_text is already English
            current_text_lang = "en" if task == "translate" else detected_lang

            # Translation from Current -> Target
            if target_language in INDIC_LANGS:
                if current_text_lang == target_language:
                    translated_text = transcribed_text
                else:
                    # Try IndicTrans2 (best for En->Indic)
                    # We need short code for current (det_lang from whisper is usually short 'fr', 'en' etc)
                    res = run_indic_trans(transcribed_text, current_text_lang, target_language)
                    if res:
                        translated_text = res
                    else:
                        # Fallback to NLLB?
                        # NLLB needs 'eng_Latn' etc.
                        pass
            else:
                # Target is International (e.g. En). Whisper X->En handled it if target=en.
                if target_language == "en":
                    translated_text = transcribed_text
                else:
                    # En -> Fr etc? Not supported by IndicTrans. 
                    # Use NLLB if possible. But here we deal with short codes.
                    # Mapping short code to NLLB code is tricky without full map.
                    # Suggest user uses /translate endpoint for text-based fallback if needed, 
                    # but here we rely on what we have.
                    translated_text = transcribed_text 

        return {
            "success": True,
            "text": transcribed_text,
            "translated_text": translated_text,
            "detected_language": detected_lang,
            "processing_time": round(time.time() - start_time, 3)
        }

    except Exception as e:
        logger.error(f"Error processing request: {e}")
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "processing_time": round(time.time() - start_time, 3)
        }
    
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

@app.get("/health")
def health():
    return {
        "status": "online",
        "models": {
            "whisper": whisper_model is not None,
            "indic_conformer": ic_model is not None,
            "indic_trans": list(it2_models.keys())
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)