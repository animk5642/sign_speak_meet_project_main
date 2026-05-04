#!/usr/bin/env python3
"""
Combined Translation Server
- Whisper Large-v3 (for International -> English)
- IndicConformer (for Indian -> Indian text)
- IndicTrans2 (for Translation involving Indian languages)
latest server
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
hf_logging.set_verbosity_error()

app = FastAPI(title="Universal Translator (Whisper + Indic)", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

HF_TOKEN = os.environ.get("HF_TOKEN", None)
API_KEY = os.environ.get("API_KEY", "")

if torch.cuda.device_count() > 1:
    DEVICE_WHISPER = "cuda:0"
    DEVICE_INDIC = "cuda:1"
    DEVICE_NLLB = "cuda:0"
    logger.info(f"Multi-GPU Mode: Whisper/NLLB on {DEVICE_WHISPER}, Indic models on {DEVICE_INDIC}")
else:
    DEVICE_WHISPER = "cuda" if torch.cuda.is_available() else "cpu"
    DEVICE_INDIC = DEVICE_WHISPER
    DEVICE_NLLB = DEVICE_WHISPER
    logger.info(f"Single Device Mode: All models on {DEVICE_WHISPER}")

IC_MODEL_NAME   = "ai4bharat/indic-conformer-600m-multilingual"
IT2_INDIC_EN    = "ai4bharat/indictrans2-indic-en-1B"
IT2_EN_INDIC    = "ai4bharat/indictrans2-en-indic-1B"
IT2_INDIC_INDIC = "ai4bharat/indictrans2-indic-indic-1B"
WHISPER_MODEL_NAME = "large-v3"

logger.info("=" * 60)
logger.info("Universal Translator Server starting")
logger.info(f"HF Token present: {HF_TOKEN is not None}")
logger.info("=" * 60)

# Whisper
whisper_model = None
whisper_lock = threading.Lock()
try:
    import whisper
    logger.info(f"[WHISPER] Loading {WHISPER_MODEL_NAME}...")
    whisper_model = whisper.load_model(WHISPER_MODEL_NAME, device=DEVICE_WHISPER)
    logger.info(f"[WHISPER] Loaded on {DEVICE_WHISPER} ✓")
except Exception as e:
    logger.error(f"[WHISPER] Failed to load: {e}")
    traceback.print_exc()

# IndicConformer
ic_model = None
ic_lock = threading.Lock()
try:
    logger.info(f"[INDIC-STT] Loading {IC_MODEL_NAME}...")
    ic_model = AutoModel.from_pretrained(IC_MODEL_NAME, token=HF_TOKEN, trust_remote_code=True).to(DEVICE_INDIC)
    ic_model.eval()
    logger.info(f"[INDIC-STT] Loaded on {DEVICE_INDIC} ✓")
except Exception as e:
    logger.error(f"[INDIC-STT] Failed to load: {e}")
    traceback.print_exc()

# IndicTrans2
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
    if key in it2_models:
        return
    try:
        logger.info(f"[INDIC-TRANS] Loading {key} ({model_name})...")
        tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=HF_TOKEN)
        mdl = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, trust_remote_code=True, token=HF_TOKEN,
            torch_dtype=torch.float16 if "cuda" in DEVICE_INDIC else torch.float32,
        ).to(DEVICE_INDIC)
        mdl.eval()
        it2_tokenizers[key] = tok
        it2_models[key] = mdl
        logger.info(f"[INDIC-TRANS] {key} Loaded ✓")
    except Exception as e:
        logger.error(f"[INDIC-TRANS] {key} Failed: {e}")

if INDIC_PROCESSOR_AVAILABLE:
    load_it2_model("indic-en",    IT2_INDIC_EN)
    load_it2_model("en-indic",    IT2_EN_INDIC)
    load_it2_model("indic-indic", IT2_INDIC_INDIC)

# NLLB
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

# VAD
vad_model = None
get_speech_timestamps = None
try:
    logger.info("[VAD] Loading Silero VAD...")
    vad_model, vad_utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False, onnx=False)
    (get_speech_timestamps, _, _, _, _) = vad_utils
    vad_model = vad_model.cpu()
    logger.info("[VAD] Loaded ✓")
except Exception as e:
    logger.warning(f"[VAD] Failed to load: {e}")

INDIC_LANGS = {
    "as", "bn", "brx", "doi", "gu", "hi", "kn", "ks", "kok", "mai",
    "ml", "mni", "mr", "ne", "or", "pa", "sa", "sat", "sd", "ta", "te", "ur"
}

IT2_CODE_MAP = {
    "as": "asm_Beng", "bn": "ben_Beng", "brx": "brx_Deva",
    "doi": "doi_Deva", "gu": "guj_Gujr", "hi": "hin_Deva",
    "kn": "kan_Knda", "ks": "kas_Arab", "kok": "kok_Deva",
    "mai": "mai_Deva", "ml": "mal_Mlym", "mni": "mni_Mtei",
    "mr": "mar_Deva", "ne": "npi_Deva", "or": "ory_Orya",
    "pa": "pan_Guru", "sa": "san_Deva", "sat": "sat_Olck",
    "sd": "snd_Arab", "ta": "tam_Taml", "te": "tel_Telu",
    "ur": "urd_Arab", "en": "eng_Latn",
}

_ALWAYS_HALLUCINATION_PHRASES = [
    "please subscribe", "like and subscribe",
    "don't forget to subscribe", "hit the bell",
    "see you in the next video", "see you next time",
    "चैनल को सब्सक्राइब", "谢谢观看", "ご視聴ありがとう",
]

_ENERGY_DEPENDENT_HALLUCINATION_PHRASES = [
    "thanks for watching", "thank you for watching",
    "thank you for listening", "thanks for listening",
    "thank you", "thanks",
    "देखने के लिए धन्यवाद", "धन्यवाद",
    "شكرا للمشاهدة", "شكرا",
]

_HALLUCINATION_ENERGY_THRESHOLD = 0.01

def get_audio_rms(file_path: str) -> float:
    try:
        data, _ = sf.read(file_path, dtype="float32", always_2d=True)
        mono = data.mean(axis=1)
        return float(np.sqrt(np.mean(mono ** 2)))
    except Exception as e:
        logger.warning(f"[RMS] Could not compute RMS for {file_path}: {e}")
        return 1.0

def _is_hallucination(text: str, audio_path: Optional[str] = None) -> bool:
    if not text or not text.strip():
        return True
    low = text.strip().lower()
    if any(p in low for p in _ALWAYS_HALLUCINATION_PHRASES):
        return True
    if any(p in low for p in _ENERGY_DEPENDENT_HALLUCINATION_PHRASES):
        if audio_path:
            rms = get_audio_rms(audio_path)
            logger.info(f"[HALLUCINATION] RMS={rms:.5f} threshold={_HALLUCINATION_ENERGY_THRESHOLD}")
            if rms < _HALLUCINATION_ENERGY_THRESHOLD:
                logger.info(f"[HALLUCINATION] Low-energy → filtering: '{text}'")
                return True
            return False
        return False
    return False

def get_audio_duration(file_path: str) -> float:
    try:
        f = sf.SoundFile(file_path)
        return len(f) / f.samplerate
    except:
        return 0.0

def load_audio_tensor(audio_path: str) -> torch.Tensor:
    wav, sr = torchaudio.load(audio_path)
    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(wav)
    return wav

def run_indic_trans(text: str, src_lang: str, tgt_lang: str) -> str:
    if not text or not INDIC_PROCESSOR_AVAILABLE:
        return text
    if src_lang == "en":
        direction = "en-indic"
    elif tgt_lang == "en":
        direction = "indic-en"
    else:
        direction = "indic-indic"
    if direction not in it2_models:
        return f"[Model {direction} missing]"
    src_code = IT2_CODE_MAP.get(src_lang)
    tgt_code = IT2_CODE_MAP.get(tgt_lang)
    if not src_code or not tgt_code:
        return None
    try:
        with it2_lock:
            tokenizer = it2_tokenizers[direction]
            model = it2_models[direction]
            ip = IndicProcessor(inference=True)
            batch = ip.preprocess_batch([text], src_lang=src_code, tgt_lang=tgt_code)
            inputs = tokenizer(batch, truncation=True, padding="longest", return_tensors="pt", return_attention_mask=True).to(DEVICE_INDIC)
            with torch.no_grad():
                generated = model.generate(**inputs, use_cache=True, min_length=0, max_length=512, num_beams=5, num_return_sequences=1)
            with tokenizer.as_target_tokenizer():
                decoded = tokenizer.batch_decode(generated.detach().cpu().tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=True)
            translations = ip.postprocess_batch(decoded, lang=tgt_code)
            return translations[0]
    except Exception as e:
        logger.error(f"IndicTrans Error: {e}")
        return "[Translation Error]"

def run_nllb_trans(text: str, src_code: str, tgt_code: str) -> str:
    if not nllb_model or not text:
        return text
    try:
        with nllb_lock:
            nllb_tokenizer.src_lang = src_code
            inputs = nllb_tokenizer(text, return_tensors="pt").to(DEVICE_NLLB)
            forced_bos_token_id = nllb_tokenizer.lang_code_to_id[tgt_code]
            with torch.no_grad():
                generated_tokens = nllb_model.generate(**inputs, forced_bos_token_id=forced_bos_token_id, max_length=512)
            result = nllb_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
            return result
    except Exception as e:
        logger.error(f"NLLB Error: {e}")
        return "[NLLB Error]"

def transcribe_indic_conformer(audio_path: str, lang: str) -> str:
    if ic_model is None:
        raise RuntimeError("IndicConformer is not loaded")
    wav = load_audio_tensor(audio_path)
    with ic_lock:
        with torch.no_grad():
            text = ic_model(wav, lang, "ctc")
    if isinstance(text, list):
        text = text[0]
    return str(text).strip()

def transcribe_whisper(audio_path: str, task: str = "transcribe", language: str = None) -> Tuple[str, str]:
    if whisper_model is None:
        raise RuntimeError("Whisper is not loaded")
    with whisper_lock:
        audio = whisper.load_audio(audio_path)
        opts = {"task": task, "beam_size": 5}
        if language:
            opts["language"] = language
        result = whisper_model.transcribe(audio, **opts)
    return result["text"].strip(), result.get("language", "en")

@app.post("/translate")
async def translate_endpoint(
    text: str = Form(...),
    source_language: str = Form(...),
    target_language: str = Form(...),
    x_api_key: str = Header(default="")
):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    start_time = time.time()

    def to_short_code(nllb_code):
        for k, v in IT2_CODE_MAP.items():
            if v == nllb_code:
                return k
        return nllb_code

    src_short = to_short_code(source_language)
    tgt_short = to_short_code(target_language)
    translated = ""
    used_model = "none"

    if (src_short in INDIC_LANGS or src_short == 'en') and (tgt_short in INDIC_LANGS or tgt_short == 'en'):
        res = run_indic_trans(text, src_short, tgt_short)
        if res is not None:
            translated = res
            used_model = "indic_trans2"

    if not translated:
        translated = run_nllb_trans(text, source_language, target_language)
        used_model = "nllb"

    return {"success": True, "translated_text": translated, "used_model": used_model, "processing_time": round(time.time() - start_time, 3)}

@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language_mode: str = Form(default="auto"),
    target_language: str = Form(default="en"),
    x_api_key: str = Header(default="")
):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    start_time = time.time()
    temp_path = None
    try:
        suffix = os.path.splitext(file.filename or "audio.wav")[-1] or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            temp_path = tmp.name

        detected_lang = language_mode
        transcribed_text = ""
        translated_text = ""

        if language_mode in INDIC_LANGS:
            transcribed_text = transcribe_indic_conformer(temp_path, language_mode)
            detected_lang = language_mode
            if _is_hallucination(transcribed_text, audio_path=temp_path):
                logger.info(f"[HALLUCINATION] Filtered: '{transcribed_text}'")
                transcribed_text = ""
            if target_language != language_mode and transcribed_text:
                res = run_indic_trans(transcribed_text, language_mode, target_language)
                if res:
                    translated_text = res
            else:
                translated_text = transcribed_text
        else:
            task = "transcribe"
            if target_language == "en" and language_mode != "en":
                task = "translate"
            w_lang = None if language_mode == "auto" else language_mode
            text, det_lang = transcribe_whisper(temp_path, task=task, language=w_lang)
            if _is_hallucination(text, audio_path=temp_path):
                logger.info(f"[HALLUCINATION] Filtered: '{text}'")
                text = ""
            transcribed_text = text
            if language_mode == "auto":
                detected_lang = det_lang
            current_text_lang = "en" if task == "translate" else detected_lang
            if target_language in INDIC_LANGS:
                if current_text_lang == target_language:
                    translated_text = transcribed_text
                else:
                    res = run_indic_trans(transcribed_text, current_text_lang, target_language)
                    if res:
                        translated_text = res
            else:
                translated_text = transcribed_text

        return {"success": True, "text": transcribed_text, "translated_text": translated_text, "detected_language": detected_lang, "processing_time": round(time.time() - start_time, 3)}

    except Exception as e:
        logger.error(f"Error processing request: {e}")
        traceback.print_exc()
        return {"success": False, "error": str(e), "processing_time": round(time.time() - start_time, 3)}
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

@app.get("/health")
def health():
    return {"status": "online", "models": {"whisper": whisper_model is not None, "indic_conformer": ic_model is not None, "indic_trans": list(it2_models.keys())}}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
