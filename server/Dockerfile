# ---------------------------------------------------------
# CUDA base (Ubuntu 22.04 + CUDA 12.1.1)
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
# Whisper also needs recent torch/transformers
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
  scipy \
  fastapi \
  uvicorn[standard] \
  python-multipart \
  onnxruntime-gpu

# ---------------------------------------------------------
# Whisper
# ---------------------------------------------------------
RUN pip install --no-cache-dir openai-whisper

# ---------------------------------------------------------
# IndicTransToolkit — PyPI
# ---------------------------------------------------------
RUN pip install --no-cache-dir indictranstoolkit

# ---------------------------------------------------------
# App
# ---------------------------------------------------------
WORKDIR /app
COPY server.py /app/server.py

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
