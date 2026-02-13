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
