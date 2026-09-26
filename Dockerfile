FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/cache/hf \
    HOME=/tmp \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WORKDIR /app

# torch wheels from PyPI bundle the CUDA runtime; only the host driver is needed.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Triton (used by torch's native kernels, e.g. rotary embeddings) compiles a
# small helper with a C compiler on first use.
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*

COPY src/ src/

CMD ["sh", "-c", "python src/generate_data.py && python src/train.py"]
