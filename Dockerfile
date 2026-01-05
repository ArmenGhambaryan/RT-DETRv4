# -------------------------------------------------------
# Base: PyTorch + CUDA 12.1 + cuDNN 9 + Python 3.11
# -------------------------------------------------------
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

# -------------------------------------------------------
# System deps
# -------------------------------------------------------
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    build-essential \
  && rm -rf /var/lib/apt/lists/*

RUN python --version

# -------------------------------------------------------
# Python tooling
# -------------------------------------------------------
RUN pip install --upgrade pip setuptools wheel

# -------------------------------------------------------
# Copy & install ONLY requirements first (cache-friendly)
# -------------------------------------------------------
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# -------------------------------------------------------
# Useful extras
# -------------------------------------------------------
RUN pip install opencv-python onnxruntime-gpu tqdm \
    onnx onnxsim
#  && pip install --upgrade tensorrt-cu12

# -------------------------------------------------------
# Copy full project LAST
# -------------------------------------------------------
COPY . /workspace

# -------------------------------------------------------
# Default
# -------------------------------------------------------
CMD ["bash"]
