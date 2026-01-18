FROM ubuntu:22.04

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libboost-math-dev \
    libboost-python-dev \
    libboost-timer-dev \
    libboost-thread-dev \
    libboost-system-dev \
    libboost-filesystem-dev \
    libopenblas-dev \
    libomp-dev \
    cmake \
    pkg-config \
    git \
    python3-pip \
    curl \
    libcurl4-openssl-dev \
    wget && \
    rm -rf /var/lib/apt/lists/*
    
# Install Python dependencies
RUN pip3 install huggingface-hub openai gradio

# Build llama.cpp with OpenBLAS
RUN git clone https://github.com/ggerganov/llama.cpp && \
    cd llama.cpp && \
    cmake -B build -S . \
        -DLLAMA_BUILD_SERVER=ON \
        -DLLAMA_BUILD_EXAMPLES=ON \
        -DGGML_BLAS=ON \
        -DGGML_BLAS_VENDOR=OpenBLAS \
        -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build --config Release --target llama-server -j $(nproc)

# Download model
RUN mkdir -p /models && \
    wget -O /models/model.q8_k_xl.gguf https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-UD-Q8_K_XL.gguf

# Copy app and startup script
COPY app.py /app.py
COPY start.sh /start.sh
RUN chmod +x /start.sh

# Expose ports
EXPOSE 7860 8080

# Start services
CMD ["/start.sh"]