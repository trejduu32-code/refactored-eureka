#!/bin/bash

# Start llama-server in background
cd /llama.cpp/build
./bin/llama-server --host 0.0.0.0 --port 8080 --model /models/model.q8_k_xl.gguf --ctx-size 32768 --threads 2  &

# Wait for server to initialize
echo "Waiting for server to start..."
until curl -s "http://localhost:8080/v1/models" >/dev/null; do
    sleep 1
done

echo "Server is ready. Starting Gradio app."

# Start Gradio UI
cd /
python3 app.py