#!/bin/bash
set -euxo pipefail


ORG=Qwen
MODEL=Qwen2.5-0.5B-Instruct

mkdir -p data/${MODEL}
cd data/${MODEL}

FILES=(
    "config.json"
    "tokenizer.json"
    "tokenizer_config.json"
    "model.safetensors"
)

# loop over the names config.json, tokenizer.json, and model.safetensors and download them all
for file in "${FILES[@]}"; do
    curl -L "https://huggingface.co/${ORG}/${MODEL}/resolve/main/${file}" -o $file
done
