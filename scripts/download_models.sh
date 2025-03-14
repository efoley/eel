#!/bin/bash
set -euxo pipefail


ORG=Qwen
MODEL=Qwen2.5-Coder-0.5B

mkdir -p data/${MODEL}
cd data/${MODEL}

# loop over the names config.json, tokenizer.json, and model.safetensors and download them all
for file in config.json tokenizer.json model.safetensors; do
    curl -L "https://huggingface.co/${ORG}/${MODEL}/resolve/main/${file}" -o $file
done
