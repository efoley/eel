Eel is "Eric's infErence Library".

This is simple LLM inference with a thin python wrapper.

# Install

Create a new python environment, e.g.

`python -m venv .venv`

and with that environment activated, install the dependencies:

`pip install -r requirements.txt`

Make the C code:

`make`

# Usage

Run `scripts/download_models.sh` to download the models.

Run `scripts/main.py data/Qwen2.5-0.5B-Instruct`. This will run a timing test and output something like:

```
(eel) eric@macbook eel % scripts/main.py data/Qwen2.5-0.5B-Instruct
Prompt time: 1.5839s
Total time: 3.9726s
Certainly! A large language model, or LLM, is a type of artificial intelligence that can generate human-like text based on a given input. These models are designed to be highly accurate, creative, and adaptable, making them useful in a wide range of applications, including but not limited to:

1. **Chatbots**:
```

# Performance

At present, there is just a simple C implementation using only 32-bit floats.

On MacOS, matrix-vector multiplies use Accelerate BLAS, but the performance improvement from that over simple hand-coded matrix-vector multiply is minimal.

