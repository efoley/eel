#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

from tokenizers import Tokenizer
from eel import init_state, load_config, load_weights, load_model

from eel.eel import forward


def main():
    parser = argparse.ArgumentParser(description="Run eel.")
    parser.add_argument("model_dir", help="Path to the HF model directory")

    args = parser.parse_args()

    model_dir = Path(args.model_dir)

    config = load_config(model_dir / "config.json")

    weights, owned_tensors = load_weights(config, model_dir / "model.safetensors")

    model = load_model(config, weights)

    state = init_state(config)

    for i in range(10):
        logits = forward(model, state, i, i)
        print(logits)


if __name__ == "__main__":
    main()