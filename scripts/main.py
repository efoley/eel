#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from eel import init_state, load_config, load_weights, load_model

from eel.eel import forward


def test(tokenizer, model, state) -> str:
    prompt = "Give me a short introduction to large language model."
    messages = [
        {
            "role": "system",
            "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
        },
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    token_ids = tokenizer(text)["input_ids"]

    # feed in the prompt
    pos = 0
    logits = None
    for token_id in token_ids:
        logits = forward(model, state, token_id, pos)
        pos += 1
        assert np.all(np.isfinite(logits))

    output_tokens = []

    # sample starting with last token of prompt
    last_output_token_id = np.argmax(logits)
    output_tokens.append(last_output_token_id)
    #
    for _ in range(20):
        logits = forward(model, state, last_output_token_id, pos)
        pos += 1
        assert np.all(np.isfinite(logits))
        print(logits)
        last_output_token_id = np.argmax(logits)
        output_tokens.append(last_output_token_id)

    print(tokenizer.decode(output_tokens, skip_special_tokens=True))


def main():
    parser = argparse.ArgumentParser(description="Run eel.")
    parser.add_argument("model_dir", help="Path to the HF model directory")

    parser.add_argument(
        "--mode",
        choices=["test", "chat", "perplexity", "passcode"],
        default="test",
        help="The mode to run the script in (default: test)",
    )

    args = parser.parse_args()

    model_dir = Path(args.model_dir)

    config = load_config(model_dir / "config.json")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    weights = load_weights(config, model_dir / "model.safetensors")

    model = load_model(config, weights)

    state = init_state(config)

    if args.mode == "test":
        test(tokenizer, model, state)
    elif args.mode == "chat":
        # TODO
        pass
    elif args.mode == "perplexity":
        # TODO
        pass
    elif args.mode == "passcode":
        # TODO
        pass
    else:
        raise ValueError("Invalid mode")


if __name__ == "__main__":
    main()
