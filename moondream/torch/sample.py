import argparse
import json
import os

import torch
from PIL import Image
from transformers import AutoTokenizer

from .rope import precompute_freqs_cis
from .text import lm_head, text_decoder, text_encoder
from .weights import load_weights_into_model
from .moondream import MoondreamModel, MoondreamConfig

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", "-i", type=str, required=True)
    parser.add_argument("--prompt", "-p", type=str, required=True)
    parser.add_argument("--model", "-m", type=str, required=True)
    parser.add_argument("--config", "-c", type=str, default="{}")
    parser.add_argument("--max-tokens", "-t", type=int, default=200)
    parser.add_argument("--sampler", "-s", type=str, default="greedy")
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.set_default_device("cuda")
    elif torch.backends.mps.is_available():
        torch.set_default_device("mps")

    # Load config.
    config = json.loads(args.config)

    # Load model.
    # model_path = args.model
    # if not os.path.exists(model_path):
    #     raise FileNotFoundError(f"Model not found at {model_path}")
    # if model_path.endswith(".pt"):
    #     model = load_from_pt(model_path, **config)
    # elif model_path.endswith(".safetensors"):
    #     model = load_from_safetensors(model_path, **config)
    # else:
    #     raise ValueError(f"Invalid model format: {model_path}")

    # Load model.
    config = MoondreamConfig()
    model = MoondreamModel(config)
    load_weights_into_model(args.model, model)

    # Encode image.
    image_path = args.image
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
    image = Image.open(image_path)
    with torch.no_grad():
        image_tensor = model._encode_image(image)

    # Encode text, and create inputs_embeds.
    tokenizer = AutoTokenizer.from_pretrained("vikhyatk/moondream2")
    prompt = f"\n\nQuestion: {args.prompt}\n\nAnswer:"
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        input_ids = torch.cat(
            [torch.tensor([[tokenizer.eos_token_id]]), input_ids], dim=1
        )
        inputs_embeds = text_encoder(input_ids, model.text)
        inputs_embeds = torch.cat(
            [
                inputs_embeds[:, 0:1, :],
                image_tensor.unsqueeze(0),
                inputs_embeds[:, 1:, :],
            ],
            dim=1,
        )

    kv_cache = torch.empty(
        config.text.n_layers,
        2,  # k, v
        1,  # bsz
        config.text.n_heads,
        config.text.max_context,
        config.text.dim // config.text.n_heads,
        dtype=torch.float16,
    )
    freqs_cis = precompute_freqs_cis(32, 2048)
    pos = 0

    for _ in range(args.max_tokens):
        with torch.no_grad():
            hidden, kv_cache_update = text_decoder(
                inputs_embeds,
                model.text,
                kv_cache[:, :, :, :, :pos, :],
                freqs_cis,
                config.text,
            )
            logits = lm_head(hidden, model.text)
            kv_cache[:, :, :, :, pos : pos + kv_cache_update.size(-2), :] = (
                kv_cache_update
            )
            pos += kv_cache_update.size(-2)

            if args.sampler == "multinomial":
                next_token = torch.multinomial(
                    torch.softmax(logits, dim=-1), num_samples=1
                ).squeeze(0)
            elif args.sampler == "greedy":
                next_token = torch.argmax(logits, dim=-1)
            else:
                raise ValueError(f"Invalid sampler: {args.sampler}")

            if next_token == tokenizer.eos_token_id:
                print()
                break

            input_ids = next_token.unsqueeze(0)
            inputs_embeds = text_encoder(input_ids, model.text)

            output_text = tokenizer.batch_decode(input_ids)[0]
            print(output_text, end="", flush=True)
