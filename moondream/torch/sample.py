import argparse
import json
import os
import torch

from PIL import Image
from tqdm import tqdm

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
    parser.add_argument("--benchmark", "-b", action="store_true")
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.set_default_device("cuda")
    elif torch.backends.mps.is_available():
        torch.set_default_device("mps")

    # Load config.
    config = json.loads(args.config)

    # Load model.
    config = MoondreamConfig()
    model = MoondreamModel(config)
    load_weights_into_model(args.model, model)

    # Encode image.
    image_path = args.image
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
    image = Image.open(image_path)

    if not args.benchmark:
        encoded_image = model.encode_image(image)
        for t in model.query(encoded_image, args.prompt, stream=True)["answer"]:
            print(t, end="", flush=True)
        print()
    else:
        model.compile()

        # Warmup runs
        for _ in tqdm(range(5), desc="Warmup"):
            encoded_image = model.encode_image(image)
            for _ in model.query(encoded_image, args.prompt, stream=True)["answer"]:
                pass

        # Benchmark runs
        encode_times = []
        query_speeds = []
        for i in tqdm(range(10), desc="Benchmark"):
            # Measure encode time
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            encoded_image = model.encode_image(image)
            end.record()
            torch.cuda.synchronize()
            encode_time = start.elapsed_time(end)
            encode_times.append(encode_time)

            # Measure query speed
            tokens = []
            query_start = torch.cuda.Event(enable_timing=True)
            query_end = torch.cuda.Event(enable_timing=True)
            query_start.record()
            for t in model.query(encoded_image, args.prompt, stream=True)["answer"]:
                tokens.append(t)
            query_end.record()
            torch.cuda.synchronize()
            query_time = query_start.elapsed_time(query_end)
            tokens_per_sec = len(tokens) / (query_time / 1000.0)  # Convert ms to s
            query_speeds.append(tokens_per_sec)

        # Print results
        print("\nBenchmark Results (10 runs):")
        print("Image Encoding Time (ms):")
        print(f"  Mean: {sum(encode_times)/len(encode_times):.2f}")
        print(f"  Min:  {min(encode_times):.2f}")
        print(f"  Max:  {max(encode_times):.2f}")
        print("\nQuery Speed (tokens/sec):")
        print(f"  Mean: {sum(query_speeds)/len(query_speeds):.2f}")
        print(f"  Min:  {min(query_speeds):.2f}")
        print(f"  Max:  {max(query_speeds):.2f}")
