import argparse
import json
import os
import torch

from PIL import Image, ImageDraw
from tqdm import tqdm

from .weights import load_weights_into_model
from .moondream import MoondreamModel, MoondreamConfig

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", "-i", type=str, required=True)
    parser.add_argument("--prompt", "-p", type=str, required=True, help="Prompt for query endpoint or object for detect/point endpoints")
    parser.add_argument("--model", "-m", type=str, required=True)
    parser.add_argument("--config", "-c", type=str, default=None)
    parser.add_argument("--max-tokens", "-t", type=int, default=200)
    parser.add_argument("--sampler", "-s", type=str, default="greedy")
    parser.add_argument("--benchmark", "-b", action="store_true")
    parser.add_argument("--endpoint", "-e", type=str, choices=['caption', 'query', 'detect', 'point'], required=True,
                      help="Select the endpoint to use: caption, query, detect, or point")
    parser.add_argument("--caption-type", "-ct", type=str, choices=['short', 'normal'], default='normal',
                      help="Type of caption when using caption endpoint")
    args = parser.parse_args()
    if args.endpoint == 'query' and not args.prompt:
        parser.error("--prompt is required when using the query endpoint")
    if args.endpoint in ['detect', 'point'] and not args.prompt:
        parser.error("--prompt is required when using detect or point endpoints (specify the object to detect/point)")

    if torch.cuda.is_available():
        torch.set_default_device("cuda")
    elif torch.backends.mps.is_available():
        torch.set_default_device("mps")

    # Load config.

    # Load model.
    if args.config is not None:
        with open(args.config, "r") as f:
            config = json.load(f)
        config = MoondreamConfig.from_dict(config)
    else:
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

        if args.endpoint == 'caption':
            print(f"Caption: {args.caption_type}")
            for t in model.caption(encoded_image, args.caption_type, stream=True)["caption"]:
                print(t, end="", flush=True)
            print()

        elif args.endpoint == 'query':
            print("Query:", args.prompt)
            for t in model.query(encoded_image, args.prompt, stream=True)["answer"]:
                print(t, end="", flush=True)
            print()

        elif args.endpoint == 'detect':
            print(f"Detect: {args.prompt}")
            result = model.detect(encoded_image, args.prompt)
            objs = result["objects"]
            print(f"Found {len(objs)}")
            print(result)
            draw = ImageDraw.Draw(image)
            for obj in objs:
                x_min, y_min, x_max, y_max = (
                    obj["x_min"] * image.width,
                    obj["y_min"] * image.height,
                    obj["x_max"] * image.width,
                    obj["y_max"] * image.height,
                )
                draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=2)
            image.save("detect.jpg")
            print(f"Detection results saved to detect.jpg")

        elif args.endpoint == 'point':
            print(f"Point: {args.prompt}")
            points = model.point(encoded_image, args.prompt)["points"]
            print(f"Found {len(points)}")
            draw = ImageDraw.Draw(image)
            for point in points:
                x, y = point["x"] * image.width, point["y"] * image.height
                draw.ellipse([x - 5, y - 5, x + 5, y + 5], fill="red")
            image.save("point.jpg")
            print(f"Point results saved to point.jpg")

        # Detect gaze
        model.detect_gaze(encoded_image, (0.5, 0.5))
    else:
        torch._dynamo.reset()
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
