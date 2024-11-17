import time
import tracemalloc

from PIL import Image

import moondream as md
from moondream.preprocess import create_patches

MODEL_PATH = "../../onnx/out/moondream-latest-int4.bin"


class Colors:
    HEADER = "\033[95m"  # Purple
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"


def format_memory(memory_mb):
    """Format memory size with appropriate unit"""
    if memory_mb < 1024:
        return f"{memory_mb:.2f} MB"
    else:
        return f"{memory_mb/1024:.2f} GB"


def print_section(title):
    """Print a section header with dynamic padding to center the text"""
    total_width = 65
    text_length = len(title) + 2  # Add 2 for spaces around title
    total_padding = total_width - text_length
    left_padding = total_padding // 2
    right_padding = total_padding - left_padding
    print(
        f"\n{Colors.HEADER}{Colors.BOLD}{'-'*left_padding} {title} {'-'*right_padding}{Colors.ENDC}"
    )


def print_metric(label, value, color=Colors.BLUE):
    """Print a metric with consistent formatting"""
    print(f"| {color}{label}{Colors.ENDC}: {value}")


def log_memory_and_time(operation_name, start_time, start_memory):
    """Log memory and time differences for an operation"""
    end_time = time.time()
    current_memory = get_memory_usage()
    time_diff = end_time - start_time
    memory_diff = current_memory - start_memory

    print("\nStats")
    print_metric("Time", f"{time_diff:.2f} seconds")
    print_metric("Memory usage", format_memory(current_memory))

    # Color-code memory increase based on significance
    color = (
        Colors.GREEN
        if memory_diff < 10
        else Colors.YELLOW if memory_diff < 100 else Colors.RED
    )
    print_metric("Memory increase", format_memory(memory_diff), color)

    return end_time, current_memory


def get_memory_usage():
    """Get current memory usage in MB"""
    current, peak = tracemalloc.get_traced_memory()
    return current / 1024 / 1024


# Start tracking memory
tracemalloc.start()

# Initial memory measurement
initial_memory = get_memory_usage()
print_section("Initial State")
print_metric("Initial memory usage", format_memory(initial_memory))

# Load image
print_section("Image Loading")
start_time = time.time()
start_memory = get_memory_usage()
image = Image.open("../../assets/demo-1.jpg")
log_memory_and_time("Image Loading", start_time, start_memory)

# Initialize model
print_section("Model Initialization")
start_time = time.time()
start_memory = get_memory_usage()
model = md.VL(MODEL_PATH)
log_memory_and_time("Model Initialization", start_time, start_memory)

# Encode image
print_section("Image Encoding")
start_time = time.time()
start_memory = get_memory_usage()
encoded_image = model.encode_image(image)
log_memory_and_time("Image Encoding", start_time, start_memory)

# Generate caption
print_section("Caption Generation")
print(f"{Colors.BOLD}Caption:{Colors.ENDC}", end="", flush=True)
start_time = time.time()
start_memory = get_memory_usage()
tokens = 0
for tok in model.caption(encoded_image, stream=True)["caption"]:
    print(tok, end="", flush=True)
    tokens += 1
print()
end_time, end_memory = log_memory_and_time("Caption Stats", start_time, start_memory)
print_metric("Token generation speed", f"{tokens / (end_time - start_time):.2f} tok/s")

# Generate answer to question
question = "How many people are in this image?"
print_section("Question Answering")
print(f"{Colors.BOLD}Question:{Colors.ENDC} {question}")
print(f"{Colors.BOLD}Answer:{Colors.ENDC}", end="", flush=True)
start_time = time.time()
start_memory = get_memory_usage()
tokens = 0
for tok in model.query(encoded_image, question, stream=True)["answer"]:
    print(tok, end="", flush=True)
    tokens += 1
print()
end_time, end_memory = log_memory_and_time(
    "Question Answering Stats", start_time, start_memory
)
print_metric("Token generation speed", f"{tokens / (end_time - start_time):.2f} tok/s")

# Final summary
print_section("Final Summary")
final_memory = get_memory_usage()
current, peak = tracemalloc.get_traced_memory()

print_metric("Final memory usage", format_memory(final_memory))
print_metric("Total memory increase", format_memory(final_memory - initial_memory))
print_metric("Peak memory usage", format_memory(peak / 1024 / 1024))

# Stop tracking memory
tracemalloc.stop()
