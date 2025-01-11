"""
Gaze Detection Video Processor using Moondream 2
------------------------------------------------
Read the README.md file for more information on how to use this script. Contact us in our discord for any questions if you get stuck.
"""

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoModelForCausalLM
from tqdm import tqdm
import os
import glob
from typing import List, Dict, Tuple, Optional
from contextlib import contextmanager


def initialize_model() -> Optional[AutoModelForCausalLM]:
    """Initialize the Moondream 2 model with error handling."""
    try:
        print("\nInitializing Moondream 2 model...")
        model_id = "vikhyatk/moondream2"
        revision = "2025-01-09"  # Specify revision for stability

        if torch.cuda.is_available():
            print(f"GPU detected: {torch.cuda.get_device_name(0)}")
            device = "cuda"
        else:
            print("No GPU detected, using CPU")
            device = "cpu"

        print("Loading model from HuggingFace...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            trust_remote_code=True,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
            device_map={"": device} if device == "cuda" else None,
        )

        if device == "cpu":
            model = model.to(device)
        model.eval()

        print("✓ Model initialized successfully")
        return model
    except Exception as e:
        print(f"\nError initializing model: {e}")
        return None


@contextmanager
def video_handler(
    input_path: str, output_path: str
) -> Tuple[cv2.VideoCapture, cv2.VideoWriter]:
    """Context manager for handling video capture and writer."""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {input_path}")

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    try:
        yield cap, out
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()


def fig2rgb_array(fig: plt.Figure) -> np.ndarray:
    """Convert matplotlib figure to RGB array"""
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    w, h = fig.canvas.get_width_height()
    img_array = np.asarray(buf).reshape((h, w, 4))
    rgb_array = img_array[:, :, :3]  # Drop alpha channel
    return rgb_array


def visualize_frame(
    frame: np.ndarray, faces: List[Dict], model: AutoModelForCausalLM, pil_image: Image
) -> np.ndarray:
    """Visualize a single frame using matplotlib"""
    try:
        # Create figure without margins
        fig = plt.figure(figsize=(frame.shape[1] / 100, frame.shape[0] / 100), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1])

        # Display frame
        ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Sort faces by x_min coordinate for stable colors
        faces = sorted(faces, key=lambda f: (f["y_min"], f["x_min"]))

        # Generate colors
        colors = plt.cm.rainbow(np.linspace(0, 1, max(1, len(faces))))

        # Process each face
        for face, color in zip(faces, colors):
            try:
                # Calculate face box coordinates
                x_min = int(float(face["x_min"]) * frame.shape[1])
                y_min = int(float(face["y_min"]) * frame.shape[0])
                width = int(float(face["x_max"] - face["x_min"]) * frame.shape[1])
                height = int(float(face["y_max"] - face["y_min"]) * frame.shape[0])

                # Draw face rectangle
                rect = plt.Rectangle(
                    (x_min, y_min), width, height, fill=False, color=color, linewidth=2
                )
                ax.add_patch(rect)

                # Calculate face center
                face_center = (
                    float(face["x_min"] + face["x_max"]) / 2,
                    float(face["y_min"] + face["y_max"]) / 2,
                )

                # Try to detect gaze
                try:
                    gaze_result = model.detect_gaze(pil_image, face_center)
                    if isinstance(gaze_result, dict) and "gaze" in gaze_result:
                        gaze = gaze_result["gaze"]
                    else:
                        gaze = gaze_result
                except Exception as e:
                    print(f"Error detecting gaze: {e}")
                    continue

                if (
                    gaze is not None
                    and isinstance(gaze, dict)
                    and "x" in gaze
                    and "y" in gaze
                ):
                    gaze_x = int(float(gaze["x"]) * frame.shape[1])
                    gaze_y = int(float(gaze["y"]) * frame.shape[0])
                    face_center_x = x_min + width // 2
                    face_center_y = y_min + height // 2

                    # Draw gaze line with gradient effect
                    points = 50
                    alphas = np.linspace(0.8, 0, points)

                    # Calculate points along the line
                    x_points = np.linspace(face_center_x, gaze_x, points)
                    y_points = np.linspace(face_center_y, gaze_y, points)

                    # Draw gradient line segments
                    for i in range(points - 1):
                        ax.plot(
                            [x_points[i], x_points[i + 1]],
                            [y_points[i], y_points[i + 1]],
                            color=color,
                            alpha=alphas[i],
                            linewidth=4,
                        )

                    # Draw gaze point
                    ax.scatter(gaze_x, gaze_y, color=color, s=100, zorder=5)
                    ax.scatter(gaze_x, gaze_y, color="white", s=50, zorder=6)

            except Exception as e:
                print(f"Error processing face: {e}")
                continue

        # Configure axes
        ax.set_xlim(0, frame.shape[1])
        ax.set_ylim(frame.shape[0], 0)
        ax.axis("off")

        # Convert matplotlib figure to image
        frame_rgb = fig2rgb_array(fig)

        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # Clean up
        plt.close(fig)

        return frame_bgr

    except Exception as e:
        print(f"Error in visualize_frame: {e}")
        plt.close("all")
        return frame


def process_video(
    input_path: str, output_path: str, model: AutoModelForCausalLM
) -> None:
    """Process video file and create new video with gaze visualization"""
    with video_handler(input_path, output_path) as (cap, out):
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        print(f"Processing video: {total_frames} frames at {fps} FPS")

        # Process frames
        with tqdm(
            total=total_frames, desc=f"Processing {os.path.basename(input_path)}"
        ) as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                try:
                    # Convert frame for model
                    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                    # Detect faces
                    detection_result = model.detect(pil_image, "face")

                    # Handle different possible return formats
                    if (
                        isinstance(detection_result, dict)
                        and "objects" in detection_result
                    ):
                        faces = detection_result["objects"]
                    elif isinstance(detection_result, list):
                        faces = detection_result
                    else:
                        print(
                            f"Unexpected detection result format: {type(detection_result)}"
                        )
                        faces = []

                    # Ensure each face has the required coordinates
                    faces = [
                        face
                        for face in faces
                        if all(k in face for k in ["x_min", "y_min", "x_max", "y_max"])
                    ]

                    if not faces:
                        processed_frame = frame
                    else:
                        # Visualize frame with matplotlib
                        processed_frame = visualize_frame(
                            frame, faces, model, pil_image
                        )

                    # Write frame
                    out.write(processed_frame)
                    pbar.update(1)

                    # Force matplotlib to clean up
                    plt.close("all")

                except Exception as e:
                    print(f"Error processing frame: {e}")
                    out.write(frame)  # Write original frame on error
                    pbar.update(1)
                    plt.close("all")  # Clean up even on error


if __name__ == "__main__":
    # Ensure input and output directories exist
    input_dir = os.path.join(os.path.dirname(__file__), "input")
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Find all video files in input directory
    video_extensions = [".mp4", ".avi", ".mov", ".mkv"]
    input_videos = []
    for ext in video_extensions:
        input_videos.extend(glob.glob(os.path.join(input_dir, f"*{ext}")))

    if not input_videos:
        print("No video files found in input directory")
        exit(1)

    # Initialize model once for all videos
    model = initialize_model()
    if model is None:
        print("Failed to initialize model")
        exit(1)

    # Process each video file
    for input_video in input_videos:
        base_name = os.path.basename(input_video)
        output_video = os.path.join(output_dir, f"processed_{base_name}")
        try:
            process_video(input_video, output_video, model)
        except Exception as e:
            print(f"Error processing {base_name}: {e}")
            continue
