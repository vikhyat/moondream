#!/usr/bin/env python3
import cv2, os, subprocess, argparse
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import numpy as np

# Constants
TEST_MODE_DURATION = 3  # Process only first 3 seconds in test mode
FFMPEG_PRESETS = [
    "ultrafast",
    "superfast",
    "veryfast",
    "faster",
    "fast",
    "medium",
    "slow",
    "slower",
    "veryslow",
]
FONT = cv2.FONT_HERSHEY_SIMPLEX  # Font for bounding-box-style labels

# Detection parameters
IOU_THRESHOLD = 0.5  # IoU threshold for considering boxes related

# Hitmarker parameters
HITMARKER_SIZE = 20  # Size of the hitmarker in pixels
HITMARKER_GAP = 3  # Size of the empty space in the middle (reduced from 8)
HITMARKER_THICKNESS = 2  # Thickness of hitmarker lines
HITMARKER_COLOR = (255, 255, 255)  # White color for hitmarker
HITMARKER_SHADOW_COLOR = (80, 80, 80)  # Lighter gray for shadow effect
HITMARKER_SHADOW_OFFSET = 1  # Smaller shadow offset


def load_moondream():
    """Load Moondream model and tokenizer."""
    model = AutoModelForCausalLM.from_pretrained(
        "vikhyatk/moondream2", trust_remote_code=True, device_map={"": "cuda"}
    )
    tokenizer = AutoTokenizer.from_pretrained("vikhyatk/moondream2")
    return model, tokenizer


def get_video_properties(video_path):
    """Get basic video properties."""
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video.release()
    return {"fps": fps, "frame_count": frame_count, "width": width, "height": height}


def is_valid_box(box):
    """Check if box coordinates are reasonable."""
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1

    # Reject boxes that are too large (over 90% of frame in both dimensions)
    if width > 0.9 and height > 0.9:
        return False

    # Reject boxes that are too small (less than 1% of frame)
    if width < 0.01 or height < 0.01:
        return False

    return True


def split_frame_into_tiles(frame, rows, cols):
    """Split a frame into a grid of tiles."""
    height, width = frame.shape[:2]
    tile_height = height // rows
    tile_width = width // cols
    tiles = []
    tile_positions = []

    for i in range(rows):
        for j in range(cols):
            y1 = i * tile_height
            y2 = (i + 1) * tile_height if i < rows - 1 else height
            x1 = j * tile_width
            x2 = (j + 1) * tile_width if j < cols - 1 else width

            tile = frame[y1:y2, x1:x2]
            tiles.append(tile)
            tile_positions.append((x1, y1, x2, y2))

    return tiles, tile_positions


def convert_tile_coords_to_frame(box, tile_pos, frame_shape):
    """Convert coordinates from tile space to frame space."""
    frame_height, frame_width = frame_shape[:2]
    tile_x1, tile_y1, tile_x2, tile_y2 = tile_pos
    tile_width = tile_x2 - tile_x1
    tile_height = tile_y2 - tile_y1

    x1_tile_abs = box[0] * tile_width
    y1_tile_abs = box[1] * tile_height
    x2_tile_abs = box[2] * tile_width
    y2_tile_abs = box[3] * tile_height

    x1_frame_abs = tile_x1 + x1_tile_abs
    y1_frame_abs = tile_y1 + y1_tile_abs
    x2_frame_abs = tile_x1 + x2_tile_abs
    y2_frame_abs = tile_y1 + y2_tile_abs

    x1_norm = x1_frame_abs / frame_width
    y1_norm = y1_frame_abs / frame_height
    x2_norm = x2_frame_abs / frame_width
    y2_norm = y2_frame_abs / frame_height

    x1_norm = max(0.0, min(1.0, x1_norm))
    y1_norm = max(0.0, min(1.0, y1_norm))
    x2_norm = max(0.0, min(1.0, x2_norm))
    y2_norm = max(0.0, min(1.0, y2_norm))

    return [x1_norm, y1_norm, x2_norm, y2_norm]


def merge_tile_detections(tile_detections, iou_threshold=0.5):
    """Merge detections from different tiles using NMS-like approach."""
    if not tile_detections:
        return []

    all_boxes = []
    all_keywords = []

    # Collect all boxes and their keywords
    for detections in tile_detections:
        for box, keyword in detections:
            all_boxes.append(box)
            all_keywords.append(keyword)

    if not all_boxes:
        return []

    # Convert to numpy for easier processing
    boxes = np.array(all_boxes)

    # Calculate areas
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    # Sort boxes by area
    order = areas.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        if order.size == 1:
            break

        # Calculate IoU with rest of boxes
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        # Get indices of boxes with IoU less than threshold
        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]

    return [(all_boxes[i], all_keywords[i]) for i in keep]


def detect_ads_in_frame(model, tokenizer, image, detect_keyword, rows=1, cols=1):
    """Detect objects in a frame using grid-based detection."""
    if rows == 1 and cols == 1:
        return detect_ads_in_frame_single(model, tokenizer, image, detect_keyword)

    # Convert numpy array to PIL Image if needed
    if not isinstance(image, Image.Image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Split frame into tiles
    tiles, tile_positions = split_frame_into_tiles(image, rows, cols)

    # Process each tile
    tile_detections = []
    for tile, tile_pos in zip(tiles, tile_positions):
        # Convert tile to PIL Image
        tile_pil = Image.fromarray(tile)

        # Detect objects in tile
        response = model.detect(tile_pil, detect_keyword)

        if response and "objects" in response and response["objects"]:
            objects = response["objects"]
            tile_objects = []

            for obj in objects:
                if all(k in obj for k in ["x_min", "y_min", "x_max", "y_max"]):
                    box = [obj["x_min"], obj["y_min"], obj["x_max"], obj["y_max"]]

                    if is_valid_box(box):
                        # Convert tile coordinates to frame coordinates
                        frame_box = convert_tile_coords_to_frame(
                            box, tile_pos, image.shape
                        )
                        tile_objects.append((frame_box, detect_keyword))

            if tile_objects:  # Only append if we found valid objects
                tile_detections.append(tile_objects)

    # Merge detections from all tiles
    merged_detections = merge_tile_detections(tile_detections)
    return merged_detections


def detect_ads_in_frame_single(model, tokenizer, image, detect_keyword):
    """Single-frame detection function."""
    detected_objects = []

    # Convert numpy array to PIL Image if needed
    if not isinstance(image, Image.Image):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Detect objects
    response = model.detect(image, detect_keyword)

    # Check if we have valid objects
    if response and "objects" in response and response["objects"]:
        objects = response["objects"]

        for obj in objects:
            if all(k in obj for k in ["x_min", "y_min", "x_max", "y_max"]):
                box = [obj["x_min"], obj["y_min"], obj["x_max"], obj["y_max"]]
                # If box is valid (not full-frame), add it
                if is_valid_box(box):
                    detected_objects.append((box, detect_keyword))

    return detected_objects


def draw_hitmarker(
    frame, center_x, center_y, size=HITMARKER_SIZE, color=HITMARKER_COLOR, shadow=True
):
    """Draw a COD-style hitmarker cross with more space in the middle."""
    half_size = size // 2

    # Draw shadow first if enabled
    if shadow:
        # Top-left to center shadow
        cv2.line(
            frame,
            (
                center_x - half_size + HITMARKER_SHADOW_OFFSET,
                center_y - half_size + HITMARKER_SHADOW_OFFSET,
            ),
            (
                center_x - HITMARKER_GAP + HITMARKER_SHADOW_OFFSET,
                center_y - HITMARKER_GAP + HITMARKER_SHADOW_OFFSET,
            ),
            HITMARKER_SHADOW_COLOR,
            HITMARKER_THICKNESS,
        )
        # Top-right to center shadow
        cv2.line(
            frame,
            (
                center_x + half_size + HITMARKER_SHADOW_OFFSET,
                center_y - half_size + HITMARKER_SHADOW_OFFSET,
            ),
            (
                center_x + HITMARKER_GAP + HITMARKER_SHADOW_OFFSET,
                center_y - HITMARKER_GAP + HITMARKER_SHADOW_OFFSET,
            ),
            HITMARKER_SHADOW_COLOR,
            HITMARKER_THICKNESS,
        )
        # Bottom-left to center shadow
        cv2.line(
            frame,
            (
                center_x - half_size + HITMARKER_SHADOW_OFFSET,
                center_y + half_size + HITMARKER_SHADOW_OFFSET,
            ),
            (
                center_x - HITMARKER_GAP + HITMARKER_SHADOW_OFFSET,
                center_y + HITMARKER_GAP + HITMARKER_SHADOW_OFFSET,
            ),
            HITMARKER_SHADOW_COLOR,
            HITMARKER_THICKNESS,
        )
        # Bottom-right to center shadow
        cv2.line(
            frame,
            (
                center_x + half_size + HITMARKER_SHADOW_OFFSET,
                center_y + half_size + HITMARKER_SHADOW_OFFSET,
            ),
            (
                center_x + HITMARKER_GAP + HITMARKER_SHADOW_OFFSET,
                center_y + HITMARKER_GAP + HITMARKER_SHADOW_OFFSET,
            ),
            HITMARKER_SHADOW_COLOR,
            HITMARKER_THICKNESS,
        )

    # Draw main hitmarker
    # Top-left to center
    cv2.line(
        frame,
        (center_x - half_size, center_y - half_size),
        (center_x - HITMARKER_GAP, center_y - HITMARKER_GAP),
        color,
        HITMARKER_THICKNESS,
    )
    # Top-right to center
    cv2.line(
        frame,
        (center_x + half_size, center_y - half_size),
        (center_x + HITMARKER_GAP, center_y - HITMARKER_GAP),
        color,
        HITMARKER_THICKNESS,
    )
    # Bottom-left to center
    cv2.line(
        frame,
        (center_x - half_size, center_y + half_size),
        (center_x - HITMARKER_GAP, center_y + HITMARKER_GAP),
        color,
        HITMARKER_THICKNESS,
    )
    # Bottom-right to center
    cv2.line(
        frame,
        (center_x + half_size, center_y + half_size),
        (center_x + HITMARKER_GAP, center_y + HITMARKER_GAP),
        color,
        HITMARKER_THICKNESS,
    )


def draw_ad_boxes(frame, detected_objects, detect_keyword, box_style="censor"):
    """Draw detection visualizations over detected objects.

    Args:
        frame: The video frame to draw on
        detected_objects: List of (box, keyword) tuples
        detect_keyword: The detection keyword
        box_style: Visualization style ('censor', 'bounding-box', or 'hitmarker')
    """
    height, width = frame.shape[:2]

    for box, keyword in detected_objects:
        try:
            # Convert normalized coordinates to pixel coordinates
            x1 = int(box[0] * width)
            y1 = int(box[1] * height)
            x2 = int(box[2] * width)
            y2 = int(box[3] * height)

            # Ensure coordinates are within frame boundaries
            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(0, min(x2, width - 1))
            y2 = max(0, min(y2, height - 1))

            # Only draw if box has reasonable size
            if x2 > x1 and y2 > y1:
                if box_style == "censor":
                    # Draw solid black rectangle
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), -1)
                elif box_style == "bounding-box":
                    # Draw red rectangle with thicker line
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

                    # Add label with background
                    label = detect_keyword  # Use exact capitalization
                    label_size = cv2.getTextSize(label, FONT, 0.7, 2)[0]
                    cv2.rectangle(
                        frame, (x1, y1 - 25), (x1 + label_size[0], y1), (0, 0, 255), -1
                    )
                    cv2.putText(
                        frame,
                        label,
                        (x1, y1 - 6),
                        FONT,
                        0.7,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                elif box_style == "hitmarker":
                    # Calculate center of the box
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2

                    # Draw hitmarker at the center
                    draw_hitmarker(frame, center_x, center_y)

                    # Optional: Add small label above hitmarker
                    label = detect_keyword  # Use exact capitalization
                    label_size = cv2.getTextSize(label, FONT, 0.5, 1)[0]
                    cv2.putText(
                        frame,
                        label,
                        (center_x - label_size[0] // 2, center_y - HITMARKER_SIZE - 5),
                        FONT,
                        0.5,
                        HITMARKER_COLOR,
                        1,
                        cv2.LINE_AA,
                    )
        except Exception as e:
            print(f"Error drawing {box_style} style box: {str(e)}")

    return frame


def filter_temporal_outliers(detections_dict):
    """Filter out extremely large detections that take up most of the frame.
    Only keeps detections that are reasonable in size.

    Args:
        detections_dict: Dictionary of {frame_number: [(box, keyword), ...]}
    """
    filtered_detections = {}

    for t, detections in detections_dict.items():
        # Only keep detections that aren't too large
        valid_detections = []
        for box, keyword in detections:
            # Calculate box size as percentage of frame
            width = box[2] - box[0]
            height = box[3] - box[1]
            area = width * height

            # If box is less than 90% of frame, keep it
            if area < 0.9:
                valid_detections.append((box, keyword))

        if valid_detections:
            filtered_detections[t] = valid_detections

    return filtered_detections


def describe_frames(
    video_path, model, tokenizer, detect_keyword, test_mode=False, rows=1, cols=1
):
    """Extract and detect objects in frames."""
    props = get_video_properties(video_path)
    fps = props["fps"]

    # If in test mode, only process first 3 seconds
    if test_mode:
        frame_count = min(int(fps * TEST_MODE_DURATION), props["frame_count"])
    else:
        frame_count = props["frame_count"]

    ad_detections = {}  # Store detection results by frame number

    print("Extracting frames and detecting objects...")
    video = cv2.VideoCapture(video_path)

    # Process every frame
    frame_count_processed = 0
    with tqdm(total=frame_count) as pbar:
        while frame_count_processed < frame_count:
            ret, frame = video.read()
            if not ret:
                break

            # Detect objects in the frame
            detected_objects = detect_ads_in_frame(
                model, tokenizer, frame, detect_keyword, rows=rows, cols=cols
            )

            # Store results for every frame, even if empty
            ad_detections[frame_count_processed] = detected_objects

            frame_count_processed += 1
            pbar.update(1)

    video.release()

    if frame_count_processed == 0:
        print("No frames could be read from video")
        return {}

    # Filter out only extremely large detections
    ad_detections = filter_temporal_outliers(ad_detections)
    return ad_detections


def create_detection_video(
    video_path,
    ad_detections,
    detect_keyword,
    output_path=None,
    ffmpeg_preset="medium",
    test_mode=False,
    box_style="censor",
):
    """Create video with detection boxes."""
    if output_path is None:
        # Create outputs directory if it doesn't exist
        outputs_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "outputs"
        )
        os.makedirs(outputs_dir, exist_ok=True)

        # Clean the detect_keyword for filename
        safe_keyword = "".join(
            x for x in detect_keyword if x.isalnum() or x in (" ", "_", "-")
        )
        safe_keyword = safe_keyword.replace(" ", "_")

        # Create output filename
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(
            outputs_dir, f"{box_style}_{safe_keyword}_{base_name}.mp4"
        )

    print(f"Will save output to: {output_path}")

    props = get_video_properties(video_path)
    fps, width, height = props["fps"], props["width"], props["height"]

    # If in test mode, only process first few seconds
    if test_mode:
        frame_count = min(int(fps * TEST_MODE_DURATION), props["frame_count"])
    else:
        frame_count = props["frame_count"]

    video = cv2.VideoCapture(video_path)

    # Create temp output path by adding _temp before the extension
    base, ext = os.path.splitext(output_path)
    temp_output = f"{base}_temp{ext}"

    out = cv2.VideoWriter(
        temp_output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    print("Creating detection video...")
    frame_count_processed = 0

    with tqdm(total=frame_count) as pbar:
        while frame_count_processed < frame_count:
            ret, frame = video.read()
            if not ret:
                break

            # Get detections for this exact frame
            if frame_count_processed in ad_detections:
                current_detections = ad_detections[frame_count_processed]
                if current_detections:
                    frame = draw_ad_boxes(
                        frame, current_detections, detect_keyword, box_style=box_style
                    )

            out.write(frame)
            frame_count_processed += 1
            pbar.update(1)

    video.release()
    out.release()

    # Convert to web-compatible format more efficiently
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                temp_output,
                "-c:v",
                "libx264",
                "-preset",
                ffmpeg_preset,
                "-crf",
                "23",
                "-movflags",
                "+faststart",  # Better web playback
                "-loglevel",
                "error",
                output_path,
            ],
            check=True,
        )

        os.remove(temp_output)  # Remove the temporary file

        if not os.path.exists(output_path):
            print(
                f"Warning: FFmpeg completed but output file not found at {output_path}"
            )
            return None

        return output_path

    except subprocess.CalledProcessError as e:
        print(f"Error running FFmpeg: {str(e)}")
        if os.path.exists(temp_output):
            os.remove(temp_output)
        return None


def process_video(
    video_path,
    detect_keyword,
    test_mode=False,
    ffmpeg_preset="medium",
    rows=1,
    cols=1,
    box_style="censor",
):
    """Process a single video file."""
    print(f"\nProcessing: {video_path}")
    print(f"Looking for: {detect_keyword}")

    # Load model
    print("Loading Moondream model...")
    model, tokenizer = load_moondream()

    # Process video - detect objects
    ad_detections = describe_frames(
        video_path, model, tokenizer, detect_keyword, test_mode, rows, cols
    )

    # Create video with detection boxes
    output_path = create_detection_video(
        video_path,
        ad_detections,
        detect_keyword,
        ffmpeg_preset=ffmpeg_preset,
        test_mode=test_mode,
        box_style=box_style,
    )

    if output_path is None:
        print("\nError: Failed to create output video")
        return None

    print(f"\nOutput saved to: {output_path}")
    return output_path


def main():
    """Process all videos in the inputs directory."""
    parser = argparse.ArgumentParser(
        description="Detect objects in videos using Moondream2"
    )
    parser.add_argument(
        "--test", action="store_true", help="Process only first 3 seconds of each video"
    )
    parser.add_argument(
        "--preset",
        choices=FFMPEG_PRESETS,
        default="medium",
        help="FFmpeg encoding preset (default: medium). Faster presets = lower quality",
    )
    parser.add_argument(
        "--detect",
        type=str,
        default="face",
        help='Object to detect in the video (default: face, use --detect "thing to detect" to override)',
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=1,
        help="Number of rows to split each frame into (default: 1)",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=1,
        help="Number of columns to split each frame into (default: 1)",
    )
    parser.add_argument(
        "--box-style",
        choices=["censor", "bounding-box", "hitmarker"],
        default="censor",
        help="Style of detection visualization (default: censor)",
    )
    args = parser.parse_args()

    input_dir = "inputs"
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    video_files = [
        f
        for f in os.listdir(input_dir)
        if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm"))
    ]

    if not video_files:
        print("No video files found in 'inputs' directory")
        return

    print(f"Found {len(video_files)} videos to process")
    print(f"Will detect: {args.detect}")
    if args.test:
        print("Running in test mode - processing only first 3 seconds of each video")
    print(f"Using FFmpeg preset: {args.preset}")
    print(f"Grid size: {args.rows}x{args.cols}")
    print(f"Box style: {args.box_style}")

    success_count = 0
    for video_file in video_files:
        video_path = os.path.join(input_dir, video_file)
        output_path = process_video(
            video_path,
            args.detect,
            test_mode=args.test,
            ffmpeg_preset=args.preset,
            rows=args.rows,
            cols=args.cols,
            box_style=args.box_style,
        )
        if output_path:
            success_count += 1

    print(
        f"\nProcessing complete. Successfully processed {success_count} out of {len(video_files)} videos."
    )


if __name__ == "__main__":
    main()
