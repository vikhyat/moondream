#!/usr/bin/env python3
import cv2, os, subprocess, argparse
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, SamModel, SamProcessor
from tqdm import tqdm
import numpy as np
from datetime import datetime
import colorsys
import random
from deep_sort_integration import DeepSORTTracker
from scenedetect import detect, ContentDetector
from functools import lru_cache

# Constants
DEFAULT_TEST_MODE_DURATION = 3  # Process only first 3 seconds in test mode by default
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

# SAM parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model variables as None
sam_model = None
sam_processor = None
slimsam_model = None
slimsam_processor = None

@lru_cache(maxsize=2)  # Cache both regular and slim SAM models
def get_sam_model(slim=False):
    """Get cached SAM model and processor."""
    global sam_model, sam_processor, slimsam_model, slimsam_processor
    
    if slim:
        if slimsam_model is None:
            print("Loading SlimSAM model for the first time...")
            slimsam_model = SamModel.from_pretrained("nielsr/slimsam-50-uniform").to(device)
            slimsam_processor = SamProcessor.from_pretrained("nielsr/slimsam-50-uniform")
        return slimsam_model, slimsam_processor
    else:
        if sam_model is None:
            print("Loading SAM model for the first time...")
            sam_model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
            sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
        return sam_model, sam_processor

def load_sam_model(slim=False):
    """Load SAM model and processor with caching."""
    return get_sam_model(slim=slim)

def generate_color_pair():
    """Generate a generic light blue and dark blue color pair for SAM visualization."""
    dark_rgb = [0, 0, 139]  # Dark blue
    light_rgb = [173, 216, 230]  # Light blue
    return dark_rgb, light_rgb

def create_mask_overlay(image, masks, points=None, labels=None):
    """Create a mask overlay with contours for multiple SAM visualizations.
    
    Args:
        image: PIL Image to overlay masks on
        masks: List of binary masks or single mask
        points: Optional list of (x,y) points for labels
        labels: Optional list of label strings for each point
    """
    # Convert single mask to list for uniform processing
    if not isinstance(masks, list):
        masks = [masks]
    
    # Create empty overlays
    overlay = np.zeros((*image.size[::-1], 4), dtype=np.uint8)
    outline = np.zeros((*image.size[::-1], 4), dtype=np.uint8)
    
    # Process each mask
    for i, mask in enumerate(masks):
        # Convert binary mask to uint8
        mask_uint8 = (mask > 0).astype(np.uint8)
        
        # Dilation to fill gaps
        kernel = np.ones((5, 5), np.uint8)
        mask_dilated = cv2.dilate(mask_uint8, kernel, iterations=1)
        
        # Find contours of the dilated mask
        contours, _ = cv2.findContours(mask_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Generate random color pair for this segmentation
        dark_color, light_color = generate_color_pair()
        
        # Add to the overlays
        overlay[mask_dilated > 0] = [*light_color, 90]  # Light color with 35% opacity
        cv2.drawContours(outline, contours, -1, (*dark_color, 255), 2)  # Dark color outline
    
    # Convert to PIL images
    mask_overlay = Image.fromarray(overlay, 'RGBA')
    outline_overlay = Image.fromarray(outline, 'RGBA')
    
    # Composite the layers
    result = image.convert('RGBA')
    result.paste(mask_overlay, (0, 0), mask_overlay)
    result.paste(outline_overlay, (0, 0), outline_overlay)
    
    # Add labels if provided
    if points and labels:
        result_array = np.array(result)
        for (x, y), label in zip(points, labels):
            label_size = cv2.getTextSize(label, FONT, 0.5, 1)[0]
            cv2.putText(
                result_array,
                label,
                (int(x - label_size[0] // 2), int(y - 20)),
                FONT,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        result = Image.fromarray(result_array)
    
    return result

def process_sam_detection(image, center_x, center_y, slim=False):
    """Process a single detection point with SAM.
    
    Returns:
        tuple: (mask, result_pil) where mask is the binary mask and result_pil is the visualization
    """
    if not isinstance(image, Image.Image):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Get appropriate model from cache
    model, processor = get_sam_model(slim)
    
    # Process the image with SAM
    inputs = processor(
        image,
        input_points=[[[center_x, center_y]]],
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    mask = processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu()
    )[0][0][0].numpy()
    
    # Create the visualization
    result = create_mask_overlay(image, mask)
    return mask, result

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


def is_valid_bounding_box(bounding_box):
    """Check if bounding box coordinates are reasonable."""
    x1, y1, x2, y2 = bounding_box
    width = x2 - x1
    height = y2 - y1

    # Reject boxes that are too large (over 90% of frame in both dimensions)
    if width > 0.9 and height > 0.9:
        return False

    # Reject boxes that are too small (less than 1% of frame)
    if width < 0.01 or height < 0.01:
        return False

    return True


def split_frame_into_grid(frame, grid_rows, grid_cols):
    """Split a frame into a grid of tiles."""
    height, width = frame.shape[:2]
    tile_height = height // grid_rows
    tile_width = width // grid_cols
    tiles = []
    tile_positions = []

    for i in range(grid_rows):
        for j in range(grid_cols):
            y1 = i * tile_height
            y2 = (i + 1) * tile_height if i < grid_rows - 1 else height
            x1 = j * tile_width
            x2 = (j + 1) * tile_width if j < grid_cols - 1 else width

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


def detect_objects_in_frame(model, tokenizer, image, target_object, grid_rows=1, grid_cols=1):
    """Detect specified objects in a frame using grid-based analysis."""
    if grid_rows == 1 and grid_cols == 1:
        return detect_objects_in_frame_single(model, tokenizer, image, target_object)

    # Convert numpy array to PIL Image if needed
    if not isinstance(image, Image.Image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Split frame into tiles
    tiles, tile_positions = split_frame_into_grid(image, grid_rows, grid_cols)

    # Process each tile
    tile_detections = []
    for tile, tile_pos in zip(tiles, tile_positions):
        # Convert tile to PIL Image
        tile_pil = Image.fromarray(tile)

        # Detect objects in tile
        response = model.detect(tile_pil, target_object)

        if response and "objects" in response and response["objects"]:
            objects = response["objects"]
            tile_objects = []

            for obj in objects:
                if all(k in obj for k in ["x_min", "y_min", "x_max", "y_max"]):
                    box = [obj["x_min"], obj["y_min"], obj["x_max"], obj["y_max"]]

                    if is_valid_bounding_box(box):
                        # Convert tile coordinates to frame coordinates
                        frame_box = convert_tile_coords_to_frame(
                            box, tile_pos, image.shape
                        )
                        tile_objects.append((frame_box, target_object))

            if tile_objects:  # Only append if we found valid objects
                tile_detections.append(tile_objects)

    # Merge detections from all tiles
    merged_detections = merge_tile_detections(tile_detections)
    return merged_detections


def detect_objects_in_frame_single(model, tokenizer, image, target_object):
    """Single-frame detection function."""
    detected_objects = []

    # Convert numpy array to PIL Image if needed
    if not isinstance(image, Image.Image):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Detect objects
    response = model.detect(image, target_object)

    # Check if we have valid objects
    if response and "objects" in response and response["objects"]:
        objects = response["objects"]

        for obj in objects:
            if all(k in obj for k in ["x_min", "y_min", "x_max", "y_max"]):
                box = [obj["x_min"], obj["y_min"], obj["x_max"], obj["y_max"]]
                # If box is valid (not full-frame), add it
                if is_valid_bounding_box(box):
                    detected_objects.append((box, target_object))

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


def draw_ad_boxes(frame, detected_objects, detect_keyword, model, box_style="censor"):
    height, width = frame.shape[:2]

    points = []
    # Only get points if we need them for hitmarker or SAM styles
    if box_style in ["hitmarker", "sam", "sam-fast"]:
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        try:
            point_response = model.point(frame_pil, detect_keyword)
            
            if isinstance(point_response, dict) and 'points' in point_response:
                points = point_response['points']
        except Exception as e:
            print(f"Error during point detection: {str(e)}")
            points = []

    # Only load SAM models and process points if we're using SAM styles and have points
    if box_style in ["sam", "sam-fast"] and points:
        # Start with the original PIL image
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Collect all masks and points
        all_masks = []
        point_coords = []
        point_labels = []
        
        for point in points:
            try:
                center_x = int(float(point["x"]) * width)
                center_y = int(float(point["y"]) * height)

                # Get mask and visualization
                mask, _ = process_sam_detection(frame_pil, center_x, center_y, slim=(box_style == "sam-fast"))
                
                # Collect mask and point data
                all_masks.append(mask)
                point_coords.append((center_x, center_y))
                point_labels.append(detect_keyword)
                
            except Exception as e:
                print(f"Error processing individual SAM point: {str(e)}")
                print(f"Point data: {point}")
        
        if all_masks:
            # Create final visualization with all masks
            result_pil = create_mask_overlay(frame_pil, all_masks, point_coords, point_labels)
            frame = cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGB2BGR)

    # Process other visualization styles
    for detection in detected_objects:
        try:
            # Handle both tracked and untracked detections
            if len(detection) == 3:  # Tracked detection with ID
                box, keyword, track_id = detection
            else:  # Regular detection without tracking
                box, keyword = detection
                track_id = None

            x1 = int(box[0] * width)
            y1 = int(box[1] * height)
            x2 = int(box[2] * width)
            y2 = int(box[3] * height)

            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(0, min(x2, width - 1))
            y2 = max(0, min(y2, height - 1))

            if x2 > x1 and y2 > y1:
                if box_style == "censor":
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), -1)
                elif box_style == "bounding-box":
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

                    label = f"{detect_keyword}" if track_id is not None else detect_keyword
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
                elif box_style == "fuzzy-blur":
                    # Extract ROI
                    roi = frame[y1:y2, x1:x2]
                    # Apply Gaussian blur with much larger kernel for intense blur
                    blurred_roi = cv2.GaussianBlur(roi, (125, 125), 0)
                    # Replace original ROI with blurred version
                    frame[y1:y2, x1:x2] = blurred_roi
                elif box_style == "pixelated-blur":
                    # Extract ROI
                    roi = frame[y1:y2, x1:x2]
                    # Pixelate by resizing down and up
                    h, w = roi.shape[:2]
                    temp = cv2.resize(roi, (10, 10), interpolation=cv2.INTER_LINEAR)
                    pixelated = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
                    # Mix up the pixelated frame slightly by adding random noise
                    noise = np.random.randint(0, 50, (h, w, 3), dtype=np.uint8)
                    pixelated = cv2.add(pixelated, noise)
                    # Apply stronger Gaussian blur to smooth edges
                    blurred_pixelated = cv2.GaussianBlur(pixelated, (15, 15), 0)
                    # Replace original ROI
                    frame[y1:y2, x1:x2] = blurred_pixelated
                elif box_style == "obfuscated-pixel":
                    # Calculate expansion amount based on 10% of object dimensions
                    box_width = x2 - x1
                    box_height = y2 - y1
                    expand_x = int(box_width * 0.10)
                    expand_y = int(box_height * 0.10)
                    
                    # Expand the bounding box by 10% in all directions
                    x1_expanded = max(0, x1 - expand_x)
                    y1_expanded = max(0, y1 - expand_y)
                    x2_expanded = min(width - 1, x2 + expand_x)
                    y2_expanded = min(height - 1, y2 + expand_y)
                    
                    # Extract ROI with much larger padding for true background sampling
                    padding = 100  # Much larger padding to get true background
                    y1_pad = max(0, y1_expanded - padding)
                    y2_pad = min(height, y2_expanded + padding)
                    x1_pad = max(0, x1_expanded - padding)
                    x2_pad = min(width, x2_expanded + padding)
                    
                    # Get the padded region including background
                    padded_roi = frame[y1_pad:y2_pad, x1_pad:x2_pad]
                    
                    # Create mask that excludes a larger region around the detection
                    h, w = y2_expanded - y1_expanded, x2_expanded - x1_expanded
                    bg_mask = np.ones(padded_roi.shape[:2], dtype=bool)
                    
                    # Exclude a larger region around the detection from background sampling
                    exclusion_padding = 50  # Area to exclude around detection
                    exclude_y1 = padding - exclusion_padding
                    exclude_y2 = padding + h + exclusion_padding
                    exclude_x1 = padding - exclusion_padding
                    exclude_x2 = padding + w + exclusion_padding
                    
                    # Make sure exclusion coordinates are valid
                    exclude_y1 = max(0, exclude_y1)
                    exclude_y2 = min(padded_roi.shape[0], exclude_y2)
                    exclude_x1 = max(0, exclude_x1)
                    exclude_x2 = min(padded_roi.shape[1], exclude_x2)
                    
                    # Mark the exclusion zone in the mask
                    bg_mask[exclude_y1:exclude_y2, exclude_x1:exclude_x2] = False
                    
                    # If we have enough background pixels, calculate average color
                    if np.any(bg_mask):
                        bg_color = np.mean(padded_roi[bg_mask], axis=0).astype(np.uint8)
                    else:
                        # Fallback to edges if we couldn't get enough background
                        edge_samples = np.concatenate([
                            padded_roi[0],  # Top edge
                            padded_roi[-1],  # Bottom edge
                            padded_roi[:, 0],  # Left edge
                            padded_roi[:, -1]  # Right edge
                        ])
                        bg_color = np.mean(edge_samples, axis=0).astype(np.uint8)
                    
                    # Create base pixelated version (of the expanded region)
                    temp = cv2.resize(frame[y1_expanded:y2_expanded, x1_expanded:x2_expanded], 
                                   (6, 6), interpolation=cv2.INTER_LINEAR)
                    pixelated = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
                    
                    # Blend heavily towards background color
                    blend_factor = 0.9  # Much stronger blend with background
                    blended = cv2.addWeighted(
                        pixelated, 1 - blend_factor,
                        np.full((h, w, 3), bg_color, dtype=np.uint8), blend_factor,
                        0
                    )
                    
                    # Replace original ROI with blended version (using expanded coordinates)
                    frame[y1_expanded:y2_expanded, x1_expanded:x2_expanded] = blended
                elif box_style == "intense-pixelated-blur":
                    # Expand the bounding box by pixels in all directions
                    x1_expanded = max(0, x1 - 15)
                    y1_expanded = max(0, y1 - 15)
                    x2_expanded = min(width - 1, x2 + 25)
                    y2_expanded = min(height - 1, y2 + 25)

                    # Extract ROI
                    roi = frame[y1_expanded:y2_expanded, x1_expanded:x2_expanded]
                    # Pixelate by resizing down and up
                    h, w = roi.shape[:2]
                    temp = cv2.resize(roi, (10, 10), interpolation=cv2.INTER_LINEAR)
                    pixelated = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
                    # Mix up the pixelated frame slightly by adding random noise
                    noise = np.random.randint(0, 50, (h, w, 3), dtype=np.uint8)
                    pixelated = cv2.add(pixelated, noise)
                    # Apply stronger Gaussian blur to smooth edges
                    blurred_pixelated = cv2.GaussianBlur(pixelated, (15, 15), 0)
                    # Replace original ROI
                    frame[y1_expanded:y2_expanded, x1_expanded:x2_expanded] = blurred_pixelated
                elif box_style == "hitmarker":
                    if points:
                        for point in points:
                            try:
                                print(f"Processing point: {point}")
                                center_x = int(float(point["x"]) * width)
                                center_y = int(float(point["y"]) * height)
                                print(f"Converted coordinates: ({center_x}, {center_y})")

                                draw_hitmarker(frame, center_x, center_y)

                                label = f"{detect_keyword}" if track_id is not None else detect_keyword
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
                                print(f"Error processing individual point: {str(e)}")
                                print(f"Point data: {point}")

        except Exception as e:
            print(f"Error drawing {box_style} style box: {str(e)}")
            print(f"Box data: {box}")
            print(f"Keyword: {keyword}")

    return frame


def filter_temporal_outliers(detections_dict):
    """Filter out extremely large detections that take up most of the frame.
    Only keeps detections that are reasonable in size.

    Args:
        detections_dict: Dictionary of {frame_number: [(box, keyword, track_id), ...]}
    """
    filtered_detections = {}

    for t, detections in detections_dict.items():
        # Only keep detections that aren't too large
        valid_detections = []
        for detection in detections:
            # Handle both tracked and untracked detections
            if len(detection) == 3:  # Tracked detection with ID
                box, keyword, track_id = detection
            else:  # Regular detection without tracking
                box, keyword = detection
                track_id = None

            # Calculate box size as percentage of frame
            width = box[2] - box[0]
            height = box[3] - box[1]
            area = width * height

            # If box is less than 90% of frame, keep it
            if area < 0.9:
                if track_id is not None:
                    valid_detections.append((box, keyword, track_id))
                else:
                    valid_detections.append((box, keyword))

        if valid_detections:
            filtered_detections[t] = valid_detections

    return filtered_detections


def describe_frames(video_path, model, tokenizer, detect_keyword, test_mode=False, test_duration=DEFAULT_TEST_MODE_DURATION, grid_rows=1, grid_cols=1):
    """Extract and detect objects in frames."""
    props = get_video_properties(video_path)
    fps = props["fps"]

    # Initialize DeepSORT tracker
    tracker = DeepSORTTracker()

    # If in test mode, only process first N seconds
    if test_mode:
        frame_count = min(int(fps * test_duration), props["frame_count"])
    else:
        frame_count = props["frame_count"]

    ad_detections = {}  # Store detection results by frame number

    print("Extracting frames and detecting objects...")
    video = cv2.VideoCapture(video_path)

    # Detect scenes first
    scenes = detect(video_path, scene_detector)
    scene_changes = set(end.get_frames() for _, end in scenes)
    print(f"Detected {len(scenes)} scenes")

    frame_count_processed = 0
    with tqdm(total=frame_count) as pbar:
        while frame_count_processed < frame_count:
            ret, frame = video.read()
            if not ret:
                break

            # Check if current frame is a scene change
            if frame_count_processed in scene_changes:
                # Detect objects in the frame
                detected_objects = detect_objects_in_frame(
                    model, tokenizer, frame, detect_keyword, grid_rows=grid_rows, grid_cols=grid_cols
                )

            # Update tracker with current detections
            tracked_objects = tracker.update(frame, detected_objects)

            # Store results for every frame, even if empty
            ad_detections[frame_count_processed] = tracked_objects

            frame_count_processed += 1
            pbar.update(1)

    video.release()

    if frame_count_processed == 0:
        print("No frames could be read from video")
        return {}

    return ad_detections


def create_detection_video(
    video_path,
    ad_detections,
    detect_keyword,
    model,
    output_path=None,
    ffmpeg_preset="medium",
    test_mode=False,
    test_duration=DEFAULT_TEST_MODE_DURATION,
    box_style="censor",
):
    """Create video with detection boxes while preserving audio."""
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
        frame_count = min(int(fps * test_duration), props["frame_count"])
        print(f"Test mode enabled: Processing first {test_duration} seconds ({frame_count} frames)")
    else:
        frame_count = props["frame_count"]
        print("Full video mode: Processing entire video")

    video = cv2.VideoCapture(video_path)

    # Create temp output path by adding _temp before the extension
    base, ext = os.path.splitext(output_path)
    temp_output = f"{base}_temp{ext}"
    temp_audio = f"{base}_audio.aac"  # Temporary audio file

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
                        frame, current_detections, detect_keyword, model, box_style=box_style
                    )

            out.write(frame)
            frame_count_processed += 1
            pbar.update(1)

    video.release()
    out.release()

    # Extract audio from original video
    try:
        if test_mode:
            # In test mode, extract only the required duration of audio
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    video_path,
                    "-t",
                    str(test_duration),
                    "-vn",  # No video
                    "-acodec",
                    "copy",
                    temp_audio,
                ],
                check=True,
            )
        else:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    video_path,
                    "-vn",  # No video
                    "-acodec",
                    "copy",
                    temp_audio,
                ],
                check=True,
            )
    except subprocess.CalledProcessError as e:
        print(f"Error extracting audio: {str(e)}")
        if os.path.exists(temp_output):
            os.remove(temp_output)
        return None

    # Merge processed video with original audio
    try:
        # Base FFmpeg command
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-i",
            temp_output,
            "-i",
            temp_audio,
            "-c:v",
            "libx264",
            "-preset",
            ffmpeg_preset,
            "-crf",
            "23",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",  # Better web playback
        ]

        if test_mode:
            # In test mode, ensure output duration matches test_duration
            ffmpeg_cmd.extend([
                "-t",
                str(test_duration),
                "-shortest"  # Ensure output duration matches shortest input
            ])

        ffmpeg_cmd.extend([
            "-loglevel",
            "error",
            output_path
        ])

        subprocess.run(ffmpeg_cmd, check=True)

        # Clean up temporary files
        os.remove(temp_output)
        os.remove(temp_audio)

        if not os.path.exists(output_path):
            print(
                f"Warning: FFmpeg completed but output file not found at {output_path}"
            )
            return None

        return output_path

    except subprocess.CalledProcessError as e:
        print(f"Error merging audio with video: {str(e)}")
        if os.path.exists(temp_output):
            os.remove(temp_output)
        if os.path.exists(temp_audio):
            os.remove(temp_audio)
        return None


def process_video(
    video_path,
    target_object,
    test_mode=False,
    test_duration=DEFAULT_TEST_MODE_DURATION,
    ffmpeg_preset="medium",
    grid_rows=1,
    grid_cols=1,
    box_style="censor",
):
    """Process a video to detect and visualize specified objects."""
    try:
        print(f"\nProcessing: {video_path}")
        print(f"Looking for: {target_object}")

        # Load model
        print("Loading Moondream model...")
        model, tokenizer = load_moondream()

        # Get video properties
        props = get_video_properties(video_path)
        
        # Initialize scene detector with ContentDetector
        scene_detector = ContentDetector(threshold=30.0)  # Adjust threshold as needed
        
        # Initialize DeepSORT tracker
        tracker = DeepSORTTracker()

        # If in test mode, only process first N seconds
        if test_mode:
            frame_count = min(int(props["fps"] * test_duration), props["frame_count"])
        else:
            frame_count = props["frame_count"]

        ad_detections = {}  # Store detection results by frame number

        print("Extracting frames and detecting objects...")
        video = cv2.VideoCapture(video_path)

        # Detect scenes first
        scenes = detect(video_path, scene_detector)
        scene_changes = set(end.get_frames() for _, end in scenes)
        print(f"Detected {len(scenes)} scenes")

        frame_count_processed = 0
        with tqdm(total=frame_count) as pbar:
            while frame_count_processed < frame_count:
                ret, frame = video.read()
                if not ret:
                    break

                # Check if current frame is a scene change
                if frame_count_processed in scene_changes:
                    print(f"Scene change detected at frame {frame_count_processed}. Resetting tracker.")
                    tracker.reset()

                # Detect objects in the frame
                detected_objects = detect_objects_in_frame(
                    model, tokenizer, frame, target_object, grid_rows=grid_rows, grid_cols=grid_cols
                )

                # Update tracker with current detections
                tracked_objects = tracker.update(frame, detected_objects)

                # Store results for every frame, even if empty
                ad_detections[frame_count_processed] = tracked_objects

                frame_count_processed += 1
                pbar.update(1)

        video.release()

        if frame_count_processed == 0:
            print("No frames could be read from video")
            return {}

        # Apply filtering
        filtered_ad_detections = filter_temporal_outliers(ad_detections)
        
        # Build detection data structure
        detection_data = {
            "video_metadata": {
                "file_name": os.path.basename(video_path),
                "fps": props["fps"],
                "width": props["width"],
                "height": props["height"],
                "total_frames": props["frame_count"],
                "duration_sec": props["frame_count"] / props["fps"],
                "detect_keyword": target_object,
                "test_mode": test_mode,
                "grid_size": f"{grid_rows}x{grid_cols}",
                "box_style": box_style,
                "timestamp": datetime.now().isoformat()
            },
            "frame_detections": [
                {
                    "frame": frame_num,
                    "timestamp": frame_num / props["fps"],
                    "objects": [
                        {
                            "keyword": kw,
                            "bbox": list(box),  # Convert numpy array to list if needed
                            "track_id": track_id if len(detection) == 3 else None
                        }
                        for detection in filtered_ad_detections.get(frame_num, [])
                        for box, kw, *track_id in [detection]  # Unpack detection tuple, track_id will be empty list if not present
                    ]
                }
                for frame_num in range(props["frame_count"] if not test_mode else min(int(props["fps"] * test_duration), props["frame_count"]))
            ]
        }
        
        # Save filtered data
        outputs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
        os.makedirs(outputs_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        json_path = os.path.join(outputs_dir, f"{box_style}_{target_object}_{base_name}_detections.json")
        
        from persistence import save_detection_data
        if not save_detection_data(detection_data, json_path):
            print("Warning: Failed to save detection data")

        # Create video with filtered data
        output_path = create_detection_video(
            video_path,
            filtered_ad_detections,
            target_object,
            model,
            ffmpeg_preset=ffmpeg_preset,
            test_mode=test_mode,
            test_duration=test_duration,
            box_style=box_style,
        )

        if output_path is None:
            print("\nError: Failed to create output video")
            return None

        print(f"\nOutput saved to: {output_path}")
        print(f"Detection data saved to: {json_path}")
        return output_path
        
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Process all videos in the inputs directory."""
    parser = argparse.ArgumentParser(
        description="Detect objects in videos using Moondream2"
    )
    parser.add_argument(
        "--test", action="store_true", help="Process only first 3 seconds of each video"
    )
    parser.add_argument(
        "--test-duration",
        type=int,
        default=DEFAULT_TEST_MODE_DURATION,
        help=f"Number of seconds to process in test mode (default: {DEFAULT_TEST_MODE_DURATION})"
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
        choices=["censor", "bounding-box", "hitmarker", "sam", "sam-fast", "fuzzy-blur", 
                "pixelated-blur", "intense-pixelated-blur", "obfuscated-pixel"],
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
            test_duration=args.test_duration,
            ffmpeg_preset=args.preset,
            grid_rows=args.rows,
            grid_cols=args.cols,
            box_style=args.box_style,
        )
        if output_path:
            success_count += 1

    print(
        f"\nProcessing complete. Successfully processed {success_count} out of {len(video_files)} videos."
    )


if __name__ == "__main__":
    main()
