import argparse
import datasets
import torch
import json
from tqdm import tqdm

from ..torch.config import MoondreamConfig
from ..torch.moondream import MoondreamModel
from ..torch.weights import load_weights_into_model
import numpy as np
from typing import List, Tuple


coco_classes = [
    "None",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "street sign",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "hat",
    "backpack",
    "umbrella",
    "shoe",
    "eye glasses",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "plate",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "mirror",
    "dining table",
    "window",
    "desk",
    "toilet",
    "door",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "blender",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
    "hair brush",
]

COCO_LABELS = {}

for i, c in enumerate(coco_classes):
    COCO_LABELS[i] = c


def calculate_iou(
    box1: Tuple[float, float, float, float], box2: Tuple[float, float, float, float]
) -> float:
    """Calculate IoU between two boxes (x1, y1, x2, y2 format)"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    return intersection / (box1_area + box2_area - intersection)


def calculate_map(
    ground_truth_boxes: List[List[Tuple[float, float, float, float]]],
    predicted_boxes: List[List[Tuple[float, float, float, float, float]]],
    iou_threshold: float = 0.5,
) -> float:
    """
    Calculate mAP for object detection

    Args:
        ground_truth_boxes: List of lists of ground truth boxes per image [(x1, y1, x2, y2)]
        predicted_boxes: List of lists of predicted boxes per image [(x1, y1, x2, y2, confidence)]
        iou_threshold: IoU threshold for considering a detection as correct

    Returns:
        mean Average Precision
    """
    total_precision = 0
    num_classes = len(ground_truth_boxes)

    for class_idx in range(num_classes):
        # Get all predictions and ground truths for this class
        gt_boxes = ground_truth_boxes[class_idx]
        pred_boxes = predicted_boxes[class_idx]

        # Sort predictions by confidence
        pred_boxes = sorted(pred_boxes, key=lambda x: x[4], reverse=True)

        # Initialize arrays for precision-recall calculation
        num_gt = len(gt_boxes)
        if num_gt == 0:
            continue

        tp = np.zeros(len(pred_boxes))
        fp = np.zeros(len(pred_boxes))
        gt_matched = [False] * num_gt

        # Match each prediction to ground truth
        for pred_idx, pred_box in enumerate(pred_boxes):
            max_iou = 0
            max_idx = -1

            # Find best matching ground truth box
            for gt_idx, gt_box in enumerate(gt_boxes):
                if gt_matched[gt_idx]:
                    continue

                iou = calculate_iou(pred_box[:4], gt_box)
                if iou > max_iou:
                    max_iou = iou
                    max_idx = gt_idx

            # If IoU exceeds threshold, count as true positive
            if max_iou >= iou_threshold:
                tp[pred_idx] = 1
                gt_matched[max_idx] = True
            else:
                fp[pred_idx] = 1

        # Calculate cumulative precision and recall
        cumsum_tp = np.cumsum(tp)
        cumsum_fp = np.cumsum(fp)
        recalls = cumsum_tp / num_gt
        precisions = cumsum_tp / (cumsum_tp + cumsum_fp)

        # Calculate average precision using all points
        ap = 0
        for t in np.arange(0, 1.1, 0.1):
            if np.sum(recalls >= t) == 0:
                p = 0
            else:
                p = np.max(precisions[recalls >= t])
            ap += p / 11

        total_precision += ap

    return total_precision / num_classes


def get_total_map(results_by_label, frequency_by_label):
    total_count = 0
    total_map = 0
    for results, frequency in zip(
        results_by_label.values(), frequency_by_label.values()
    ):
        cur_total_map = sum(results)
        total_map += cur_total_map
        total_count += frequency
    return total_map / total_count


def eval_coco_bbox(model, iou_threshold=0.5, debug=False):
    dataset = datasets.load_dataset(
        "moondream/coco-val-2017-bbox-cleaned", split="validation"
    )

    total = 0
    results_by_label = {}  # map to list of raw map results for each label
    frequency_by_label = {}  #  many images contain a given label

    for row in tqdm(dataset, disable=debug):
        width = row["image"].width
        height = row["image"].height
        total += 1

        objects = json.loads(row["objects"])

        gt_label_to_boxes = {}

        for bbox, label in zip(objects["bbox"], objects["label"]):
            if label not in gt_label_to_boxes:
                gt_label_to_boxes[label] = []
            x1, y1, w, h = bbox
            gt_label_to_boxes[label].append((x1, y1, x1 + w, y1 + h))

        unique_labels = [label for label in set(objects["label"])]

        for label in unique_labels:

            encoded_image = model.encode_image(row["image"])
            model_answer = model.detect(encoded_image, COCO_LABELS[label])["objects"]

            moondream_boxes = []

            for box in model_answer:
                moondream_boxes.append(
                    (
                        box["x_min"] * width,
                        box["y_min"] * height,
                        box["x_max"] * width,
                        box["y_max"] * height,
                        1.0,  # Using default confidence of 1.0
                    )
                )
            map_result = calculate_map(
                [gt_label_to_boxes[label]], [moondream_boxes], iou_threshold
            )
            if debug and map_result == 0:
                print(
                    f"0 Map result for index {total} and label {label} ({COCO_LABELS[label]})"
                )

            if label not in results_by_label:
                results_by_label[label] = []
            results_by_label[label].append(map_result)

            if label not in frequency_by_label:
                frequency_by_label[label] = 0
            frequency_by_label[label] += 1

        if total % 100 == 0:
            print(
                f"Total map: {get_total_map(results_by_label, frequency_by_label)*100:.2f}, ({total} images)"
            )

    return {
        "results_by_label": results_by_label,
        "frequency_by_label": frequency_by_label,
        "total_map": get_total_map(results_by_label, frequency_by_label),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    # This repo doesn't have moondream deps we need
    if torch.cuda.is_available():
        torch.set_default_device("cuda")
    elif torch.backends.mps.is_available():
        torch.set_default_device("mps")

    config = MoondreamConfig()
    model = MoondreamModel(config)
    load_weights_into_model(args.model, model)

    result = eval_coco_bbox(model, 0.5, args.debug)

    print(f"Overall MAP: {result['total_map']*100:.2f}")
