import argparse
import torch
import numpy as np
import datasets

from typing import List, Tuple
from tqdm import tqdm
from torch.utils.data import Dataset

from ..torch.config import MoondreamConfig
from ..torch.moondream import MoondreamModel
from ..torch.weights import load_weights_into_model

LABEL_NAME = "item"


def calculate_iou(
    box1: Tuple[float, float, float, float],
    box2: Tuple[float, float, float, float],
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
    Calculate mAP for object detection.

    Args:
        ground_truth_boxes: List (per image) of ground truth boxes [(x1, y1, x2, y2)]
        predicted_boxes: List (per image) of predicted boxes [(x1, y1, x2, y2, confidence)]
        iou_threshold: IoU threshold to consider a detection as correct

    Returns:
        Average Precision for the image.
    """
    total_ap = 0.0
    num_images = len(ground_truth_boxes)

    for gt_boxes, pred_boxes in zip(ground_truth_boxes, predicted_boxes):
        # Sort predictions by confidence descending
        pred_boxes = sorted(pred_boxes, key=lambda x: x[4], reverse=True)

        num_gt = len(gt_boxes)
        if num_gt == 0:
            continue

        tp = np.zeros(len(pred_boxes))
        fp = np.zeros(len(pred_boxes))
        gt_matched = [False] * num_gt

        for pred_idx, pred_box in enumerate(pred_boxes):
            max_iou = 0.0
            max_idx = -1
            for gt_idx, gt_box in enumerate(gt_boxes):
                if gt_matched[gt_idx]:
                    continue
                iou = calculate_iou(pred_box[:4], gt_box)
                if iou > max_iou:
                    max_iou = iou
                    max_idx = gt_idx
            if max_iou >= iou_threshold:
                tp[pred_idx] = 1
                gt_matched[max_idx] = True
            else:
                fp[pred_idx] = 1

        cumsum_tp = np.cumsum(tp)
        cumsum_fp = np.cumsum(fp)
        recalls = cumsum_tp / num_gt
        precisions = cumsum_tp / (cumsum_tp + cumsum_fp + 1e-6)

        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(recalls >= t) == 0:
                p = 0
            else:
                p = np.max(precisions[recalls >= t])
            ap += p / 11.0
        total_ap += ap

    return total_ap / num_images if num_images > 0 else 0.0


class Shopping(Dataset):

    def __init__(self):
        self.dataset: datasets.Dataset = datasets.load_dataset(
            "moondream/shopping",
            split="test",
        ).cast_column("image", datasets.Image())
        self.dataset = self.dataset.shuffle(87)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset[idx]
        image = row["image"].convert("RGB")
        boxes = row["boxes"]
        width = row["width"]
        height = row["height"]
        image_id = row["image_id"]

        gt_boxes = [(box[0], box[1], box[0] + box[2], box[1] + box[3]) for box in boxes]
        return {
            "image": image,
            "gt_boxes": gt_boxes,
            "width": width,
            "height": height,
            "image_id": image_id,
        }


def eval_shopping_map(model, iou_threshold=0.5, debug=False):
    dataset = Shopping()
    total = 0
    aps = []

    pbar = tqdm(dataset, desc="COCO mAP")
    for data in pbar:
        image = data["image"]
        width = data["width"]
        height = data["height"]
        gt_boxes = data["gt_boxes"]
        total += 1

        encoded_image = model.encode_image(image)
        model_answer = model.detect(encoded_image, LABEL_NAME)["objects"]

        pred_boxes = []
        for box in model_answer:
            pred_boxes.append(
                (
                    box["x_min"] * width,
                    box["y_min"] * height,
                    box["x_max"] * width,
                    box["y_max"] * height,
                    1.0,
                )
            )
        ap = calculate_map([gt_boxes], [pred_boxes], iou_threshold)
        if debug and ap == 0:
            print(f"0 mAP for image {data['image_id']} ({LABEL_NAME})")
        aps.append(ap)

        if total % 10 == 0:
            current_map = np.mean(aps) * 100
            print(f"Processed {total} images, current overall mAP: {current_map:.2f}")

    overall_map = np.mean(aps)
    return {"total_map": overall_map}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, required=True, help="Path to model weights"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--gpu", type=int, default=7, help="GPU device id to use")
    args = parser.parse_args()

    # Set the device based on the specified GPU id
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    config = MoondreamConfig()
    model = MoondreamModel(config)
    load_weights_into_model(args.model, model)
    model.compile()
    model.to(device)

    result = eval_shopping_map(model, iou_threshold=0.5, debug=args.debug)
    print(f"Overall mAP: {result['total_map'] * 100:.2f}")
