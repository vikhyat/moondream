import argparse
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset

from ..torch.config import MoondreamConfig
from ..torch.moondream import MoondreamModel
from ..torch.weights import load_weights_into_model


Box = Tuple[float, float, float, float]  # (x1, y1, x2, y2) – in proportion form


def iou(a: Box, b: Box) -> float:
    """Corner-format IoU. Returns 0 when either box has zero area."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)

    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def match(gt: List[Box], pr: List[Box], iou_thr: float) -> Tuple[int, int, int]:
    """
    Greedy one-to-one matching with no confidences.
    Predictions are taken in the order produced by the model.
    """
    tp = fp = 0
    seen = [False] * len(gt)

    for p in pr:
        best, best_i = 0.0, -1
        for i, g in enumerate(gt):
            if seen[i]:
                continue
            iou_ = iou(p, g)
            if iou_ > best:
                best, best_i = iou_, i
        if best >= iou_thr:
            tp += 1
            seen[best_i] = True
        else:
            fp += 1

    fn = len(gt) - tp
    return tp, fp, fn


class WasteDetection(torch.utils.data.Dataset):
    def __init__(self, name: str = "moondream/waste_detection", split: str = "test"):
        self.ds = load_dataset(name, split=split)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int) -> Dict:
        s = self.ds[idx]
        img = (
            s["image"]
            if isinstance(s["image"], Image.Image)
            else Image.fromarray(s["image"])
        )
        W, H = float(s.get("width", img.width)), float(s.get("height", img.height))

        lbl_to_boxes = defaultdict(list)
        for (xc, yc, bw, bh), lbl in zip(s["boxes"], s["labels"]):
            x1 = xc - bw / 2
            y1 = yc - bh / 2
            x2 = xc + bw / 2
            y2 = yc + bh / 2
            lbl_to_boxes[lbl].append((x1, y1, x2, y2))

        return {"image": img, "gt": lbl_to_boxes, "W": W, "H": H}


def evaluate(
    model: MoondreamModel,
    iou_thr: float,
    debug: bool,
):
    ds = WasteDetection(split="test")
    TP = FP = FN = 0

    for s in tqdm(ds, disable=debug, desc="Waste"):
        img, gts = s["image"], s["gt"]
        enc = model.encode_image(img)

        for lbl, gt_boxes in gts.items():
            preds: List[Box] = [
                (
                    o["x_min"],
                    o["y_min"],
                    o["x_max"],
                    o["y_max"],
                )
                for o in model.detect(enc, lbl)["objects"]
            ]
            tp, fp, fn = match(gt_boxes, preds, iou_thr)
            TP += tp
            FP += fp
            FN += fn

    prec = TP / (TP + FP) if TP + FP else 0.0
    rec = TP / (TP + FN) if TP + FN else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(precision=prec, recall=rec, f1=f1, tp=TP, fp=FP, fn=FN)


def load_model(path: str, device: torch.device) -> MoondreamModel:
    cfg = MoondreamConfig()
    model = MoondreamModel(cfg)
    load_weights_into_model(path, model)
    model.compile()
    model.to(device)
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--iou_thr", type=float, default=0.5)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = load_model(args.model, device)
    res = evaluate(model, args.iou_thr, args.debug)

    print(f"Precision: {res['precision']*100:.2f}%")
    print(f"Recall: {res['recall']*100:.2f}%")
    print(f"F1 Score:  {res['f1']*100:.2f}%")
    print(f"TP: {res['tp']}  FP: {res['fp']}  FN: {res['fn']}")


if __name__ == "__main__":
    """
    Eval to accompany finetune_region.py.
    """
    main()
