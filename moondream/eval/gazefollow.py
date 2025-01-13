import torch
import datasets
import math

from tqdm import tqdm

from ..torch.config import MoondreamConfig
from ..torch.moondream import MoondreamModel
from ..torch.weights import load_weights_into_model


def eval_gazefollow(model, debug=False):
    dataset = datasets.load_dataset("vikhyatk/gazefollow", split="test")

    mean_l2_error = []
    min_l2_error = []
    total = 0

    for i, row in tqdm(enumerate(dataset), total=len(dataset)):
        heads = []

        for gaze in row["gazes"]:
            head_bbox = gaze["head_bbox"]  # xmin, ymin, xmax, ymax
            eye_coord = (gaze["eye"]["x"], gaze["eye"]["y"])
            mean_target_gaze = (gaze["gaze"]["x"], gaze["gaze"]["y"])

            # Check if a head already exists with the same approximate bbox.
            # If so, use that head instead of creating a new one.
            for head in heads:
                if (
                    abs(head["head_bbox"]["xmin"] - head_bbox["xmin"]) < 0.001
                    and abs(head["head_bbox"]["xmax"] - head_bbox["xmax"]) < 0.001
                    and abs(head["head_bbox"]["ymin"] - head_bbox["ymin"]) < 0.001
                    and abs(head["head_bbox"]["ymax"] - head_bbox["ymax"]) < 0.001
                ):
                    head["gazes"].append(mean_target_gaze)
                    break
            else:
                heads.append(
                    {
                        "head_bbox": head_bbox,
                        "eye_coord": eye_coord,
                        "gazes": [mean_target_gaze],
                    }
                )

        for head in heads:
            pred_gaze = model.detect_gaze(
                row["image"],
                eye=head["eye_coord"],
                face={
                    "x_min": head["head_bbox"]["xmin"],
                    "y_min": head["head_bbox"]["ymin"],
                    "x_max": head["head_bbox"]["xmax"],
                    "y_max": head["head_bbox"]["ymax"],
                },
                unstable_settings={"force_detect": True},
            )["gaze"]

            mean_target_gaze = (
                sum(gaze[0] for gaze in head["gazes"]) / len(head["gazes"]),
                sum(gaze[1] for gaze in head["gazes"]) / len(head["gazes"]),
            )
            mean_l2 = math.sqrt(
                (mean_target_gaze[0] - pred_gaze["x"]) ** 2
                + (mean_target_gaze[1] - pred_gaze["y"]) ** 2
            )
            min_l2 = min(
                math.sqrt(
                    (target_gaze[0] - pred_gaze["x"]) ** 2
                    + (target_gaze[1] - pred_gaze["y"]) ** 2
                )
                for target_gaze in head["gazes"]
            )

            mean_l2_error.append(mean_l2)
            min_l2_error.append(min_l2)
            total += 1

            if i % 100 == 0 and debug:
                print("Mean L2 error:", sum(mean_l2_error) / total)
                print("Min L2 error:", sum(min_l2_error) / total)

    return {
        "mean_l2": sum(mean_l2_error) / total,
        "min_l2": sum(min_l2_error) / total,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)

    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.set_default_device("cuda")
    elif torch.backends.mps.is_available():
        torch.set_default_device("mps")

    config = MoondreamConfig()
    model = MoondreamModel(config)
    load_weights_into_model(args.model, model)

    results = eval_gazefollow(model, debug=args.debug)

    print(f"Mean L2 error: {results['mean_l2']:.4f}")
    print(f"Min L2 error: {results['min_l2']:.4f}")
