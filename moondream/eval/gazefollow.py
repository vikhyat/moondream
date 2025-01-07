import torch
import datasets
import math

from tqdm import tqdm

from ..torch.config import MoondreamConfig
from ..torch.moondream import MoondreamModel
from ..torch.weights import load_weights_into_model

dataset = datasets.load_dataset("vikhyatk/gazefollow", split="test")

torch.set_default_device("cuda")
model = MoondreamModel(MoondreamConfig())
load_weights_into_model("model.pt", model)

mean_l2_error = []
min_l2_error = []
total = 0


for i, row in tqdm(enumerate(dataset), total=len(dataset)):
    encoded_image = model.encode_image(row["image"])

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

        if i % 100 == 0:
            print("Mean L2 error:", sum(mean_l2_error) / total)
            print("Min L2 error:", sum(min_l2_error) / total)


print()
print("Single prediction mode")
print("Final score:")
print("Mean L2 error:", sum(mean_l2_error) / total)
print("Min L2 error:", sum(min_l2_error) / total)
