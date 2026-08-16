from dataclasses import dataclass
from typing import List, Literal, Optional, TypedDict, Union


@dataclass
class EncodedImage:
    pass


@dataclass
class Base64EncodedImage(EncodedImage):
    image_url: str


SamplingSettings = TypedDict(
    "SamplingSettings",
    {
        "temperature": float,
        "top_p": float,
        "max_tokens": int,
        "max_objects": int,
    },
    total=False,
)

Region = TypedDict(
    "Region", {"x_min": float, "y_min": float, "x_max": float, "y_max": float}
)
DetectOutput = TypedDict("DetectOutput", {"objects": List[Region]})

Point = TypedDict("Point", {"x": float, "y": float})
PointOutput = TypedDict("PointOutput", {"points": List[Point]})

SpatialRef = List[float]  # [x, y] point or [x1, y1, x2, y2] bbox, normalized to [0, 1]

PointGroundTruth = TypedDict(
    "PointGroundTruth",
    {"points": List[Point], "boxes": List[Region]},
    total=False,
)

DetectGroundTruth = TypedDict("DetectGroundTruth", {"boxes": List[Region]})

FinetuneGroundTruth = Union[PointGroundTruth, DetectGroundTruth]
Skill = Literal["query", "point", "detect"]

ReasoningGrounding = TypedDict(
    "ReasoningGrounding",
    {
        "start_idx": int,
        "end_idx": int,
        "points": List[List[int]],  # List of [x, y] coordinate pairs
    },
)

Reasoning = TypedDict(
    "Reasoning",
    {
        "text": str,
        "grounding": List[ReasoningGrounding],
    },
)

QueryTarget = TypedDict(
    "QueryTarget",
    {"answer": str, "reasoning": Reasoning},
    total=False,
)

PointTarget = TypedDict(
    "PointTarget",
    {"points": List[Point], "boxes": List[Region]},
    total=False,
)

DetectTarget = TypedDict("DetectTarget", {"boxes": List[Region]})

SFTTarget = Union[QueryTarget, PointTarget, DetectTarget]

RolloutOutput = TypedDict(
    "RolloutOutput",
    {
        "answer": str,
        "reasoning": Optional[Reasoning],
        "points": List[Point],
        "objects": List[Region],
    },
    total=False,
)

SkillRequest = TypedDict(
    "SkillRequest",
    {
        "skill": Skill,
        "image_url": str,
        "question": str,
        "object": str,
        "spatial_refs": List[SpatialRef],
        "reasoning": bool,
        "settings": SamplingSettings,
    },
    total=False,
)

Rollout = TypedDict(
    "Rollout",
    {
        "skill": Skill,
        "finish_reason": str,
        "output": RolloutOutput,
        "answer_tokens": List[int],
        "thinking_tokens": List[int],
        "has_answer_separator": bool,
        "coords": List[object],
        "sizes": List[object],
    },
    total=False,
)

RolloutsResponse = TypedDict(
    "RolloutsResponse",
    {
        "request": SkillRequest,
        "rollouts": List[Rollout],
        "rewards": Optional[List[float]],
    },
    total=False,
)

TrainStepOutput = TypedDict(
    "TrainStepOutput",
    {
        "step": int,
        "applied": bool,
        "kl": Optional[float],
        "router_kl": Optional[float],
        "grad_norm": Optional[float],
        "sft_loss": Optional[float],
        "reward_mean": Optional[float],
        "reward_std": Optional[float],
    },
    total=False,
)

MetricsLogOutput = TypedDict(
    "MetricsLogOutput",
    {
        "ok": bool,
        "step": int,
        "logged_count": int,
    },
    total=False,
)

FinetuneInfo = TypedDict(
    "FinetuneInfo",
    {
        "finetune_id": str,
        "name": str,
        "rank": int,
        "created_at_ms": int,
        "updated_at_ms": int,
    },
    total=False,
)

CheckpointInfo = TypedDict(
    "CheckpointInfo",
    {
        "checkpoint_id": str,
        "finetune_id": str,
        "step": int,
        "expires_at_ms": Optional[int],
        "created_at_ms": int,
        "updated_at_ms": int,
    },
    total=False,
)

CheckpointListOutput = TypedDict(
    "CheckpointListOutput",
    {
        "checkpoints": List[CheckpointInfo],
        "next_cursor": Optional[str],
        "has_more": bool,
    },
    total=False,
)

SaveCheckpointOutput = TypedDict(
    "SaveCheckpointOutput",
    {"ok": bool, "checkpoint": CheckpointInfo},
    total=False,
)

RLGroup = TypedDict(
    "RLGroup",
    {
        "mode": Literal["rl"],
        "request": SkillRequest,
        "rollouts": List[Rollout],
        "rewards": List[float],
    },
    total=False,
)

SFTGroup = TypedDict(
    "SFTGroup",
    {
        "mode": Literal["sft"],
        "request": SkillRequest,
        "target": SFTTarget,
    },
    total=False,
)