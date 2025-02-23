import torch
import torch.nn as nn
from torch.utils.data import Dataset
import math
from safetensors.torch import save_file

from tqdm import tqdm
from datasets import load_dataset
from bitsandbytes.optim import AdamW8bit
import wandb

from ..torch.weights import load_weights_into_model
from ..torch.moondream import MoondreamModel, MoondreamConfig, text_encoder
from ..torch.text import _produce_hidden, _lm_head, TextConfig

# This is a intended to be a basic starting point for fine-tuning the text encoder.
# Your optimal hyperparams and data may be different.
MODEL_PATH = ""
# Your data should end with the eos token. Here is the textual representation.
ANSWER_EOS = "<|endoftext|>"
LR = 3e-6
EPOCHS = 3
GRAD_ACCUM_STEPS = 128


def lr_schedule(step, max_steps):
    x = step / max_steps
    if x < 0.1:
        return 0.1 * LR + 0.9 * LR * x / 0.1
    else:
        return 0.1 * LR + 0.9 * LR * (1 + math.cos(math.pi * (x - 0.1))) / 2


def text_loss(
    inputs_embeds: torch.Tensor, w: nn.Module, labels: torch.Tensor, config: TextConfig
):
    _, q_len, _ = inputs_embeds.shape
    hidden_BTC = _produce_hidden(inputs_embeds, w, config)
    lm_logits = _lm_head(hidden_BTC, w)

    loss = None
    if labels is not None:
        _, _, l_len = labels.shape
        shift_index = (q_len - l_len) - 1
        shifted_logits = lm_logits[..., shift_index:-1, :].contiguous()
        shifted_labels = labels.contiguous()
        loss = nn.CrossEntropyLoss()(
            shifted_logits.view(-1, shifted_logits.size(-1)),
            shifted_labels.view(-1),
        )
    return loss


class DocciDataset(Dataset):
    def __init__(self, split="train"):
        self.data = load_dataset("google/docci", trust_remote_code=True)[split]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        description = sample["description"]
        return {
            "image": sample["image"],
            "qa": {
                "question": "\n\nQuestion: Describe this image.\n\nAnswer:",
                "answer": f"{description}{ANSWER_EOS}",
            },
        }


def main():
    if torch.cuda.is_available():
        torch.set_default_device("cuda")
    elif torch.backends.mps.is_available():
        torch.set_default_device("mps")

    wandb.init(
        project="moondream-ft",
        config={
            "EPOCHS": EPOCHS,
            "GRAD_ACCUM_STEPS": GRAD_ACCUM_STEPS,
            "LR": LR,
        },
    )

    config = MoondreamConfig()
    model = MoondreamModel(config)
    load_weights_into_model(MODEL_PATH, model)

    optimizer = AdamW8bit(
        [
            {"params": model.text.parameters()},
        ],
        lr=LR,
        betas=(0.9, 0.95),
        eps=1e-6,
    )

    dataset = DocciDataset("train")

    total_steps = EPOCHS * len(dataset) // GRAD_ACCUM_STEPS
    pbar = tqdm(total=total_steps)

    i = 0
    for epoch in range(EPOCHS):
        for sample in dataset:
            i += 1
            with torch.no_grad():
                img_emb = model._run_vision_encoder(sample["image"])
            bos_emb = text_encoder(
                torch.tensor([[model.config.tokenizer.bos_id]], device=model.device),
                model.text,
            )
            question_tokens = model.tokenizer.encode(sample["qa"]["question"]).ids
            question_emb = text_encoder(
                torch.tensor([[question_tokens]], device=model.device),
                model.text,
            ).squeeze(0)
            answer_tokens = model.tokenizer.encode(sample["qa"]["answer"]).ids
            answer_emb = text_encoder(
                torch.tensor([[answer_tokens]], device=model.device),
                model.text,
            ).squeeze(0)
            inputs_embeds = torch.cat(
                [bos_emb, img_emb[None], question_emb, answer_emb], dim=1
            )
            loss = text_loss(
                inputs_embeds=inputs_embeds,
                w=model.text,
                labels=torch.tensor([[answer_tokens]], device=model.device),
                config=config.text,
            )

            loss.backward()

            if i % GRAD_ACCUM_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()

                lr = lr_schedule(i / GRAD_ACCUM_STEPS, total_steps)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr
                pbar.set_postfix({"step": i // GRAD_ACCUM_STEPS, "loss": loss.item()})
                pbar.update(1)
                wandb.log(
                    {"loss/train": loss.item(), "lr": optimizer.param_groups[0]["lr"]}
                )
    wandb.finish()
    # Add save path: ex. home/model.safetensors
    save_file(
        model.state_dict(),
        "moondream_finetune.safetensors",
    )


if __name__ == "__main__":
    """
    Replace paths with your appropriate paths.
    To run: python -m moondream.finetune.finetune_text
    """
    main()
