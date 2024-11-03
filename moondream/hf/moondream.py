from typing import List, Literal, Optional

import torch
from PIL import Image
from transformers import PreTrainedModel

from .configuration_moondream import MoondreamConfig, PhiConfig
from .modeling_phi import PhiForCausalLM
from .region_model import RegionModel
from .vision_encoder import VisionEncoder


class Moondream(PreTrainedModel):
    config_class = MoondreamConfig
    _supports_flash_attn_2 = True

    def __init__(self, config):
        super().__init__(config)
        self.vision_encoder = VisionEncoder(
            use_flash_attn=config._attn_implementation == "flash_attention_2"
        )
        self.region_model = RegionModel()

        if type(config.text_config) == dict:
            phi_config = PhiConfig(
                **config.text_config, attn_implementation=config._attn_implementation
            )
        else:
            phi_config = config.text_config
        self.text_model = PhiForCausalLM(phi_config)

    @property
    def device(self):
        return self.text_model.device

    def encode_image(self, image):
        with torch.no_grad():
            return self.vision_encoder(image)

    def input_embeds(self, prompt, image_embeds, tokenizer):
        def _tokenize(txt):
            return tokenizer(
                txt, return_tensors="pt", add_special_tokens=False
            ).input_ids.to(self.device)

        text_emb = self.text_model.get_input_embeddings()

        # Add BOS token
        embeds = []
        embeds.append(
            text_emb((torch.tensor([[tokenizer.bos_token_id]], device=self.device)))
        )

        if "<image>" not in prompt:
            embeds.append(text_emb(_tokenize(prompt)))
        else:
            assert prompt.count("<image>") == 1
            before, after = prompt.split("<image>")
            if len(before) > 0:
                embeds.append(text_emb(_tokenize(before)))
            embeds.append(image_embeds.to(self.device))
            if len(after) > 0:
                embeds.append(text_emb(_tokenize(after)))

        return torch.cat(embeds, dim=1)

    def get_input_embeddings(self):
        return self.text_model.get_input_embeddings()

    def generate(
        self,
        image_embeds,
        prompt,
        tokenizer,
        max_new_tokens=128,
        **kwargs,
    ):
        generate_config = {
            "eos_token_id": tokenizer.eos_token_id,
            "bos_token_id": tokenizer.bos_token_id,
            "pad_token_id": tokenizer.bos_token_id,
            "max_new_tokens": max_new_tokens,
            **kwargs,
        }

        with torch.no_grad():
            inputs_embeds = self.input_embeds(prompt, image_embeds, tokenizer)
            attention_mask = torch.ones(
                (inputs_embeds.shape[0], inputs_embeds.shape[1]), device=self.device
            )
            output_ids = self.text_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                **generate_config,
            )

        return tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    # Note: Not ready for use yet, intended for September release.
    def caption(
        self,
        images: List[Image.Image],
        tokenizer,
        length: Optional[Literal["short"]] = None,
        **kwargs,
    ):
        image_embeds = self.encode_image(images)

        templated_prompts = [
            f"<image>\n\n{'Short caption' if length == 'short' else 'Caption'}:"
            for _ in images
        ]
        inputs_embeds = torch.stack(
            [
                self.input_embeds(prompt, image_embed.unsqueeze(0), tokenizer)[0]
                for prompt, image_embed in zip(templated_prompts, image_embeds)
            ]
        )
        attention_mask = torch.ones(
            (inputs_embeds.shape[0], inputs_embeds.shape[1]), device=self.device
        )

        generate_config = {
            "eos_token_id": tokenizer.eos_token_id,
            "bos_token_id": tokenizer.bos_token_id,
            "pad_token_id": tokenizer.bos_token_id,
            "repetition_penalty": 1.2,
            "max_new_tokens": 512,
            **kwargs,
        }

        with torch.no_grad():
            output_ids = self.text_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                **generate_config,
            )

        return [
            x.strip()
            for x in tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        ]

    def answer_question(
        self,
        image_embeds,
        question,
        tokenizer,
        chat_history="",
        result_queue=None,
        max_new_tokens=256,
        **kwargs,
    ):
        prompt = f"<image>\n\n{chat_history}Question: {question}\n\nAnswer:"
        answer = self.generate(
            image_embeds,
            prompt,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )[0]
        cleaned_answer = answer.strip()

        # Use the result_queue to pass the result if it is provided
        if result_queue:
            result_queue.put(cleaned_answer)
        else:
            return cleaned_answer

    def batch_answer(
        self,
        images,
        prompts,
        tokenizer,
        **kwargs,
    ):
        image_embeds = self.encode_image(images)

        templated_prompts = [
            f"<image>\n\nQuestion: {prompt}\n\nAnswer:" for prompt in prompts
        ]
        prompt_embs = [
            self.input_embeds(prompt, image_embed.unsqueeze(0), tokenizer)[0]
            for prompt, image_embed in zip(templated_prompts, image_embeds)
        ]

        bos_emb = prompt_embs[0][0]
        max_len = max([p.shape[0] for p in prompt_embs])

        inputs_embeds = torch.cat(
            [
                torch.cat([bos_emb.repeat(max_len - p.shape[0], 1), p]).unsqueeze(0)
                for p in prompt_embs
            ],
            dim=0,
        )
        attention_mask = torch.cat(
            [
                torch.cat(
                    [
                        torch.zeros(
                            1,
                            max_len - p.shape[0],
                            device=self.device,
                            dtype=torch.long,
                        ),
                        torch.ones(1, p.shape[0], device=self.device, dtype=torch.long),
                    ],
                    dim=1,
                )
                for p in prompt_embs
            ],
            dim=0,
        )

        generate_config = {
            "eos_token_id": tokenizer.eos_token_id,
            "bos_token_id": tokenizer.bos_token_id,
            "pad_token_id": tokenizer.bos_token_id,
            "max_new_tokens": 512,
            **kwargs,
        }

        with torch.no_grad():
            output_ids = self.text_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                **generate_config,
            )

        return [
            x.strip()
            for x in tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        ]

    def detect(
        self,
        image: Image.Image,
        query: str,
        tokenizer,
        max_objects=50,
    ):
        prompt = f"<image>\n\nDetect: {query}\n\n"
        image_embeds = self.encode_image(image)
        inputs_embeds = self.input_embeds(prompt, image_embeds, tokenizer)
        generate_config = {
            "eos_token_id": tokenizer.eos_token_id,
            "bos_token_id": tokenizer.bos_token_id,
            "pad_token_id": tokenizer.bos_token_id,
            "max_new_tokens": 1,
        }

        past_key_values = None
        generated_boxes = []

        with torch.no_grad():
            while len(generated_boxes) < max_objects:
                # x coordinate
                attention_mask = torch.ones(
                    (inputs_embeds.shape[0], inputs_embeds.shape[1]), device=self.device
                )
                output = self.text_model.generate(
                    inputs_embeds=inputs_embeds,
                    past_key_values=past_key_values,
                    attention_mask=attention_mask,
                    return_dict_in_generate=True,
                    output_hidden_states=True,
                    **generate_config,
                )
                if output["sequences"][0][0].item() == tokenizer.eos_token_id:
                    break

                x_coord_hidden = output["hidden_states"][0][-1][:, -1, :]
                x_coord_logits = self.region_model.decode_coordinate(x_coord_hidden)
                x_coord_decoded = (
                    torch.argmax(x_coord_logits, dim=-1).to(torch.float32) / 1024
                ).to(torch.float16)
                x_coord_encoded = self.region_model.encode_coordinate(
                    x_coord_decoded
                ).unsqueeze(0)
                inputs_embeds = torch.cat(
                    [inputs_embeds, x_coord_encoded.unsqueeze(0)], dim=1
                )
                past_key_values = output["past_key_values"]

                # y coordinate
                attention_mask = torch.ones(
                    (inputs_embeds.shape[0], inputs_embeds.shape[1]), device=self.device
                )
                output = self.text_model.generate(
                    inputs_embeds=inputs_embeds,
                    past_key_values=past_key_values,
                    attention_mask=attention_mask,
                    return_dict_in_generate=True,
                    output_hidden_states=True,
                    **generate_config,
                )
                y_coord_hidden = output["hidden_states"][0][-1][:, -1, :]
                y_coord_logits = self.region_model.decode_coordinate(y_coord_hidden)
                y_coord_decoded = (
                    torch.argmax(y_coord_logits, dim=-1).to(torch.float32) / 1024
                ).to(torch.float16)
                y_coord_encoded = self.region_model.encode_coordinate(
                    y_coord_decoded
                ).unsqueeze(0)
                inputs_embeds = torch.cat(
                    [inputs_embeds, y_coord_encoded.unsqueeze(0)], dim=1
                )
                past_key_values = output["past_key_values"]

                # size (h and w)
                attention_mask = torch.ones(
                    (inputs_embeds.shape[0], inputs_embeds.shape[1]), device=self.device
                )
                output = self.text_model.generate(
                    inputs_embeds=inputs_embeds,
                    past_key_values=past_key_values,
                    attention_mask=attention_mask,
                    return_dict_in_generate=True,
                    output_hidden_states=True,
                    **generate_config,
                )
                size_hidden = output["hidden_states"][0][-1][:, -1, :]
                size_logits = self.region_model.decode_size(size_hidden)
                size_decoded = (
                    torch.argmax(size_logits, dim=-1).to(torch.float32) / 1024
                ).to(torch.float16)
                size_encoded = self.region_model.encode_size(size_decoded)
                inputs_embeds = torch.cat(
                    [inputs_embeds, size_encoded.unsqueeze(0)], dim=1
                )
                past_key_values = output["past_key_values"]

                x_center = x_coord_decoded[0].item()
                y_center = y_coord_decoded[0].item()
                w_center = size_decoded[0][0].item()
                h_center = size_decoded[0][1].item()
                x_min = max(x_center - w_center / 2, 0)
                y_min = max(y_center - h_center / 2, 0)
                x_max = min(x_center + w_center / 2, 1)
                y_max = min(y_center + h_center / 2, 1)

                generated_boxes.append(
                    {
                        "x_min": x_min,
                        "y_min": y_min,
                        "x_max": x_max,
                        "y_max": y_max,
                    }
                )

        return generated_boxes
