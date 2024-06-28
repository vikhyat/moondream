import torch
from .vision_encoder import VisionEncoder
from .configuration_moondream import MoondreamConfig
from transformers import PreTrainedModel

from .modeling_phi import PhiForCausalLM
from .configuration_moondream import PhiConfig

class Moondream(PreTrainedModel):
    config_class = MoondreamConfig
    _supports_flash_attn_2 = True

    def __init__(self, config):
        super().__init__(config)
        self.vision_encoder = VisionEncoder(
            use_flash_attn=config._attn_implementation == "flash_attention_2"
        )

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
            output_ids = self.text_model.generate(
                inputs_embeds=inputs_embeds, **generate_config
            )

        return tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    def answer_question(
        self,
        image_embeds,
        question,
        tokenizer,
        chat_history="",
        result_queue=None,
        **kwargs,
    ):
        prompt = f"<image>\n\n{chat_history}Question: {question}\n\nAnswer:"
        answer = self.generate(
            image_embeds,
            prompt,
            tokenizer=tokenizer,
            max_new_tokens=512,
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
