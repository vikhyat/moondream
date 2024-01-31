import re
import torch
from torch import nn
import transformers
from .modeling_phi import PhiForCausalLM
from .configuration_moondream import PhiConfig

transformers.logging.set_verbosity_error()


class TextModel(nn.Module):
    def __init__(self, config, tokenizer) -> None:
        super().__init__()

        self.model = PhiForCausalLM(PhiConfig(**config.phi_config))
        self.text_emb = self.model.get_input_embeddings()
        self.tokenizer = tokenizer

    def input_embeds(self, prompt, image_embeds):
        def _tokenize(txt):
            return self.tokenizer(
                txt, return_tensors="pt", add_special_tokens=False
            ).input_ids.to(self.model.device)

        # Add BOS token
        embeds = []
        embeds.append(
            self.text_emb(
                (
                    torch.tensor(
                        [[self.tokenizer.bos_token_id]], device=self.model.device
                    )
                )
            )
        )

        if "<image>" not in prompt:
            embeds.append(self.text_emb(_tokenize(prompt)))
        else:
            assert prompt.count("<image>") == 1
            before, after = prompt.split("<image>")
            embeds.append(self.text_emb(_tokenize(f"{before}<image>")))
            embeds.append(image_embeds.to(self.model.device))
            embeds.append(self.text_emb(_tokenize(f"</image>{after}")))

        return torch.cat(embeds, dim=1)

    def generate(
        self, image_embeds, prompt, eos_text="Human:", max_new_tokens=128, **kwargs
    ):
        eos_tokens = self.tokenizer(eos_text, add_special_tokens=False)[0].ids

        generate_config = {
            "eos_token_id": eos_tokens,
            "bos_token_id": self.tokenizer.bos_token_id,
            "pad_token_id": self.tokenizer.eos_token_id,
            "max_new_tokens": max_new_tokens,
            **kwargs,
        }

        with torch.no_grad():
            inputs_embeds = self.input_embeds(prompt, image_embeds)
            output_ids = self.model.generate(
                inputs_embeds=inputs_embeds, **generate_config
            )

        return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    def answer_question(
        self, image_embeds, question, chat_history="", result_queue=None,
        max_new_tokens=128, **kwargs
    ):
        prompt = f"<image>\n\n{chat_history}Question: {question}\n\nAnswer:"
        answer = self.generate(
            image_embeds,
            prompt,
            eos_text="<END>",
            max_new_tokens=max_new_tokens,
            **kwargs,
        )[0]
        cleaned_answer = re.sub("<$", "", re.sub("END$", "", answer)).strip()

        # Use the result_queue to pass the result if it is provided
        if result_queue:
            result_queue.put(cleaned_answer)
        else:
            return cleaned_answer
