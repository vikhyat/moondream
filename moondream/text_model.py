import torch
import transformers
from transformers import CodeGenTokenizerFast as Tokenizer
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from .phi.configuration_phi import PhiConfig
from .phi.modeling_phi import PhiForCausalLM
import re

transformers.logging.set_verbosity_error()


class TextModel:
    def __init__(self, model_path: str = "model") -> None:
        super().__init__()

        # Determine if CUDA (GPU) is available and use it; otherwise, use CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = Tokenizer.from_pretrained(f"{model_path}/tokenizer")
        phi_config = PhiConfig.from_pretrained(f"{model_path}/text_model_cfg.json")

        with init_empty_weights():
            self.model = PhiForCausalLM(phi_config)

        self.model = load_checkpoint_and_dispatch(
            self.model,
            f"{model_path}/text_model.pt",
            device_map={"": self.device.type},
        )

        self.text_emb = self.model.get_input_embeddings().to(self.device)

    def input_embeds(self, prompt, image_embeds):
        embeds = []

        def _add_toks(toks):
            embeds.append(self.text_emb(toks))

        def _tokenize(txt):
            return self.tokenizer(
                txt, return_tensors="pt", add_special_tokens=False
            ).input_ids.to(self.model.device)

        # Add BOS token
        _add_toks(
            torch.tensor([[self.tokenizer.bos_token_id]], device=self.model.device)
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

    def answer_question(self, image_embeds, question, **kwargs):
        prompt = f"<image>\n\nQuestion: {question}\n\nAnswer:"
        answer = self.generate(
            image_embeds,
            prompt,
            eos_text="<END>",
            max_new_tokens=128,
            **kwargs,
        )[0]

        return re.sub("<$", "", re.sub("END$", "", answer)).strip()
