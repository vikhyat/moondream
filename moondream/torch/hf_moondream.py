from transformers import PreTrainedModel, PretrainedConfig

from .config import MoondreamConfig
from .moondream import MoondreamModel

# Files sometimes don't get loaded without these...
from .image_crops import *
from .vision import *
from .text import *
from .region import *
from .utils import *


def extract_question(text):
    prefix = "<image>\n\nQuestion: "
    suffix = "\n\nAnswer:"

    if text.startswith(prefix) and text.endswith(suffix):
        return text[len(prefix) : -len(suffix)]
    else:
        return None


class HfConfig(PretrainedConfig):
    _auto_class = "AutoConfig"
    model_type = "moondream1"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.config = {}


class HfMoondream(PreTrainedModel):
    _auto_class = "AutoModelForCausalLM"
    config_class = HfConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = MoondreamModel(
            MoondreamConfig.from_dict(config.config), setup_caches=False
        )
        self._is_kv_cache_setup = False

    def _setup_caches(self):
        if not self._is_kv_cache_setup:
            self.model._setup_caches()
            self._is_kv_cache_setup = True

    @property
    def encode_image(self):
        self._setup_caches()
        return self.model.encode_image

    @property
    def query(self):
        self._setup_caches()
        return self.model.query

    @property
    def caption(self):
        self._setup_caches()
        return self.model.caption

    @property
    def detect(self):
        self._setup_caches()
        return self.model.detect

    @property
    def point(self):
        self._setup_caches()
        return self.model.point

    @property
    def detect_gaze(self):
        self._setup_caches()
        return self.model.detect_gaze

    def answer_question(
        self,
        image_embeds,
        question,
        tokenizer=None,
        chat_history="",
        result_queue=None,
        max_new_tokens=256,
        **kwargs
    ):
        answer = self.query(image_embeds, question)["answer"].strip()

        if result_queue is not None:
            result_queue.put(answer)
        return answer

    def batch_answer(self, images, prompts, tokenizer=None, **kwargs):
        answers = []
        for image, prompt in zip(images, prompts):
            answers.append(self.query(image, prompt)["answer"].strip())
        return answers

    def _unsupported_exception(self):
        raise NotImplementedError(
            "This method is not supported in the latest version of moondream. "
            "Consider upgrading to the updated API spec, or alternately pin "
            "to 'revision=2024-08-26'."
        )

    def generate(self, image_embeds, prompt, tokenizer, max_new_tokens=128, **kwargs):
        """
        Function definition remains unchanged for backwards compatibility.
        Be aware that tokenizer, max_new_takens, and kwargs are ignored.
        """
        prompt_extracted = extract_question(prompt)
        if prompt_extracted is not None:
            answer = self.model.query(
                image=image_embeds, question=prompt_extracted, stream=False
            )["answer"]
        else:
            image_embeds = self.encode_image(image_embeds)
            prompt_tokens = torch.tensor(
                [self.model.tokenizer.encode(prompt).ids],
                device=self.device,
            )

            def generator():
                for token in self.model._generate_text(
                    prompt_tokens,
                    image_embeds.kv_cache,
                    image_embeds.pos,
                    max_new_tokens,
                ):
                    yield token

            answer = "".join(list(generator()))

        return [answer]

    def get_input_embeddings(self):
        return super().get_input_embeddings()

    def input_embeds(self, *args, **kwargs):
        self._unsupported_exception()
