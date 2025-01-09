from transformers import PreTrainedModel, PretrainedConfig

from .config import MoondreamConfig
from .moondream import MoondreamModel

# Files sometimes don't get loaded without these...
from .image_crops import *
from .vision import *
from .text import *
from .region import *
from .utils import *


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
        self.model = MoondreamModel(MoondreamConfig.from_dict(config.config))

    @property
    def encode_image(self):
        return self.model.encode_image

    @property
    def query(self):
        return self.model.query

    @property
    def caption(self):
        return self.model.caption

    @property
    def detect(self):
        return self.model.detect

    @property
    def point(self):
        return self.model.point

    @property
    def detect_gaze(self):
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

    def generate(self, *args, **kwargs):
        self._unsupported_exception()

    def get_input_embeddings(self):
        return super().get_input_embeddings()

    def input_embeds(self, *args, **kwargs):
        self._unsupported_exception()
