from .vision_encoder import VisionEncoder
from .text_model import TextModel
from .configuration_moondream import MoondreamConfig
from transformers import PreTrainedModel


class Moondream(PreTrainedModel):
    config_class = MoondreamConfig

    def __init__(self, config):
        super().__init__(config)
        self.vision_encoder = VisionEncoder()
        self.text_model = TextModel(config)
