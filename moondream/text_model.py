from torch import nn
import transformers
from .modeling_phi import PhiForCausalLM
from .configuration_moondream import PhiConfig

transformers.logging.set_verbosity_error()


class TextModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        if type(config.phi_config) == dict:
            phi_config = PhiConfig(**config.phi_config)
        else:
            phi_config = config.phi_config

        self.model = PhiForCausalLM(phi_config)
        self.text_emb = self.model.get_input_embeddings()
