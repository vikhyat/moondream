from ..torch.hf_moondream import HfMoondream as Moondream, HfConfig
from transformers import AutoConfig, AutoModelForCausalLM

# Register the model and config for Auto classes
AutoConfig.register("moondream3", HfConfig)
AutoModelForCausalLM.register(HfConfig, Moondream)
