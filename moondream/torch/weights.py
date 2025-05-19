import safetensors
import torch
import torch.nn as nn

from contextlib import contextmanager
from typing import Callable, List
from .text import build_text_model
# from .vision import build_vision_model # Not used
import gc

# import time # Not used

@contextmanager
def safetensors_open(safetensors_file: str):
    """
    Simplify interfacing with safetensors files. Eliminates the need to ignore
    type errors when using the `safe_open` function.
    """
    # Open the safetensors file for reading in PyTorch framework
    with safetensors.safe_open(
        safetensors_file, framework="pt"
    ) as st:

        def get_tensor(name: str) -> torch.Tensor:
            return st.get_tensor(name)

        def get_keys() -> List[str]:
            return st.keys()

        get_tensor.keys = get_keys

        yield get_tensor


def _load_weights(get_tensor: Callable[[str], torch.Tensor], model: nn.Module) -> None:
    """Internal function to load weights using a tensor getter function."""

    model.to(dtype=torch.float16)

    vision = model.vision
    region = model.region
    # Define a mapping from expected weight names in the file to model parameters
    weight_map = {
        "vision_encoder.encoder.model.visual.patch_embed.linear.weight": vision[
            "patch_emb"
        ].weight,
        "vision_encoder.encoder.model.visual.patch_embed.linear.bias": vision[
            "patch_emb"
        ].bias,
        "vision_encoder.encoder.model.visual.pos_embed": vision.pos_emb,
        "vision_encoder.encoder.model.visual.norm.weight": vision["post_ln"].weight,
        "vision_encoder.encoder.model.visual.norm.bias": vision["post_ln"].bias,
        "vision_encoder.projection.mlp.fc1.weight": vision["proj_mlp"]["fc1"].weight,
        "vision_encoder.projection.mlp.fc1.bias": vision["proj_mlp"]["fc1"].bias,
        "vision_encoder.projection.mlp.fc2.weight": vision["proj_mlp"]["fc2"].weight,
        "vision_encoder.projection.mlp.fc2.bias": vision["proj_mlp"]["fc2"].bias,
        "text_model.transformer.embd.wte.weight": model.text.wte,
        "text_model.lm_head.ln.weight": model.text["post_ln"].weight,
        "text_model.lm_head.ln.bias": model.text["post_ln"].bias,
        "text_model.lm_head.linear.weight": model.text["lm_head"].weight,
        "text_model.lm_head.linear.bias": model.text["lm_head"].bias,
        "region_model.coordinate_encoder.weight": region["coord_encoder"].weight,
        "region_model.coordinate_encoder.bias": region["coord_encoder"].bias,
        "region_model.coordinate_decoder.fc1.weight": region["coord_decoder"][
            "fc1"
        ].weight,
        "region_model.coordinate_decoder.fc1.bias": region["coord_decoder"]["fc1"].bias,
        "region_model.coordinate_decoder.fc2.weight": region["coord_decoder"][
            "fc2"
        ].weight,
        "region_model.coordinate_decoder.fc2.bias": region["coord_decoder"]["fc2"].bias,
        "region_model.size_encoder.weight": region["size_encoder"].weight,
        "region_model.size_encoder.bias": region["size_encoder"].bias,
        "region_model.size_decoder.fc1.weight": region["size_decoder"]["fc1"].weight,
        "region_model.size_decoder.fc1.bias": region["size_decoder"]["fc1"].bias,
        "region_model.size_decoder.fc2.weight": region["size_decoder"]["fc2"].weight,
        "region_model.size_decoder.fc2.bias": region["size_decoder"]["fc2"].bias,
    }

    # Dynamically add weights for vision transformer blocks
    for i in range(len(model.vision["blocks"])):
        prefix = f"vision_encoder.encoder.model.visual.blocks.{i}"
        blk = model.vision["blocks"][i]
        weight_map.update(
            {
                f"{prefix}.norm1.weight": blk["ln1"].weight,
                f"{prefix}.norm1.bias": blk["ln1"].bias,
                f"{prefix}.norm2.weight": blk["ln2"].weight,
                f"{prefix}.norm2.bias": blk["ln2"].bias,
                f"{prefix}.attn.qkv.weight": blk["attn"]["qkv"].weight,
                f"{prefix}.attn.qkv.bias": blk["attn"]["qkv"].bias,
                f"{prefix}.attn.proj.weight": blk["attn"]["proj"].weight,
                f"{prefix}.attn.proj.bias": blk["attn"]["proj"].bias,
                f"{prefix}.mlp.fc1.weight": blk["mlp"]["fc1"].weight,
                f"{prefix}.mlp.fc1.bias": blk["mlp"]["fc1"].bias,
                f"{prefix}.mlp.fc2.weight": blk["mlp"]["fc2"].weight,
                f"{prefix}.mlp.fc2.bias": blk["mlp"]["fc2"].bias,
            }
        )

    # Dynamically add weights for text transformer blocks
    for i in range(len(model.text["blocks"])):
        prefix = f"text_model.transformer.h.{i}"
        blk = model.text["blocks"][i]
        weight_map.update(
            {
                f"{prefix}.ln.weight": blk["ln"].weight,
                f"{prefix}.ln.bias": blk["ln"].bias,
                f"{prefix}.mixer.Wqkv.weight": blk["attn"]["qkv"].weight,
                f"{prefix}.mixer.Wqkv.bias": blk["attn"]["qkv"].bias,
                f"{prefix}.mixer.out_proj.weight": blk["attn"]["proj"].weight,
                f"{prefix}.mixer.out_proj.bias": blk["attn"]["proj"].bias,
                f"{prefix}.mlp.fc1.weight": blk["mlp"]["fc1"].weight,
                f"{prefix}.mlp.fc1.bias": blk["mlp"]["fc1"].bias,
                f"{prefix}.mlp.fc2.weight": blk["mlp"]["fc2"].weight,
                f"{prefix}.mlp.fc2.bias": blk["mlp"]["fc2"].bias,
            }
        )

    # Copy data from loaded tensors to model parameters
    for key, tensor_target in weight_map.items():
        source_tensor = get_tensor(key) # get_tensor is expected to provide fp16 tensors if model is fp16
                                        # because the lambda passed to _load_weights will do .to(fp16)
        tensor_target.data.copy_(source_tensor)


    # Special handling for transposed weights
    coord_features_weight = get_tensor("region_model.coordinate_features.weight")
    region.coord_features.data.copy_(coord_features_weight.T)

    size_features_weight = get_tensor("region_model.size_features.weight")
    region.size_features.data.copy_(size_features_weight.T)


def load_weights_from_safetensors(weights_file: str, model: nn.Module) -> None:
    """Load weights from a safetensors file into the model instance,
    with support for quantized models."""
    device = next(model.parameters()).device
    print(f"Loading .safetensors file to CPU first (target device: {device})...")

    with safetensors_open(weights_file) as get_tensor:
        all_keys = get_tensor.keys()

        # Detect if the model is quantized by inspecting tensor keys
        is_quantized = any('.qweight' in key or '_quantized' in key or 'quant.' in key for key in all_keys)

        if is_quantized:
            print("Quantized model detected from safetensors file keys.")
        else:
            print("Non-quantized model detected from safetensors file keys.")

        if hasattr(model, 'text') and model.text is None:
            # Determine the dtype for the text model based on quantization status
            text_dtype = torch.int8 if is_quantized else torch.float16
            print(f"Building text model with dtype: {text_dtype} for safetensors loading.")
            # Assuming build_text_model can handle the specified dtype (e.g., int8 for quantized)
            model.text = build_text_model(model.config.text, dtype=text_dtype)
            model.text.to(device)  # Move the newly built text model to the target device
            
            if hasattr(model, 'setup_caches_flag') and model.setup_caches_flag:
                model._setup_caches()
                print("Caches set up for text model.")

        elif hasattr(model, 'text') and model.text is not None:
            # Text model already exists, check if its dtype needs adjustment
            current_text_dtype = next(model.text.parameters()).dtype
            print(f"Text model already exists. Current text model dtype: {current_text_dtype}")
            if not is_quantized and current_text_dtype != torch.float16:
                # If loading non-quantized weights and existing text model is not fp16, convert it
                print(f"Converting existing text model to float16 for non-quantized safetensors.")
                model.text.to(dtype=torch.float16)


        # --- Weight loading logic ---
        if is_quantized:
            print("Loading state_dict for quantized safetensors model.")

            tensors_processed = {}
            for k in all_keys:
                cleaned_key = k
                if cleaned_key.startswith("model."):
                    cleaned_key = cleaned_key[len("model."):]

                if "._orig_mod" in cleaned_key: # A more careful replacement might be needed
                    cleaned_key = cleaned_key.replace("._orig_mod", "")
                
                tensor = get_tensor(k).to(device) # Move tensor to target device
                tensors_processed[cleaned_key] = tensor
            
            model.load_state_dict(tensors_processed, strict=False)
        else:

            is_direct_load_style = False

            if any(key_name in all_keys for key_name in ["vision.blocks.0.attn.proj.bias", "model.vision.blocks.0.attn.proj.bias"]):
                 is_direct_load_style = True

            if is_direct_load_style:
                print("Using load_state_dict for non-quantized safetensors (direct state_dict loading path).")
                model.to(dtype=torch.float16, device=device) 

                tensors_processed = {}
                for k in all_keys:
                    cleaned_key = k
                    if cleaned_key.startswith("model."):
                        cleaned_key = cleaned_key[len("model."):]

                    tensor = get_tensor(k).to(dtype=torch.float16, device=device)
                    tensors_processed[cleaned_key] = tensor
                model.load_state_dict(tensors_processed, strict=False)
            else:
                print("Using _load_weights for non-quantized safetensors (custom mapping path).")

                name_map = {key.replace("._orig_mod", ""): key for key in all_keys}
                
                def get_tensor_for_load_weights(name: str) -> torch.Tensor:
                    source_key = name_map.get(name)
                    if source_key is None:
                        # This case should ideally not happen if _load_weights uses correct keys its a fallback
                        raise KeyError(f"Key {name} not found in safetensor after mapping. Available mapped keys: {list(name_map.keys())}")
                    return get_tensor(source_key).to(dtype=torch.float16, device=device)

                _load_weights(get_tensor_for_load_weights, model)
    
    model.to(device)
    print("✓ Successfully loaded weights from safetensors file!")


def load_weights_from_pt(weights_file: str, model: nn.Module) -> None:
    """Load weights from a PyTorch (.pt) file into the model instance."""
    device = next(model.parameters()).device
    print(f"Loading .pt file to CPU first to conserve GPU memory (target device: {device})...")

    state_dict_on_cpu = torch.load(weights_file, map_location='cpu', weights_only=True)

    is_quantized = any('.qweight' in key or '_quantized' in key or 'quant.' in key for key in state_dict_on_cpu.keys())
    if is_quantized:
        print("Quantized model detected from .pt file keys.")

    if hasattr(model, 'text') and model.text is None:
        text_dtype = torch.int8 if is_quantized else torch.float16
        print(f"Building text model with dtype: {text_dtype} for .pt loading.")
        model.text = build_text_model(model.config.text, dtype=text_dtype)
        model.text.to(device)

        if hasattr(model, 'setup_caches_flag') and model.setup_caches_flag:
            model._setup_caches()
            print("Caches set up for text model.")
    elif hasattr(model, 'text') and model.text is not None:
        current_text_dtype = next(model.text.parameters()).dtype
        print(f"Text model already exists. Current text model dtype: {current_text_dtype}")
        if not is_quantized and current_text_dtype != torch.float16:
            print(f"Converting existing text model to float16.")
            model.text.to(dtype=torch.float16)

    if not is_quantized:
        print(
            "Model is not quantized. Loading weights from PyTorch file using _load_weights. This may take a while, please be patient."
        )

        processed_tensors = {
            k.replace("._orig_mod", ""): v.to(dtype=torch.float16)
            for k, v in state_dict_on_cpu.items()
        }
        _load_weights(lambda x: processed_tensors[x], model)
        del processed_tensors # Clean up intermediate dictionary

    else: # Quantized path
        print("Loading state_dict (from CPU) into model (on device) for quantized model...")

        model.load_state_dict(state_dict_on_cpu, strict=False)

    del state_dict_on_cpu
    gc.collect()
    print("Cleaned up original CPU state_dict from .pt loading.")


def load_weights_into_model(weights_file: str, model: nn.Module) -> nn.Module:
    """
    Main function to load weights into a model.
    Determines file type and calls the appropriate loading function.
    """
    device = next(iter(model.parameters())).device
    print(f"Starting weight loading process for model on {device}...")

    if weights_file.endswith('.pt'):
        load_weights_from_pt(weights_file, model)
    elif weights_file.endswith('.safetensors'):
        print("Loading .safetensors file...")
        load_weights_from_safetensors(weights_file, model)
    else:
        print(f"Unsupported weights file format: {weights_file}. Please use .pt or .safetensors.")
        return model 

    print("✓✓ Overall weight loading process complete!")
    return model

