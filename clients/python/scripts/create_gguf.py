# This utility script is used to convert the SafeTensors model to a GGUF file,
# which can be used to perform inference using the moondream client libraries.
# We use the GGUF format instead of performing inference directly using weights
# loaded from the safetensors file for two reasons:
#
# 1. We need to support client libraries in multiple languages, and the GGUF
#    format makes it easier for us to have a shared library in the future.
# 2. We can add additional metadata, allowing us to distribute a single model
#    file instead of needing to distribute a separate config.json along with
#    the weights.

from typing import Union
from numpy.typing import DTypeLike

import argparse
import numpy as np

from gguf import GGUFWriter, TokenType, SpecialVocab
from safetensors import safe_open
from transformers import AutoTokenizer

MODEL_ARCH = "md-vl-0"


class TensorReader:
    def __init__(self, tensors):
        self.tensors = tensors

    def get(self, name: str, dtype: DTypeLike):
        return self.tensors.get_tensor(name).astype(dtype)


def add_tokenizer(gguf_out: GGUFWriter, tokenizer_name: str) -> None:
    """
    Add a tokenizer to the GGUF file.

    Args:
        gguf_out (GGUFWriter): The GGUF writer object to add the tokenizer to.
        tokenizer_name (str): The name or path of the tokenizer to add.

    Returns:
        None
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    reverse_vocab = {id_: encoded_tok for encoded_tok, id_ in tokenizer.vocab.items()}
    added_vocab = tokenizer.get_added_vocab()
    vocab_size = 51200
    tokens = []
    toktypes = []

    for i in range(vocab_size):
        if i not in reverse_vocab:
            pad_token = f"[PAD{i}]".encode("utf-8")
            tokens.append(bytearray(pad_token))
            toktypes.append(TokenType.USER_DEFINED)
        elif reverse_vocab[i] in added_vocab:
            tokens.append(reverse_vocab[i])
            if tokenizer.added_tokens_decoder[i].special:
                toktypes.append(TokenType.CONTROL)
            else:
                toktypes.append(TokenType.USER_DEFINED)
        else:
            tokens.append(reverse_vocab[i])
            toktypes.append(TokenType.NORMAL)

    gguf_out.add_tokenizer_model("gpt2")
    gguf_out.add_token_list(tokens)
    gguf_out.add_token_types(toktypes)

    special_vocab = SpecialVocab(args.tokenizer, load_merges=True)
    special_vocab.add_to_gguf(gguf_out)


def add_vision_proj(gguf_out: GGUFWriter, tensors: TensorReader) -> None:
    """
    Add the vision projection tensors to the GGUF file.

    Args:
        gguf_out (GGUFWriter): The GGUF writer object to add the tensors to.
        tensors (TensorReader): object containing the original model weights.

    Returns:
        None
    """
    gguf_out.add_tensor(
        "vision_proj.mlp.fc1.weight",
        tensors.get("vision_encoder.projection.mlp.fc1.weight", np.float16),
    )
    gguf_out.add_tensor(
        "vision_proj.mlp.fc1.bias",
        tensors.get("vision_encoder.projection.mlp.fc1.bias", np.float32),
    )
    gguf_out.add_tensor(
        "vision_proj.mlp.fc2.weight",
        tensors.get("vision_encoder.projection.mlp.fc2.weight", np.float16),
    )
    gguf_out.add_tensor(
        "vision_proj.mlp.fc2.bias",
        tensors.get("vision_encoder.projection.mlp.fc2.bias", np.float32),
    )


def add_vision_encoder(gguf_out: GGUFWriter, tensors: TensorReader) -> None:
    """
    Add the vision encoder tensors to the GGUF file.

    Args:
        gguf_out (GGUFWriter): The GGUF writer object to add the tensors to.
        tensors (TensorReader): object containing the original model weights.

    Returns:
        None
    """
    gguf_out.add_tensor(
        "vision_enc.position_embd.weight",
        tensors.get("vision_encoder.encoder.model.visual.pos_embed", np.float16),
    )
    gguf_out.add_tensor(
        "vision_enc.patch_embd.weight",
        tensors.get(
            "vision_encoder.encoder.model.visual.patch_embed.linear.weight", np.float16
        ).reshape(1152, 3, 14, 14),
    )
    gguf_out.add_tensor(
        "vision_enc.patch_embd.bias",
        tensors.get(
            "vision_encoder.encoder.model.visual.patch_embed.linear.bias", np.float32
        ),
    )

    gguf_out.add_tensor(
        "vision_enc.output_norm.weight",
        tensors.get("vision_encoder.encoder.model.visual.norm.weight", np.float32),
    )
    gguf_out.add_tensor(
        "vision_enc.output_norm.bias",
        tensors.get("vision_encoder.encoder.model.visual.norm.bias", np.float32),
    )

    # Helper function to avoid duplicating the tensor name prefix.
    def blk_tensor(i, name, dtype):
        return tensors.get(
            f"vision_encoder.encoder.model.visual.blocks.{i}.{name}", dtype
        )

    def add_tensor(blk_id, gguf_id=None):
        if gguf_id is None:
            gguf_id = blk_id

        qkv_w = blk_tensor(blk_id, "attn.qkv.weight", np.float16)
        qkv_b = blk_tensor(blk_id, "attn.qkv.bias", np.float32)

        gguf_out.add_tensor(f"vision_enc.blk.{gguf_id}.attn_qkv.weight", qkv_w)
        gguf_out.add_tensor(f"vision_enc.blk.{gguf_id}.attn_qkv.bias", qkv_b)

        gguf_out.add_tensor(
            f"vision_enc.blk.{gguf_id}.attn_out.weight",
            blk_tensor(blk_id, "attn.proj.weight", np.float16),
        )
        gguf_out.add_tensor(
            f"vision_enc.blk.{gguf_id}.attn_out.bias",
            blk_tensor(blk_id, "attn.proj.bias", np.float32),
        )

        gguf_out.add_tensor(
            f"vision_enc.blk.{gguf_id}.ln1.weight",
            blk_tensor(blk_id, "norm1.weight", np.float32),
        )
        gguf_out.add_tensor(
            f"vision_enc.blk.{gguf_id}.ln1.bias",
            blk_tensor(blk_id, "norm1.bias", np.float32),
        )

        gguf_out.add_tensor(
            f"vision_enc.blk.{gguf_id}.mlp.fc1.weight",
            blk_tensor(blk_id, "mlp.fc1.weight", np.float16),
        )
        gguf_out.add_tensor(
            f"vision_enc.blk.{gguf_id}.mlp.fc1.bias",
            blk_tensor(blk_id, "mlp.fc1.bias", np.float32),
        )
        gguf_out.add_tensor(
            f"vision_enc.blk.{gguf_id}.mlp.fc2.weight",
            blk_tensor(blk_id, "mlp.fc2.weight", np.float16),
        )
        gguf_out.add_tensor(
            f"vision_enc.blk.{gguf_id}.mlp.fc2.bias",
            blk_tensor(blk_id, "mlp.fc2.bias", np.float32),
        )

        gguf_out.add_tensor(
            f"vision_enc.blk.{gguf_id}.ln2.weight",
            blk_tensor(blk_id, "norm2.weight", np.float32),
        )
        gguf_out.add_tensor(
            f"vision_enc.blk.{gguf_id}.ln2.bias",
            blk_tensor(blk_id, "norm2.bias", np.float32),
        )

    for i in range(27):
        add_tensor(i)


def add_text_model(gguf_out: GGUFWriter, tensors: TensorReader) -> None:
    """
    Add the text model tensors to the GGUF file.

    Args:
        gguf_out (GGUFWriter): The GGUF writer object to add the tensors to.
        tensors (TensorReader): object containing the original model weights.

    Returns:
        None
    """

    gguf_out.add_tensor(
        "text.token_embd.weight",
        tensors.get("text_model.transformer.embd.wte.weight", np.float16),
    )

    for i in range(24):
        gguf_out.add_tensor(
            f"text.blk.{i}.ln.weight",
            tensors.get(f"text_model.transformer.h.{i}.ln.weight", np.float32),
        )
        gguf_out.add_tensor(
            f"text.blk.{i}.ln.bias",
            tensors.get(f"text_model.transformer.h.{i}.ln.bias", np.float32),
        )

        gguf_out.add_tensor(
            f"text.blk.{i}.attn_qkv.weight",
            tensors.get(f"text_model.transformer.h.{i}.mixer.Wqkv.weight", np.float16),
        )
        gguf_out.add_tensor(
            f"text.blk.{i}.attn_qkv.bias",
            tensors.get(f"text_model.transformer.h.{i}.mixer.Wqkv.bias", np.float32),
        )
        gguf_out.add_tensor(
            f"text.blk.{i}.attn_out.weight",
            tensors.get(
                f"text_model.transformer.h.{i}.mixer.out_proj.weight", np.float16
            ),
        )
        gguf_out.add_tensor(
            f"text.blk.{i}.attn_out.bias",
            tensors.get(
                f"text_model.transformer.h.{i}.mixer.out_proj.bias", np.float32
            ),
        )

        gguf_out.add_tensor(
            f"text.blk.{i}.mlp.fc1.weight",
            tensors.get(f"text_model.transformer.h.{i}.mlp.fc1.weight", np.float16),
        )
        gguf_out.add_tensor(
            f"text.blk.{i}.mlp.fc1.bias",
            tensors.get(f"text_model.transformer.h.{i}.mlp.fc1.bias", np.float32),
        )
        gguf_out.add_tensor(
            f"text.blk.{i}.mlp.fc2.weight",
            tensors.get(f"text_model.transformer.h.{i}.mlp.fc2.weight", np.float16),
        )
        gguf_out.add_tensor(
            f"text.blk.{i}.mlp.fc2.bias",
            tensors.get(f"text_model.transformer.h.{i}.mlp.fc2.bias", np.float32),
        )

    gguf_out.add_tensor(
        "text.output_norm.weight",
        tensors.get("text_model.lm_head.ln.weight", np.float32),
    )
    gguf_out.add_tensor(
        "text.output_norm.bias",
        tensors.get("text_model.lm_head.ln.bias", np.float32),
    )
    gguf_out.add_tensor(
        "text.lm_head.weight",
        tensors.get("text_model.lm_head.linear.weight", np.float16),
    )
    gguf_out.add_tensor(
        "text.lm_head.bias",
        tensors.get("text_model.lm_head.linear.bias", np.float32),
    )


def add_region_model(gguf_out: GGUFWriter, tensors: TensorReader) -> None:
    """
    Add the region model tensors to the GGUF file.

    Args:
        gguf_out (GGUFWriter): The GGUF writer object to add the tensors to.
        tensors (TensorReader): object containing the original model weights.

    Returns:
        None
    """

    gguf_out.add_tensor(
        "region.coordinate_features.weight",
        tensors.get("region_model.coordinate_features.weight", np.float32),
    )
    gguf_out.add_tensor(
        "region.size_features.weight",
        tensors.get("region_model.coordinate_features.weight", np.float32),
    )

    gguf_out.add_tensor(
        "region.coordinate_encoder.weight",
        tensors.get("region_model.coordinate_encoder.weight", np.float16),
    )
    gguf_out.add_tensor(
        "region.coordinate_encoder.bias",
        tensors.get("region_model.coordinate_encoder.bias", np.float32),
    )

    gguf_out.add_tensor(
        "region.size_encoder.weight",
        tensors.get("region_model.size_encoder.weight", np.float16),
    )
    gguf_out.add_tensor(
        "region.size_encoder.bias",
        tensors.get("region_model.size_encoder.bias", np.float32),
    )

    gguf_out.add_tensor(
        "region.coordinate_decoder.fc1.weight",
        tensors.get("region_model.coordinate_decoder.fc1.weight", np.float16),
    )
    gguf_out.add_tensor(
        "region.coordinate_decoder.fc1.bias",
        tensors.get("region_model.coordinate_decoder.fc1.bias", np.float32),
    )
    gguf_out.add_tensor(
        "region.coordinate_decoder.fc2.weight",
        tensors.get("region_model.coordinate_decoder.fc2.weight", np.float16),
    )
    gguf_out.add_tensor(
        "region.coordinate_decoder.fc2.bias",
        tensors.get("region_model.coordinate_decoder.fc2.bias", np.float32),
    )

    gguf_out.add_tensor(
        "region.size_decoder.fc1.weight",
        tensors.get("region_model.size_decoder.fc1.weight", np.float16),
    )
    gguf_out.add_tensor(
        "region.size_decoder.fc1.bias",
        tensors.get("region_model.size_decoder.fc1.bias", np.float32),
    )
    gguf_out.add_tensor(
        "region.size_decoder.fc2.weight",
        tensors.get("region_model.size_decoder.fc2.weight", np.float16),
    )
    gguf_out.add_tensor(
        "region.size_decoder.fc2.bias",
        tensors.get("region_model.size_decoder.fc2.bias", np.float32),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--tokenizer",
        required=True,
        help="tokenizer path (path or HuggingFace model name)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="model.gguf",
        help="GGUF output path",
    )
    parser.add_argument(
        "-m",
        "--model",
        required=True,
        help="path to the safetensors model",
    )
    args = parser.parse_args()

    # The `arch` string will be used to decide architecture support in client
    # libraries. Each version of the clients will ship with a list of supported
    # `arch` values, allowing us to throw an error and ask the user if they try
    # to run a version of the model that is not supported.
    gguf_out = GGUFWriter(args.output, arch=MODEL_ARCH)

    # 1 -> magic number that indicates weights will be in fp16.
    gguf_out.add_file_type(1)
    gguf_out.add_name("moondream-vl")

    tensors = TensorReader(safe_open(args.model, framework="np"))

    add_vision_proj(gguf_out, tensors)
    add_vision_encoder(gguf_out, tensors)
    add_text_model(gguf_out, tensors)
    add_region_model(gguf_out, tensors)
    add_tokenizer(gguf_out, args.tokenizer)

    gguf_out.write_header_to_file()
    gguf_out.write_kv_data_to_file()
    gguf_out.write_tensors_to_file()

    gguf_out.close()
