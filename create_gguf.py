import argparse

import numpy as np
from gguf import *
from safetensors import safe_open
from transformers import AutoTokenizer


def k(raw_key: str, arch: str) -> str:
    return raw_key.format(arch=arch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model", required=True, help="path to the safetensors model"
    )
    parser.add_argument(
        "-t",
        "--tokenizer",
        required=True,
        help="path to directory containing the tokenizer",
    )
    args = parser.parse_args()

    tensors = safe_open(args.model, framework="np")

    ### Vision encoder

    ftype = 1  # fp16

    fname_middle = "mmproj-"
    has_text_encoder = False
    has_llava_projector = True

    fname_out = "moondream2-mmproj-f16.gguf"
    fout = GGUFWriter(fname_out, arch="clip")

    fout.add_bool("clip.has_text_encoder", False)
    fout.add_bool("clip.has_vision_encoder", True)
    fout.add_bool("clip.has_llava_projector", True)
    fout.add_file_type(ftype)  # fp16

    model_name = "vikhyatk/moondream2"
    fout.add_name(model_name)
    fout.add_description("image encoder for " + model_name)
    fout.add_string("clip.projector_type", "mlp")

    # vision model hparams
    VISION = "clip.vision"
    fout.add_uint32("clip.vision.image_size", 378)
    fout.add_uint32("clip.vision.patch_size", 14)
    fout.add_uint32(k(KEY_EMBEDDING_LENGTH, VISION), 1152)
    fout.add_uint32(k(KEY_FEED_FORWARD_LENGTH, VISION), 4304)
    fout.add_uint32("clip.vision.projection_dim", 2048)
    fout.add_uint32(k(KEY_ATTENTION_HEAD_COUNT, VISION), 16)
    fout.add_float32(k(KEY_ATTENTION_LAYERNORM_EPS, VISION), 1e-6)
    fout.add_uint32(k(KEY_BLOCK_COUNT, VISION), 27 + 1)

    fout.add_array("clip.vision.image_mean", [0.5, 0.5, 0.5])
    fout.add_array("clip.vision.image_std", [0.5, 0.5, 0.5])
    fout.add_bool("clip.use_gelu", True)  # using regular GELU instead of quick

    # vision projection
    fout.add_tensor(
        "mm.0.weight",
        tensors.get_tensor("vision_encoder.projection.mlp.fc1.weight").astype(
            np.float16
        ),
    )
    fout.add_tensor(
        "mm.0.bias",
        tensors.get_tensor("vision_encoder.projection.mlp.fc1.bias").astype(np.float32),
    )
    fout.add_tensor(
        "mm.2.weight",
        tensors.get_tensor("vision_encoder.projection.mlp.fc2.weight").astype(
            np.float16
        ),
    )
    fout.add_tensor(
        "mm.2.bias",
        tensors.get_tensor("vision_encoder.projection.mlp.fc2.bias").astype(np.float32),
    )

    # encoder (siglip)
    fout.add_tensor(
        "v.position_embd.weight",
        tensors.get_tensor("vision_encoder.encoder.model.visual.pos_embed").astype(
            np.float16
        ),
    )
    fout.add_tensor(
        "v.patch_embd.weight",
        tensors.get_tensor(
            "vision_encoder.encoder.model.visual.patch_embed.linear.weight"
        )
        .reshape(1152, 3, 14, 14)
        .astype(np.float16),
    )
    fout.add_tensor(
        "v.patch_embd.bias",
        tensors.get_tensor(
            "vision_encoder.encoder.model.visual.patch_embed.linear.bias"
        ).astype(np.float32),
    )

    fout.add_tensor(
        "v.post_ln.weight",
        tensors.get_tensor("vision_encoder.encoder.model.visual.norm.weight").astype(
            np.float32
        ),
    )
    fout.add_tensor(
        "v.post_ln.bias",
        tensors.get_tensor("vision_encoder.encoder.model.visual.norm.bias").astype(
            np.float32
        ),
    )

    def blk_tensor(i, name):
        return tensors.get_tensor(
            rf"vision_encoder.encoder.model.visual.blocks.{i}.{name}"
        )

    def add_tensor(blk_id, gguf_id=None):
        if gguf_id is None:
            gguf_id = blk_id

        qkv_w = blk_tensor(blk_id, "attn.qkv.weight")
        qkv_b = blk_tensor(blk_id, "attn.qkv.bias")
        split_size = qkv_w.shape[0] // 3
        q_w, k_w, v_w = np.split(qkv_w, [split_size, 2 * split_size], axis=0)
        q_b, k_b, v_b = np.split(qkv_b, [split_size, 2 * split_size], axis=0)

        fout.add_tensor(f"v.blk.{gguf_id}.attn_q.weight", q_w.astype(np.float16))
        fout.add_tensor(f"v.blk.{gguf_id}.attn_q.bias", q_b.astype(np.float32))
        fout.add_tensor(f"v.blk.{gguf_id}.attn_k.weight", k_w.astype(np.float16))
        fout.add_tensor(f"v.blk.{gguf_id}.attn_k.bias", k_b.astype(np.float32))
        fout.add_tensor(f"v.blk.{gguf_id}.attn_v.weight", v_w.astype(np.float16))
        fout.add_tensor(f"v.blk.{gguf_id}.attn_v.bias", v_b.astype(np.float32))

        fout.add_tensor(
            f"v.blk.{gguf_id}.attn_out.weight",
            blk_tensor(blk_id, "attn.proj.weight").astype(np.float16),
        )
        fout.add_tensor(
            f"v.blk.{gguf_id}.attn_out.bias",
            blk_tensor(blk_id, "attn.proj.bias").astype(np.float32),
        )

        fout.add_tensor(
            f"v.blk.{gguf_id}.ln1.weight",
            blk_tensor(blk_id, "norm1.weight").astype(np.float32),
        )
        fout.add_tensor(
            f"v.blk.{gguf_id}.ln1.bias",
            blk_tensor(blk_id, "norm1.bias").astype(np.float32),
        )

        fout.add_tensor(
            f"v.blk.{gguf_id}.ffn_down.weight",
            blk_tensor(blk_id, "mlp.fc1.weight").astype(np.float16),
        )
        fout.add_tensor(
            f"v.blk.{gguf_id}.ffn_down.bias",
            blk_tensor(blk_id, "mlp.fc1.bias").astype(np.float32),
        )
        fout.add_tensor(
            f"v.blk.{gguf_id}.ffn_up.weight",
            blk_tensor(blk_id, "mlp.fc2.weight").astype(np.float16),
        )
        fout.add_tensor(
            f"v.blk.{gguf_id}.ffn_up.bias",
            blk_tensor(blk_id, "mlp.fc2.bias").astype(np.float32),
        )

        fout.add_tensor(
            f"v.blk.{gguf_id}.ln2.weight",
            blk_tensor(blk_id, "norm2.weight").astype(np.float32),
        )
        fout.add_tensor(
            f"v.blk.{gguf_id}.ln2.bias",
            blk_tensor(blk_id, "norm2.bias").astype(np.float32),
        )

    for i in range(27):
        add_tensor(i)

    # Duplicate the last block (llava-cli skips over this)
    add_tensor(26, 27)

    fout.write_header_to_file()
    fout.write_kv_data_to_file()
    fout.write_tensors_to_file()
    fout.close()

    ### Text model

    # general GGUF init
    fname_out = "moondream2-text-model-f16.gguf"
    fout = GGUFWriter(fname_out, arch="phi2")
    ftype = 1

    fout.add_name("moondream2")
    fout.add_context_length(2048)
    fout.add_embedding_length(2048)
    fout.add_feed_forward_length(4 * 2048)
    fout.add_block_count(24)
    fout.add_head_count(32)
    fout.add_head_count_kv(32)
    fout.add_layer_norm_eps(1e-5)
    fout.add_rope_dimension_count(32)
    fout.add_file_type(ftype)
    fout.add_add_bos_token(False)

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
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

    fout.add_tokenizer_model("gpt2")
    fout.add_token_list(tokens)
    fout.add_token_types(toktypes)

    special_vocab = SpecialVocab(args.tokenizer, load_merges=True)
    special_vocab.add_to_gguf(fout)

    fout.add_tensor(
        "token_embd.weight",
        tensors.get_tensor("text_model.transformer.embd.wte.weight").astype(np.float16),
    )

    for i in range(24):
        fout.add_tensor(
            f"blk.{i}.attn_norm.weight",
            tensors.get_tensor(f"text_model.transformer.h.{i}.ln.weight").astype(
                np.float32
            ),
        )
        fout.add_tensor(
            f"blk.{i}.attn_norm.bias",
            tensors.get_tensor(f"text_model.transformer.h.{i}.ln.bias").astype(
                np.float32
            ),
        )

        fout.add_tensor(
            f"blk.{i}.attn_qkv.weight",
            tensors.get_tensor(
                f"text_model.transformer.h.{i}.mixer.Wqkv.weight"
            ).astype(np.float16),
        )
        fout.add_tensor(
            f"blk.{i}.attn_qkv.bias",
            tensors.get_tensor(f"text_model.transformer.h.{i}.mixer.Wqkv.bias").astype(
                np.float32
            ),
        )
        fout.add_tensor(
            f"blk.{i}.attn_output.weight",
            tensors.get_tensor(
                f"text_model.transformer.h.{i}.mixer.out_proj.weight"
            ).astype(np.float16),
        )
        fout.add_tensor(
            f"blk.{i}.attn_output.bias",
            tensors.get_tensor(
                f"text_model.transformer.h.{i}.mixer.out_proj.bias"
            ).astype(np.float32),
        )

        fout.add_tensor(
            f"blk.{i}.ffn_up.weight",
            tensors.get_tensor(f"text_model.transformer.h.{i}.mlp.fc1.weight").astype(
                np.float16
            ),
        )
        fout.add_tensor(
            f"blk.{i}.ffn_up.bias",
            tensors.get_tensor(f"text_model.transformer.h.{i}.mlp.fc1.bias").astype(
                np.float32
            ),
        )
        fout.add_tensor(
            f"blk.{i}.ffn_down.weight",
            tensors.get_tensor(f"text_model.transformer.h.{i}.mlp.fc2.weight").astype(
                np.float16
            ),
        )
        fout.add_tensor(
            f"blk.{i}.ffn_down.bias",
            tensors.get_tensor(f"text_model.transformer.h.{i}.mlp.fc2.bias").astype(
                np.float32
            ),
        )

    fout.add_tensor(
        "output_norm.weight",
        tensors.get_tensor("text_model.lm_head.ln.weight").astype(np.float32),
    )
    fout.add_tensor(
        "output_norm.bias",
        tensors.get_tensor("text_model.lm_head.ln.bias").astype(np.float32),
    )
    fout.add_tensor(
        "output.weight",
        tensors.get_tensor("text_model.lm_head.linear.weight").astype(np.float16),
    )
    fout.add_tensor(
        "output.bias",
        tensors.get_tensor("text_model.lm_head.linear.bias").astype(np.float32),
    )

    # save gguf
    fout.write_header_to_file()
    fout.write_kv_data_to_file()
    fout.write_tensors_to_file()
    fout.close()
