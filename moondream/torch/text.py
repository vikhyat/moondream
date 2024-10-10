import torch
from typing import Dict, Tuple
from torch.nn import functional as F

from .rope import apply_rotary_emb, precompute_freqs_cis
from .layers import layer_norm, linear, mlp
from .weights import TextModel, AttentionWeights, load_from_safetensors


def text_encoder(input_ids: torch.Tensor, w: TextModel):
    return F.embedding(input_ids, w.wte)


def attn(x: torch.Tensor, w: AttentionWeights, freqs_cis: torch.Tensor):
    bsz, q_len, d_model = x.shape
    n_heads, head_dim = w.n_heads, d_model // w.n_heads
    q, k, v = linear(x, w.qkv).chunk(3, dim=-1)

    q = q.view(bsz, q_len, n_heads, head_dim).transpose(1, 2)
    k = k.view(bsz, q_len, n_heads, head_dim).transpose(1, 2)
    v = v.view(bsz, q_len, n_heads, head_dim).transpose(1, 2)

    q_rot, q_pass = q.chunk(2, dim=-1)
    k_rot, k_pass = k.chunk(2, dim=-1)
    q_rot, k_rot = apply_rotary_emb(q_rot, k_rot, freqs_cis, torch.float32)
    q = torch.cat([q_rot, q_pass], dim=-1).to(torch.float16)
    k = torch.cat([k_rot, k_pass], dim=-1).to(torch.float16)

    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    out = out.transpose(1, 2).reshape(bsz, q_len, d_model)

    out = linear(out, w.proj)
    return out


def text_decoder(
    inputs_embeds: torch.Tensor,
    kv_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
    w: TextModel,
):
    hidden_BTC = inputs_embeds
    freqs_cis = precompute_freqs_cis(32, 2048)

    for i, block in enumerate(w.blocks):
        l_in = layer_norm(hidden_BTC, block.ln)
        l_attn = attn(l_in, block.attn, freqs_cis[: inputs_embeds.shape[1]])
        l_mlp = mlp(l_in, block.mlp)
        hidden_BTC = hidden_BTC + l_attn + l_mlp

    # We only need to compute logits for the last token position.
    hidden_BC = hidden_BTC[:, -1, :]
    hidden_BC = layer_norm(hidden_BC, w.post_ln)
    logits = linear(hidden_BC, w.lm_head)

    return logits, kv_cache


if __name__ == "__main__":
    from transformers import AutoTokenizer
    from huggingface_hub import snapshot_download

    torch.set_default_device("mps")

    weights = load_from_safetensors(
        "/Users/vikhyat/.cache/huggingface/hub/models--vikhyatk--moondream-next/snapshots/db475d4ba0bdaee23bc230ab6f11f1ad211395de/model.safetensors"
    )
    tokenizer = AutoTokenizer.from_pretrained("vikhyatk/moondream2")

    input_ids = tokenizer(
        "<|endoftext|>",
        return_tensors="pt",
    )["input_ids"]

    for _ in range(200):
        inputs_embeds = text_encoder(input_ids, weights.text)
        logits, kv_cache = text_decoder(inputs_embeds, {}, weights.text)
        # next_token = torch.argmax(logits, dim=-1)
        next_token = torch.multinomial(
            F.softmax(logits, dim=-1), num_samples=1
        ).squeeze(0)
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)
        print(tokenizer.batch_decode(input_ids))
