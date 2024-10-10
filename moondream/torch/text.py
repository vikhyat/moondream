import torch
from typing import Dict, Tuple, Optional, List
from torch.nn import functional as F

from .rope import apply_rotary_emb, precompute_freqs_cis
from .layers import layer_norm, linear, mlp
from .weights import TextModel, AttentionWeights, load_from_safetensors


def text_encoder(input_ids: torch.Tensor, w: TextModel):
    return F.embedding(input_ids, w.wte)


def attn_mask(pos, seq_len):
    """
    Create an attention mask that aligns with the bottom right of the
    attention matrix. For example, if q_len = 2 and kv_len = 5, we want the
    following:

         1 1 1 1 0
         1 1 1 1 1

    and not this, which is what we get by default if we just set is_causal.

         1 0 0 0 0
         1 1 0 0 0
    """
    mask = torch.ones(seq_len, pos + seq_len, dtype=torch.bool)
    mask[:, pos:] = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
    mask = mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dimensions
    return mask


def attn(
    x: torch.Tensor,
    w: AttentionWeights,
    freqs_cis: torch.Tensor,
    layer_kv_cache: Optional[torch.Tensor],
    i: int,
):
    bsz, q_len, d_model = x.shape
    pos = 0 if layer_kv_cache is None else layer_kv_cache.shape[3]

    n_heads, head_dim = w.n_heads, d_model // w.n_heads
    q, k, v = linear(x, w.qkv).chunk(3, dim=-1)

    q = q.view(bsz, q_len, n_heads, head_dim).transpose(1, 2)
    k = k.view(bsz, q_len, n_heads, head_dim).transpose(1, 2)
    v = v.view(bsz, q_len, n_heads, head_dim).transpose(1, 2)

    q_rot, q_pass = q.chunk(2, dim=-1)
    k_rot, k_pass = k.chunk(2, dim=-1)
    q_rot, k_rot = apply_rotary_emb(
        q_rot, k_rot, freqs_cis[pos : pos + q_len], torch.float32
    )
    q = torch.cat([q_rot, q_pass], dim=-1).to(torch.float16)
    k = torch.cat([k_rot, k_pass], dim=-1).to(torch.float16)

    if layer_kv_cache is not None:
        k = torch.cat([layer_kv_cache[0], k], dim=2)
        v = torch.cat([layer_kv_cache[1], v], dim=2)

    out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask(pos, q_len))
    out = out.transpose(1, 2).reshape(bsz, q_len, d_model)
    out = linear(out, w.proj)
    return out, torch.stack([k, v])


def text_decoder(
    inputs_embeds: torch.Tensor,
    w: TextModel,
    kv_cache: Dict[int, torch.Tensor],
    freqs_cis: torch.Tensor,
):
    hidden_BTC = inputs_embeds

    if 0 in kv_cache:  # i.e., not empty
        cached_len = kv_cache[0].size(3)
        hidden_BTC = hidden_BTC[:, cached_len:, :]

    for i, block in enumerate(w.blocks):
        l_in = layer_norm(hidden_BTC, block.ln)
        l_attn, kv_cache[i] = attn(
            l_in, block.attn, freqs_cis, kv_cache.get(i, None), i
        )
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
    freqs_cis = precompute_freqs_cis(32, 2048)

    input_ids = tokenizer(
        "<|endoftext|>One must imagine Sisyphus",
        return_tensors="pt",
    )["input_ids"]
    last_str = ""
    kv_cache = {}
    for _ in range(200):
        with torch.no_grad():
            inputs_embeds = text_encoder(input_ids, weights.text)
            logits, kv_cache = text_decoder(
                inputs_embeds, weights.text, kv_cache, freqs_cis
            )
            if args.sampler == "greedy":
                next_token = torch.argmax(logits, dim=-1)
            else:
                next_token = torch.multinomial(
                    F.softmax(logits, dim=-1), num_samples=1
                ).squeeze(0)
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)
            output_text = tokenizer.batch_decode(input_ids)[0]
            print(output_text[len(last_str) :], end="", flush=True)
            last_str = output_text
    print()
